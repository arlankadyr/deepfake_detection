"""Checkpoint discovery, preprocessing, and inference for DeepfakeCheck."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

MODEL_BUILDERS = {
    "resnet50": "resnet50",
    "deit_small": "deit_small_patch16_224",
}

TRAINING_SIZES = (20_000, 10_000, 5_000, 2_000)


def discover_checkpoints(
    directory: Path | None = None,
) -> dict[str, list[dict[str, int]]]:
    """Return valid model/size/seed combinations found in the checkpoint folder."""
    checkpoint_dir = directory or CHECKPOINTS_DIR
    found: dict[str, set[tuple[int, int]]] = {name: set() for name in MODEL_BUILDERS}
    if not checkpoint_dir.is_dir():
        return {name: [] for name in MODEL_BUILDERS}

    for path in checkpoint_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".pt":
            continue
        for model_name in MODEL_BUILDERS:
            prefix = f"{model_name}_biggan_n"
            if not path.name.startswith(prefix) or "_seed" not in path.name:
                continue
            size_text, separator, seed_text = path.name[len(prefix) : -3].partition("_seed")
            if not separator or not size_text.isdigit() or not seed_text.isdigit():
                continue
            size = int(size_text)
            if size not in TRAINING_SIZES:
                continue
            found[model_name].add((size, int(seed_text)))

    return {
        model_name: [
            {"train_size": size, "seed": seed}
            for size, seed in sorted(pairs, key=lambda pair: (-pair[0], pair[1]))
        ]
        for model_name, pairs in found.items()
    }


def _torch_and_timm() -> tuple[Any, Any]:
    """Import the large model libraries only when model information is needed."""
    import timm
    import torch

    return torch, timm


def _create_model(model_name: str) -> Any:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    _, timm = _torch_and_timm()
    return timm.create_model(MODEL_BUILDERS[model_name], pretrained=False, num_classes=1)


@lru_cache(maxsize=2)
def parameter_count(model_name: str) -> int:
    """Compute the number of trainable and non-trainable model parameters."""
    model = _create_model(model_name)
    try:
        return sum(parameter.numel() for parameter in model.parameters())
    finally:
        del model


@lru_cache(maxsize=8)
def load_model(model_name: str, train_size: int, seed: int) -> tuple[Any, Any]:
    """Build and cache one model loaded from its matching student checkpoint."""
    torch, _ = _torch_and_timm()
    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_biggan_n{train_size}_seed{seed}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model(model_name)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image) -> Any:
    """Apply the documented 224x224 ImageNet preprocessing pipeline."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    pipeline = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return pipeline(image.convert("RGB"))


def predict_image(
    image: Image.Image,
    model_name: str,
    train_size: int,
    seed: int,
) -> dict[str, float | str]:
    """Run one checkpoint and return only a real model-derived result."""
    torch, _ = _torch_and_timm()
    model, device = load_model(model_name, train_size, seed)
    tensor = preprocess_image(image).unsqueeze(0).to(device)

    import time

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logit = model(tensor).reshape(-1)[0]
        probability = float(torch.sigmoid(logit).item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "model": model_name,
        "fake_probability": probability,
        "label": "AI-generated" if probability >= 0.5 else "Real",
        "inference_ms": elapsed_ms,
    }
