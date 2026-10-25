"""FastAPI routes for the DeepfakeCheck demonstration app."""

from __future__ import annotations

import json
import logging
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartParser

from app.inference import MODEL_BUILDERS, CHECKPOINTS_DIR, discover_checkpoints, parameter_count, predict_image
from app.schemas import ModelsResponse, PredictResponse

logger = logging.getLogger("deepfakecheck")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
RESULTS_FILE = PROJECT_ROOT / "results" / "metrics.json"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BODY_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
FORMAT_CONTENT_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MODEL_DETAILS = {
    "resnet50": {
        "display_name": "ResNet-50",
        "architecture": "resnet50",
    },
    "deit_small": {
        "display_name": "DeiT-S",
        "architecture": "deit_small_patch16_224",
    },
}

# Starlette uses SpooledTemporaryFile for multipart uploads. Keep the spool
# threshold above the maximum request body so no accepted upload reaches disk.
MultiPartParser.spool_max_size = MAX_REQUEST_BODY_BYTES + 1


class InMemoryUploadLimitMiddleware:
    """Bound and buffer prediction requests in RAM before multipart parsing."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/predict"
        ):
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST_BODY_BYTES:
                response_body = b'{"detail":"Request body is too large; image uploads must be 10 MB or smaller."}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(response_body)).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": response_body})
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        buffered_body = bytes(body)
        already_sent = False

        async def replay_body() -> dict[str, Any]:
            nonlocal already_sent
            if already_sent:
                return {"type": "http.disconnect"}
            already_sent = True
            return {"type": "http.request", "body": buffered_body, "more_body": False}

        await self.app(scope, replay_body, send)


def _combinations_for(models: list[str], runs: dict[str, list[dict[str, int]]]) -> set[tuple[int, int]]:
    combinations = [
        {(run["train_size"], run["seed"]) for run in runs.get(model_name, [])}
        for model_name in models
    ]
    if not combinations or any(not pairs for pairs in combinations):
        return set()
    return set.intersection(*combinations)


def _decode_upload(content_type: str | None, content: bytes) -> Image.Image:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty or corrupted.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as probe:
                detected_format = probe.format
                probe.verify()
            if detected_format not in FORMAT_CONTENT_TYPES:
                raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported.")
            if FORMAT_CONTENT_TYPES[detected_format] != content_type:
                raise HTTPException(status_code=400, detail="The file content does not match its image type.")
            with Image.open(BytesIO(content)) as decoded:
                decoded.load()
                return decoded.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(status_code=400, detail="The uploaded image is corrupted or invalid.") from None


@asynccontextmanager
async def lifespan(_: FastAPI):
    runs = discover_checkpoints(CHECKPOINTS_DIR)
    summary = ", ".join(f"{model_name}: {len(model_runs)} available" for model_name, model_runs in runs.items())
    logger.info("Checkpoint scan complete (%s).", summary)
    yield


app = FastAPI(title="DeepfakeCheck", description="A research model demonstration and results viewer.", lifespan=lifespan)
app.add_middleware(InMemoryUploadLimitMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/results", include_in_schema=False)
async def results_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "results.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models", response_model=ModelsResponse)
async def models() -> dict[str, Any]:
    available = discover_checkpoints(CHECKPOINTS_DIR)
    model_rows: list[dict[str, Any]] = []
    all_sizes: set[int] = set()

    for model_name, details in MODEL_DETAILS.items():
        runs = available[model_name]
        sizes = sorted({run["train_size"] for run in runs}, reverse=True)
        all_sizes.update(sizes)
        seeds_by_size = {
            str(size): sorted({run["seed"] for run in runs if run["train_size"] == size})
            for size in sizes
        }
        model_rows.append(
            {
                "model": model_name,
                **details,
                "parameter_count": parameter_count(model_name),
                "training_data": "GenImage BigGAN subset",
                "available_runs": runs,
                "training_sizes": sizes,
                "seeds_by_size": seeds_by_size,
            }
        )

    return {"models": model_rows, "training_sizes": sorted(all_sizes, reverse=True)}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Form("both"),
    train_size: int | None = Form(None),
    seed: int | None = Form(None),
) -> dict[str, Any]:
    selected_models = list(MODEL_BUILDERS) if model == "both" else [model]
    if model not in {"both", *MODEL_BUILDERS.keys()}:
        await file.close()
        raise HTTPException(status_code=422, detail="Choose both, resnet50, or deit_small.")

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image uploads must be 10 MB or smaller.")
        image = _decode_upload(file.content_type, content)
    finally:
        await file.close()

    available = discover_checkpoints(CHECKPOINTS_DIR)
    common_runs = _combinations_for(selected_models, available)
    if train_size is None:
        selected_sizes = [size for size, _ in common_runs]
        train_size = max(selected_sizes) if selected_sizes else None
    if train_size is None:
        raise HTTPException(status_code=503, detail="Model not trained yet: no checkpoint is available for the selected model.")

    matching_seeds = sorted(seed_value for size, seed_value in common_runs if size == train_size)
    if seed is None:
        seed = matching_seeds[0] if matching_seeds else None
    if seed is None or (train_size, seed) not in common_runs:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet: no checkpoint is available for the selected configuration (training size {train_size}, seed {seed}).",
        )

    predictions: list[dict[str, Any]] = []
    for model_name in selected_models:
        try:
            prediction = await run_in_threadpool(predict_image, image, model_name, train_size, seed)
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Model not trained yet: the selected checkpoint is missing.") from None
        except Exception:
            logger.exception("Inference failed for model %s at training size %s and seed %s.", model_name, train_size, seed)
            raise HTTPException(status_code=500, detail="The selected model checkpoint could not be loaded or used.") from None
        predictions.append(prediction)

    labels = {result["label"] for result in predictions}
    return {"predictions": predictions, "models_agree": len(labels) <= 1}


@app.get("/api/results", response_model=None)
async def experiment_results() -> JSONResponse | dict[str, Any]:
    if not RESULTS_FILE.is_file():
        return JSONResponse(status_code=404, content={"detail": "No experiment results yet."})
    try:
        with RESULTS_FILE.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except (json.JSONDecodeError, OSError):
        raise HTTPException(status_code=500, detail="The experiment results file could not be read.") from None
