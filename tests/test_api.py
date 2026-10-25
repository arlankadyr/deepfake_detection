from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import inference, main


@pytest.fixture(autouse=True)
def isolated_project_paths(tmp_path, monkeypatch):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(main, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(inference, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(main, "RESULTS_FILE", results_dir / "metrics.json")
    inference.load_model.cache_clear()
    yield checkpoints_dir
    inference.load_model.cache_clear()


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 16), color=(120, 80, 40)).save(output, format="PNG")
    return output.getvalue()


def test_health_endpoint_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_inventory_reports_architectures_and_parameters(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    models = {item["model"]: item for item in response.json()["models"]}
    assert models["resnet50"]["architecture"] == "resnet50"
    assert models["deit_small"]["architecture"] == "deit_small_patch16_224"
    assert models["resnet50"]["parameter_count"] > 0
    assert models["deit_small"]["parameter_count"] > 0
    assert models["resnet50"]["training_data"] == "GenImage BigGAN subset"
    assert models["resnet50"]["available_runs"] == []


def test_home_and_results_pages_load(client):
    home = client.get("/")
    results = client.get("/results")
    assert home.status_code == 200
    assert "DeepfakeCheck" in home.text
    assert results.status_code == 200
    assert "Experiment results" in results.text


def test_predict_without_checkpoint_returns_503(client):
    response = client.post(
        "/api/predict",
        files={"file": ("small.png", png_bytes(), "image/png")},
        data={"model": "resnet50", "train_size": "20000", "seed": "42"},
    )
    assert response.status_code == 503
    assert "Model not trained yet" in response.json()["detail"]


def test_predict_with_random_test_checkpoint_returns_probability(client, isolated_project_paths):
    import timm
    import torch

    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    checkpoint = isolated_project_paths / "resnet50_biggan_n20000_seed42.pt"
    torch.save(model.state_dict(), checkpoint)
    del model

    response = client.post(
        "/api/predict",
        files={"file": ("small.png", png_bytes(), "image/png")},
        data={"model": "resnet50", "train_size": "20000", "seed": "42"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert 0 <= result["predictions"][0]["fake_probability"] <= 1
    assert result["predictions"][0]["label"] in {"Real", "AI-generated"}
    assert result["models_agree"] is True
    assert not list(isolated_project_paths.glob("*.png"))


def test_text_file_renamed_to_png_is_rejected(client):
    response = client.post(
        "/api/predict",
        files={"file": ("misleading.png", b"this is plain text", "image/png")},
        data={"model": "resnet50"},
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_file_over_ten_megabytes_is_rejected(client):
    oversized = b"0" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/predict",
        files={"file": ("large.png", oversized, "image/png")},
        data={"model": "resnet50"},
    )
    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_results_endpoint_returns_404_when_metrics_are_missing(client):
    response = client.get("/api/results")
    assert response.status_code == 404
    assert response.json()["detail"] == "No experiment results yet."
