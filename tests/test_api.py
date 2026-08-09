"""
Unit tests for the FastAPI inference service.
Covers the REST endpoints required by M2 and M4 (health, predict, metrics).
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set model path before importing app so tests use the real trained model
import os
os.environ["MODEL_PATH"] = "models/model.pt"
os.environ["DEVICE"] = "cpu"

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        # One warm-up prediction ensures model is loaded
        c.get("/health")
        yield c


def _make_image(color=(100, 150, 200)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestAPIHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["device"] == "cpu"


class TestAPIPredict:
    def test_predict_returns_valid_response(self, client):
        buf = _make_image()
        resp = client.post(
            "/predict", files={"file": ("cat.jpg", buf, "image/jpeg")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("cat", "dog")
        assert data["class_id"] in (0, 1)
        assert 0 <= data["confidence"] <= 1
        assert set(data["probabilities"].keys()) == {"cat", "dog"}

    def test_predict_rejects_bad_content_type(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_predict_batch(self, client):
        buf1, buf2 = _make_image((100, 100, 100)), _make_image((200, 0, 0))
        resp = client.post(
            "/predict_batch",
            files=[
                ("files", ("a.jpg", buf1, "image/jpeg")),
                ("files", ("b.jpg", buf2, "image/jpeg")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        for pred in data["predictions"]:
            assert pred["prediction"] in ("cat", "dog")


class TestAPIMetrics:
    def test_metrics_content(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "cats-vs-dogs-classifier"
        assert data["total_requests"] >= 1
        assert "endpoints" in data

    def test_monitor_performance(self, client):
        resp = client.get("/monitor/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert "accuracy" in data
        assert "drift" in data
