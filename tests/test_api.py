"""Integration tests for CropNet API endpoints using FastAPI TestClient.

NOTE: These tests mock the ONNX models since we don't bundle model files in tests.
They verify endpoint contracts, auth, validation, and response schemas.
"""
import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Must set env before import
import os
os.environ.setdefault("MODELS_DIR", "/tmp/cropnet-test-models")
os.environ.setdefault("FEEDBACK_DIR", "/tmp/cropnet-test-feedback")
os.environ.setdefault("CROPNET_API_KEY", "test-key-12345")
for crop in ["OLIVE", "DATE_PALM", "WHEAT", "CITRUS", "TOMATO", "PEPPER", "WATERMELON"]:
    os.environ.setdefault(f"ENABLE_SPECIALIST_{crop}", "false")

from api.inference.general import GeneralModel, InferenceResult, Prediction
from api.routing import Router, RoutingDecision
from api.inference.calibration import CalibrationParams


def _make_test_image(w=256, h=256, fmt="JPEG") -> bytes:
    img = Image.new("RGB", (w, h), (34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _b64_image(w=256, h=256) -> str:
    return base64.b64encode(_make_test_image(w, h)).decode()


def _mock_router():
    gen_result = InferenceResult(
        top_k=[
            Prediction(0, "Olive___olive_peacock_spot", 0.72),
            Prediction(1, "Olive___Healthy", 0.15),
            Prediction(2, "Tomato___healthy", 0.08),
        ],
        crop_agreement=4, tta_crops=5, inference_ms=120,
    )
    r = MagicMock(spec=Router)
    r.route.return_value = RoutingDecision(
        general=gen_result, specialist=None,
        model_key="general", model_type="general",
        crop_family="olive", reason="No specialist loaded",
        general_calibration=CalibrationParams(), specialist_calibration=None,
    )
    r.specialists = {}
    return r


@pytest.fixture
def client():
    """Create a TestClient with mocked model loading — skip real lifespan."""
    from api import main as api_main
    from api import feedback as fb_mod
    from contextlib import asynccontextmanager

    # Replace lifespan to skip real model loading
    @asynccontextmanager
    async def _test_lifespan(_app):
        fb_mod.init()
        yield

    original_lifespan = api_main.app.router.lifespan_context
    api_main.app.router.lifespan_context = _test_lifespan

    mock_general = MagicMock(spec=GeneralModel)
    mock_general.num_classes = 89
    mock_router = _mock_router()

    api_main._general = mock_general
    api_main._router = mock_router

    with TestClient(api_main.app, raise_server_exceptions=False) as c:
        yield c

    api_main._general = None
    api_main._router = None
    api_main.app.router.lifespan_context = original_lifespan


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "5.0.0"

    def test_health_has_thresholds(self, client):
        data = client.get("/health").json()
        assert "thresholds" in data
        assert "general" in data["thresholds"]

    def test_health_has_specialists(self, client):
        data = client.get("/health").json()
        assert "specialists_loaded" in data


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_predict_without_key_rejected(self, client):
        resp = client.post("/predict", json={"image_base64": _b64_image(), "locale": "en"})
        assert resp.status_code == 403

    def test_predict_with_wrong_key_rejected(self, client):
        resp = client.post("/predict",
            json={"image_base64": _b64_image(), "locale": "en"},
            headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_predict_with_correct_key_accepted(self, client):
        resp = client.post("/predict",
            json={"image_base64": _b64_image(), "locale": "en"},
            headers={"X-API-Key": "test-key-12345"})
        assert resp.status_code == 200

    def test_health_no_auth_required(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


# ── Predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def _post(self, client, **kwargs):
        payload = {"image_base64": _b64_image(), "locale": "en", **kwargs}
        return client.post("/predict", json=payload, headers={"X-API-Key": "test-key-12345"})

    def test_valid_prediction(self, client):
        resp = self._post(client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["advisory_only"] is True
        assert len(data["disclaimer"]) > 0

    def test_response_has_required_fields(self, client):
        data = self._post(client).json()
        required = ["status", "request_id", "language", "crop_detected",
                     "primary_prediction", "top_predictions", "model_used",
                     "model_type", "routing_reason", "is_low_confidence",
                     "advisory_only", "image_quality_warnings",
                     "recommended_action", "disclaimer"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_prediction_item_has_confidence(self, client):
        data = self._post(client).json()
        pred = data["primary_prediction"]
        assert "confidence" in pred
        assert "calibrated_confidence" in pred
        assert "label" in pred
        assert "display_name" in pred

    def test_locale_passed_through(self, client):
        data = self._post(client, locale="ar").json()
        assert data["language"] == "ar"

    def test_crop_hint_passed(self, client):
        self._post(client, crop_hint="olive")
        # Just verify it doesn't crash — hint is passed to router


# ── Invalid Input ─────────────────────────────────────────────────────────────

class TestInvalidInput:
    def test_invalid_base64(self, client):
        resp = client.post("/predict",
            json={"image_base64": "not-valid-base64!!!", "locale": "en"},
            headers={"X-API-Key": "test-key-12345"})
        assert resp.status_code == 400

    def test_corrupt_image(self, client):
        raw = base64.b64encode(b"this is not an image").decode()
        resp = client.post("/predict",
            json={"image_base64": raw, "locale": "en"},
            headers={"X-API-Key": "test-key-12345"})
        assert resp.status_code == 400

    def test_tiny_image(self, client):
        raw = base64.b64encode(_make_test_image(w=16, h=16)).decode()
        resp = client.post("/predict",
            json={"image_base64": raw, "locale": "en"},
            headers={"X-API-Key": "test-key-12345"})
        assert resp.status_code == 400


# ── Feedback ──────────────────────────────────────────────────────────────────

class TestFeedbackEndpoint:
    def test_post_feedback(self, client):
        resp = client.post("/feedback",
            json={"scan_id": "test-scan-1", "correct_label": "Olive___Healthy", "user_confirmed": True},
            headers={"X-API-Key": "test-key-12345"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_feedback_stats(self, client):
        resp = client.get("/feedback/stats", headers={"X-API-Key": "test-key-12345"})
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data


# ── Models ────────────────────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/models", headers={"X-API-Key": "test-key-12345"})
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "general" in data["models"]
        # All 7 specialists should appear
        for crop in ["olive", "date_palm", "wheat", "citrus", "tomato", "pepper", "watermelon"]:
            assert crop in data["models"], f"Missing specialist: {crop}"

    def test_model_has_tier(self, client):
        data = client.get("/models", headers={"X-API-Key": "test-key-12345"}).json()
        assert data["models"]["olive"]["tier"] == 1
        assert data["models"]["tomato"]["tier"] == 2
