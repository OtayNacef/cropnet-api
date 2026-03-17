"""Request / response schemas for CropNet API."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG image")
    locale: str       = Field("fr", description="ar | fr | en")
    crop_hint: str    = Field("", description="Optional crop hint (olive, tomato, …)")
    scan_id: str      = Field("", description="Optional client-side scan ID")


class FeedbackRequest(BaseModel):
    scan_id: str        = Field(..., description="Scan ID from /predict response")
    correct_label: str  = Field(..., description="Correct raw label")
    user_confirmed: bool = Field(True)


# ── Response fragments ────────────────────────────────────────────────────────

class PredictionItem(BaseModel):
    label: str
    display_name: str
    confidence: float           # raw (temperature-scaled TTA avg)
    calibrated_confidence: float | None = None   # placeholder for post-hoc calibration


class PredictResponse(BaseModel):
    status: str                      = "ok"
    request_id: str                  = ""
    language: str                    = "fr"

    # detection
    crop_detected: str | None        = None
    primary_prediction: PredictionItem | None = None
    top_predictions: list[PredictionItem] = []

    # routing
    model_used: str                  = ""
    model_type: str                  = "general"       # general | specialist
    routing_reason: str              = ""

    # confidence assessment
    is_low_confidence: bool          = False
    confidence_level: str            = "moderate"       # high | moderate | low | very_low
    below_threshold: bool            = False

    # advisory
    advisory_only: bool              = True
    recommended_action: str          = ""
    disclaimer: str                  = ""

    # image quality
    image_quality_warnings: list[str] = []

    # metadata
    severity: str                    = "unknown"
    is_healthy: bool                 = False
    is_tunisia: bool                 = False
    crop_agreement: str              = ""
    inference_ms: int                = 0
