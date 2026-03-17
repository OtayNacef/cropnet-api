"""
CropNet API configuration.
All tunables in one place. Env-overridable.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR      = Path(os.getenv("MODELS_DIR", "/opt/cropnet/models"))
FEEDBACK_DIR    = Path(os.getenv("FEEDBACK_DIR", "/opt/cropnet/logs"))
LOCALES_DIR     = Path(__file__).parent / "locales"

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("CROPNET_API_KEY", "")

# ── Inference ─────────────────────────────────────────────────────────────────
IMG_SIZE           = int(os.getenv("IMG_SIZE", "256"))
TTA_CROPS          = int(os.getenv("TTA_CROPS", "5"))
TEMPERATURE        = float(os.getenv("TEMPERATURE", "1.4"))
ONNX_THREADS       = int(os.getenv("ONNX_THREADS", "4"))
MAX_UPLOAD_BYTES   = 20 * 1024 * 1024   # 20 MB

# ── Thresholds (tunable per deployment) ───────────────────────────────────────
GENERAL_THRESHOLD       = float(os.getenv("GENERAL_THRESHOLD", "0.40"))
SPECIALIST_THRESHOLD    = float(os.getenv("SPECIALIST_THRESHOLD", "0.45"))
LOW_CONF_THRESHOLD      = float(os.getenv("LOW_CONF_THRESHOLD", "0.25"))
ADVISORY_ONLY_THRESHOLD = float(os.getenv("ADVISORY_ONLY_THRESHOLD", "0.15"))

# Legacy alias
CONF_THRESHOLD = GENERAL_THRESHOLD

# ── Model naming ──────────────────────────────────────────────────────────────
# Production ONNX names: cropnet-{crop}-v{N}.onnx
# Internal lineage tracked in metadata.json beside each model
GENERAL_MODEL_VERSION = os.getenv("GENERAL_MODEL_VERSION", "v1")


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, "")
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    return default


# ── Specialist registry ───────────────────────────────────────────────────────
# Each specialist can be toggled via ENABLE_SPECIALIST_{KEY}=true/false env var
SPECIALISTS: dict[str, dict] = {
    "olive": {
        "tier": 1,
        "enabled": _env_bool("ENABLE_SPECIALIST_OLIVE", True),
        "conf": float(os.getenv("SPECIALIST_THRESHOLD_OLIVE", "0.45")),
        "img_size": 256,
        "desc": "Olive diseases: peacock spot, olive knot, aculus olearius",
        "version": "v1",
    },
    "date_palm": {
        "tier": 1,
        "enabled": _env_bool("ENABLE_SPECIALIST_DATE_PALM", True),
        "conf": float(os.getenv("SPECIALIST_THRESHOLD_DATE_PALM", "0.45")),
        "img_size": 256,
        "desc": "Date palm: bayoud, black scorch, brown leaf spot",
        "version": "v1",
    },
    "wheat": {
        "tier": 1,
        "enabled": _env_bool("ENABLE_SPECIALIST_WHEAT", True),
        "conf": float(os.getenv("SPECIALIST_THRESHOLD_WHEAT", "0.45")),
        "img_size": 256,
        "desc": "Wheat: septoria, yellow rust, healthy",
        "version": "v1",
    },
    "citrus": {
        "tier": 1,
        "enabled": _env_bool("ENABLE_SPECIALIST_CITRUS", True),
        "conf": float(os.getenv("SPECIALIST_THRESHOLD_CITRUS", "0.45")),
        "img_size": 256,
        "desc": "Citrus: canker, greening/HLB, healthy",
        "version": "v1",
    },
    "tomato": {
        "tier": 2,
        "enabled": _env_bool("ENABLE_SPECIALIST_TOMATO", False),
        "conf": float(os.getenv("SPECIALIST_THRESHOLD_TOMATO", "0.45")),
        "img_size": 256,
        "desc": "Tomato diseases (PlantVillage subset)",
        "version": "v1",
    },
    "pepper": {
        "tier": 2,
        "enabled": _env_bool("ENABLE_SPECIALIST_PEPPER", False),
        "conf": float(os.getenv("SPECIALIST_THRESHOLD_PEPPER", "0.50")),
        "img_size": 256,
        "desc": "Pepper: bacterial spot, healthy",
        "version": "v1",
    },
    "watermelon": {
        "tier": 2,
        "enabled": _env_bool("ENABLE_SPECIALIST_WATERMELON", False),
        "conf": float(os.getenv("SPECIALIST_THRESHOLD_WATERMELON", "0.50")),
        "img_size": 256,
        "desc": "Watermelon / cucurbit diseases",
        "version": "v1",
    },
}

# ── Crop family mapping (label prefix → specialist key) ──────────────────────
# Used by routing to infer which specialist to invoke from general model output
CROP_FAMILIES: dict[str, str] = {
    "Olive":      "olive",
    "Date Palm":  "date_palm",
    "Palm":       "date_palm",
    "Wheat":      "wheat",
    "Citrus":     "citrus",
    "Orange":     "citrus",
    "Tomato":     "tomato",
    "Pepper":     "pepper",
    "Squash":     "watermelon",
    "Apple":      "apple",
    "Corn":       "corn",
    "Grape":      "grape",
    "Peach":      "peach",
    "Potato":     "potato",
    "Strawberry": "strawberry",
    "Cherry":     "cherry",
    "Soybean":    "soybean",
    "Raspberry":  "raspberry",
    "Blueberry":  "blueberry",
}


def infer_crop_family(label: str) -> str | None:
    """Infer crop family from label prefix. Returns specialist key or None."""
    for prefix, family in CROP_FAMILIES.items():
        if label.startswith(prefix):
            return family
    return None


# ── Model paths ───────────────────────────────────────────────────────────────
# Production naming: cropnet-{key}-{version}.onnx

def specialist_onnx_path(key: str) -> Path:
    v = SPECIALISTS[key]["version"] if key in SPECIALISTS else "v1"
    return MODELS_DIR / key / f"cropnet-{key}-{v}.onnx"


def specialist_labels_path(key: str) -> Path:
    return MODELS_DIR / key / "labels.json"


def specialist_metadata_path(key: str) -> Path:
    return MODELS_DIR / key / "metadata.json"


def general_onnx_path() -> Path:
    return MODELS_DIR / "general" / f"cropnet-general-{GENERAL_MODEL_VERSION}.onnx"


def general_labels_path() -> Path:
    return MODELS_DIR / "general" / "labels.json"


def general_metadata_path() -> Path:
    return MODELS_DIR / "general" / "metadata.json"
