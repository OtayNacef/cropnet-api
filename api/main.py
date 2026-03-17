"""
CropNet API v5 — Multi-Model Crop Disease Diagnosis
====================================================
FastAPI + ONNX Runtime  ·  General + specialist routing
Author: Fellah (fellah.tn)
"""
from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import feedback
from .auth import require_api_key
from .config import (
    GENERAL_MODEL_VERSION,
    GENERAL_THRESHOLD,
    IMG_SIZE,
    MODELS_DIR,
    SPECIALISTS,
    TTA_CROPS,
    general_labels_path,
    general_onnx_path,
    infer_crop_family,
)
from .inference.calibration import (
    CalibrationParams,
    advisory_text,
    assess,
    disclaimer,
    is_below_threshold,
    load_general_calibration,
    load_specialist_calibration,
)
from .inference.general import GeneralModel
from .inference.metadata import (
    TUNISIA_LABELS,
    is_healthy,
    load_id2label,
    recommended_action,
    severity,
    translate,
)
from .inference.preprocess import quality_warnings, validate_image_bytes
from .inference.specialists import load_available_specialists
from .routing import Router, RoutingDecision
from .schemas import FeedbackRequest, PredictRequest, PredictResponse, PredictionItem

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("cropnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Globals ───────────────────────────────────────────────────────────────────
_router: Router | None = None
_general: GeneralModel | None = None
_general_cal: CalibrationParams | None = None
_VERSION = "5.0.0"


# ── Startup ───────────────────────────────────────────────────────────────────
def _boot() -> Router:
    global _general, _general_cal

    # General model — try clean name first, fall back to legacy
    g_onnx = general_onnx_path()
    g_labels = general_labels_path()
    if not g_onnx.exists():
        # Legacy flat layout: cropnet_v2.onnx / labels.json
        g_onnx = MODELS_DIR / "cropnet_v2.onnx"
        g_labels = MODELS_DIR / "labels.json"
        log.info(f"Using legacy model path: {g_onnx}")
    if not g_onnx.exists():
        raise RuntimeError(f"General model not found. Tried {general_onnx_path()} and {g_onnx}")

    id2label = load_id2label(g_labels)
    _general = GeneralModel(g_onnx, id2label, img_size=IMG_SIZE, name="general")
    _general.warmup()

    _general_cal = load_general_calibration()

    specialists = load_available_specialists()
    spec_cals = {key: load_specialist_calibration(key) for key in specialists}

    loaded_specs = list(specialists.keys())
    log.info(f"CropNet v{_VERSION} ready: general ({_general.num_classes} cls) + {len(specialists)} specialists {loaded_specs}")
    return Router(_general, specialists, _general_cal, spec_cals)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _router
    feedback.init()
    _router = _boot()
    yield


app = FastAPI(
    title="CropNet API",
    version=_VERSION,
    description="Multi-model crop disease diagnosis for Tunisian agriculture. Advisory only.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"], allow_headers=["*"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_response(dec: RoutingDecision, scan_id: str, lang: str, qw: list[str]) -> PredictResponse:
    result = dec.specialist if dec.model_type == "specialist" and dec.specialist else dec.general
    if not result.top_k:
        raise HTTPException(500, "Model returned no predictions")

    top = result.top_k[0]
    conf_level = assess(top.confidence)
    below = is_below_threshold(top.confidence, GENERAL_THRESHOLD)
    healthy = is_healthy(top.label)

    # Calibration (apply if available)
    cal = dec.specialist_calibration if dec.model_type == "specialist" else dec.general_calibration

    top_items = [
        PredictionItem(
            label=p.label,
            display_name=translate(p.label, lang),
            confidence=round(p.confidence, 4),
            calibrated_confidence=round(cal.calibrate(p.confidence), 4) if cal else None,
        )
        for p in result.top_k[:5]
    ]

    action = recommended_action(top.label, lang)
    if not action:
        action = advisory_text(conf_level, lang)

    # Model naming: clean production names
    if dec.model_type == "specialist" and dec.model_key in SPECIALISTS:
        model_name = f"cropnet-{dec.model_key}-{SPECIALISTS[dec.model_key]['version']}"
    else:
        model_name = f"cropnet-general-{GENERAL_MODEL_VERSION}"

    return PredictResponse(
        status="ok",
        request_id=scan_id,
        language=lang,
        crop_detected=dec.crop_family,
        primary_prediction=top_items[0] if top_items else None,
        top_predictions=top_items,
        model_used=model_name,
        model_type=dec.model_type,
        routing_reason=dec.reason,
        is_low_confidence=below,
        confidence_level=conf_level,
        below_threshold=below,
        advisory_only=True,
        recommended_action=action,
        disclaimer=disclaimer(lang),
        image_quality_warnings=qw,
        severity=severity(top.label, top.confidence),
        is_healthy=healthy,
        is_tunisia=top.label in TUNISIA_LABELS,
        crop_agreement=f"{result.crop_agreement}/{result.tta_crops}",
        inference_ms=result.inference_ms + (dec.general.inference_ms if dec.specialist else 0),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    spec_info = {}
    if _router:
        for key in _router.specialists:
            spec_info[key] = {"tier": SPECIALISTS[key]["tier"], "version": SPECIALISTS[key]["version"]}
    return {
        "status": "ok",
        "version": _VERSION,
        "general_model": str(general_onnx_path()),
        "general_model_version": GENERAL_MODEL_VERSION,
        "general_classes": _general.num_classes if _general else 0,
        "specialists_loaded": spec_info,
        "tta_crops": TTA_CROPS,
        "thresholds": {
            "general": GENERAL_THRESHOLD,
            "low_confidence": float(__import__("os").getenv("LOW_CONF_THRESHOLD", "0.25")),
            "advisory_only": float(__import__("os").getenv("ADVISORY_ONLY_THRESHOLD", "0.15")),
        },
        "tunisia_classes": len(TUNISIA_LABELS),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _key: str = Security(require_api_key)):
    if not _router:
        raise HTTPException(503, "Models not loaded")

    try:
        raw = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 encoding")

    check = validate_image_bytes(raw)
    if not check.ok:
        raise HTTPException(400, check.issue)

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    qw = quality_warnings(img)
    lang = req.locale if req.locale in ("ar", "fr", "en") else "fr"
    scan_id = req.scan_id or str(uuid.uuid4())

    t0 = time.time()
    dec = _router.route(img, crop_hint=req.crop_hint)
    resp = _build_response(dec, scan_id, lang, qw)

    log.info(
        f"[{scan_id[:8]}] {resp.model_used} → {resp.primary_prediction.label if resp.primary_prediction else '?'} "
        f"({resp.primary_prediction.confidence:.0%} {resp.confidence_level}) "
        f"crop={dec.crop_family} route={dec.model_type} {int((time.time()-t0)*1000)}ms"
    )

    try:
        feedback.append({
            "type": "scan", "scan_id": scan_id, "timestamp": time.time(),
            "predicted": resp.primary_prediction.label if resp.primary_prediction else None,
            "confidence": resp.primary_prediction.confidence if resp.primary_prediction else 0,
            "confidence_level": resp.confidence_level,
            "model_used": dec.model_key, "model_type": dec.model_type,
            "crop_family": dec.crop_family, "crop_hint": req.crop_hint,
            "locale": lang, "below_threshold": resp.below_threshold,
            "top3": [p.label for p in resp.top_predictions[:3]],
            "quality_warnings": qw,
        })
    except Exception as e:
        log.warning(f"Feedback write failed: {e}")

    return resp


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(...), locale: str = "fr",
    crop_hint: str = "", scan_id: str = "",
    _key: str = Security(require_api_key),
):
    """Multipart form upload variant of /predict."""
    if not _router:
        raise HTTPException(503, "Models not loaded")
    raw = await file.read()
    check = validate_image_bytes(raw)
    if not check.ok:
        raise HTTPException(400, check.issue)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    qw = quality_warnings(img)
    lang = locale if locale in ("ar", "fr", "en") else "fr"
    sid = scan_id or str(uuid.uuid4())
    dec = _router.route(img, crop_hint=crop_hint)
    return _build_response(dec, sid, lang, qw)


@app.post("/feedback")
def post_feedback(req: FeedbackRequest, _key: str = Security(require_api_key)):
    feedback.append({
        "type": "feedback", "scan_id": req.scan_id, "timestamp": time.time(),
        "user_confirmed": req.user_confirmed, "correct_label": req.correct_label,
    })
    return {"status": "ok", "message": "Feedback recorded"}


@app.get("/feedback/stats")
def get_feedback_stats(_key: str = Security(require_api_key)):
    return feedback.stats()


@app.get("/models")
def list_models(_key: str = Security(require_api_key)):
    from .config import specialist_onnx_path as sp
    models = {
        "general": {
            "status": "loaded" if _general else "not_loaded",
            "version": GENERAL_MODEL_VERSION,
            "classes": _general.num_classes if _general else 0,
        }
    }
    for key, cfg in SPECIALISTS.items():
        models[key] = {
            "tier": cfg["tier"],
            "version": cfg["version"],
            "enabled": cfg["enabled"],
            "loaded": key in (_router.specialists if _router else {}),
            "onnx_exists": sp(key).exists(),
            "conf_threshold": cfg["conf"],
            "description": cfg["desc"],
        }
    return {"models": models}
