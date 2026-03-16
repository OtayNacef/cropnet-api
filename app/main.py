"""
CropNet API v4 — Tunisia-Optimized Crop Disease Detection
FastAPI + ONNX Runtime + ViT (55 classes, incl. Tunisia crops)
Author: Fellah (fellah.tn)
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from PIL import Image, ImageEnhance
from pydantic import BaseModel, Field
from scipy.special import softmax

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(os.getenv("MODEL_PATH", "/opt/cropnet/models/cropnet_v2.onnx"))
LABELS_PATH  = Path(os.getenv("LABELS_PATH", "/opt/cropnet/models/labels.json"))
API_KEY      = os.getenv("CROPNET_API_KEY", "")          # empty = no auth
CONF_THRESH  = float(os.getenv("CONF_THRESHOLD", "0.40"))
TTA_CROPS    = int(os.getenv("TTA_CROPS", "5"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "1.4"))    # calibration: soften overconfident logits
IMG_SIZE     = int(os.getenv("IMG_SIZE", "224"))

# ── App ───────────────────────────────────────────────────────────────────────
# NOTE: app + middleware are initialized after lifespan() is defined below

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str | None = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return key

# ── Labels ────────────────────────────────────────────────────────────────────
with open(LABELS_PATH) as f:
    _label_data = json.load(f)

ID2LABEL: dict[int, str] = {int(k): v for k, v in _label_data["id2label"].items()}
NUM_CLASSES = len(ID2LABEL)

# Human-readable translations (AR / FR / EN)
LABEL_TRANSLATIONS: dict[str, dict[str, str]] = {
    # PlantVillage
    "Apple___Apple_scab":                           {"fr": "Tavelure du pommier",               "ar": "جرب التفاح",                     "en": "Apple Scab"},
    "Apple___Black_rot":                            {"fr": "Pourridié noir du pommier",          "ar": "عفن التفاح الأسود",              "en": "Apple Black Rot"},
    "Apple___Cedar_apple_rust":                     {"fr": "Rouille du pommier",                 "ar": "صدأ التفاح",                     "en": "Cedar Apple Rust"},
    "Apple___healthy":                              {"fr": "Pommier sain",                       "ar": "تفاحة سليمة",                    "en": "Healthy Apple"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"fr": "Cercosporiose du maïs",        "ar": "تبقع أوراق الذرة",               "en": "Corn Gray Leaf Spot"},
    "Corn_(maize)___Common_rust_":                  {"fr": "Rouille commune du maïs",            "ar": "صدأ الذرة الشائع",               "en": "Corn Common Rust"},
    "Corn_(maize)___Northern_Leaf_Blight":          {"fr": "Helminthosporiose du maïs",          "ar": "لفحة أوراق الذرة الشمالية",      "en": "Corn Northern Leaf Blight"},
    "Corn_(maize)___healthy":                       {"fr": "Maïs sain",                          "ar": "ذرة سليمة",                      "en": "Healthy Corn"},
    "Grape___Black_rot":                            {"fr": "Pourriture noire de la vigne",       "ar": "عفن العنب الأسود",               "en": "Grape Black Rot"},
    "Grape___Esca_(Black_Measles)":                 {"fr": "Esca (rougeot) de la vigne",         "ar": "مرض إيسكا في العنب",             "en": "Grape Esca"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":   {"fr": "Tache foliaire de la vigne",         "ar": "تبقع أوراق العنب",               "en": "Grape Leaf Blight"},
    "Grape___healthy":                              {"fr": "Vigne saine",                        "ar": "عنب سليم",                       "en": "Healthy Grape"},
    "Orange___Haunglongbing_(Citrus_greening)":     {"fr": "Greening des agrumes (HLB)",         "ar": "مرض اخضرار الحمضيات",            "en": "Citrus Greening (HLB)"},
    "Peach___Bacterial_spot":                       {"fr": "Tache bactérienne du pêcher",        "ar": "التبقع البكتيري للخوخ",          "en": "Peach Bacterial Spot"},
    "Peach___healthy":                              {"fr": "Pêcher sain",                        "ar": "خوخ سليم",                       "en": "Healthy Peach"},
    "Pepper,_bell___Bacterial_spot":                {"fr": "Tache bactérienne du poivron",       "ar": "التبقع البكتيري للفلفل",         "en": "Bell Pepper Bacterial Spot"},
    "Pepper,_bell___healthy":                       {"fr": "Poivron sain",                       "ar": "فلفل سليم",                      "en": "Healthy Bell Pepper"},
    "Potato___Early_blight":                        {"fr": "Alternariose de la pomme de terre",  "ar": "اللفحة المبكرة للبطاطا",         "en": "Potato Early Blight"},
    "Potato___Late_blight":                         {"fr": "Mildiou de la pomme de terre",       "ar": "اللفحة المتأخرة للبطاطا",        "en": "Potato Late Blight"},
    "Potato___healthy":                             {"fr": "Pomme de terre saine",               "ar": "بطاطا سليمة",                    "en": "Healthy Potato"},
    "Squash___Powdery_mildew":                      {"fr": "Oïdium de la courge",                "ar": "البياض الدقيقي للقرع",           "en": "Squash Powdery Mildew"},
    "Strawberry___Leaf_scorch":                     {"fr": "Brûlure des feuilles du fraisier",   "ar": "حرق أوراق الفراولة",             "en": "Strawberry Leaf Scorch"},
    "Strawberry___healthy":                         {"fr": "Fraisier sain",                      "ar": "فراولة سليمة",                   "en": "Healthy Strawberry"},
    "Tomato___Bacterial_spot":                      {"fr": "Tache bactérienne de la tomate",     "ar": "التبقع البكتيري للطماطم",        "en": "Tomato Bacterial Spot"},
    "Tomato___Early_blight":                        {"fr": "Alternariose de la tomate",          "ar": "اللفحة المبكرة للطماطم",         "en": "Tomato Early Blight"},
    "Tomato___Late_blight":                         {"fr": "Mildiou de la tomate",               "ar": "اللفحة المتأخرة للطماطم",        "en": "Tomato Late Blight"},
    "Tomato___Leaf_Mold":                           {"fr": "Moisissure des feuilles de tomate",  "ar": "عفن أوراق الطماطم",              "en": "Tomato Leaf Mold"},
    "Tomato___Septoria_leaf_spot":                  {"fr": "Septoriose de la tomate",            "ar": "تبقع سبتوريا للطماطم",           "en": "Tomato Septoria Leaf Spot"},
    "Tomato___Spider_mites Two-spotted_spider_mite":{"fr": "Acariens bisporus sur tomate",       "ar": "عث العنكبوت على الطماطم",        "en": "Tomato Spider Mites"},
    "Tomato___Target_Spot":                         {"fr": "Tache cible de la tomate",           "ar": "التبقع الهدفي للطماطم",          "en": "Tomato Target Spot"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":       {"fr": "Virus TYLCV de la tomate",           "ar": "فيروس تجعد أوراق الطماطم",       "en": "Tomato Yellow Leaf Curl Virus"},
    "Tomato___Tomato_mosaic_virus":                 {"fr": "Virus de la mosaïque de la tomate",  "ar": "فيروس فسيفساء الطماطم",          "en": "Tomato Mosaic Virus"},
    "Tomato___healthy":                             {"fr": "Tomate saine",                       "ar": "طماطم سليمة",                    "en": "Healthy Tomato"},
    # Tunisia-specific
    "Olive - Peacock Spot (Spilocea oleagina)":     {"fr": "Œil de paon de l'olivier",           "ar": "عين الطاووس في الزيتون",         "en": "Olive Peacock Spot"},
    "Olive - Olive Knot (Pseudomonas savastanoi)":  {"fr": "Tuberculose de l'olivier",           "ar": "سرطان الزيتون",                  "en": "Olive Knot"},
    "Olive - Healthy":                              {"fr": "Olivier sain",                       "ar": "زيتون سليم",                     "en": "Healthy Olive"},
    "Date Palm - Bayoud Disease (Fusarium oxysporum)": {"fr": "Fusariose du palmier dattier",    "ar": "مرض البيوض في نخيل التمر",       "en": "Date Palm Bayoud Disease"},
    "Date Palm - Black Scorch (Ceratocystis radicicola)": {"fr": "Brûlure noire du palmier",     "ar": "الحرق الأسود في النخيل",         "en": "Date Palm Black Scorch"},
    "Date Palm - Healthy":                          {"fr": "Palmier dattier sain",               "ar": "نخلة تمر سليمة",                 "en": "Healthy Date Palm"},
    "Citrus - Canker (Xanthomonas citri)":          {"fr": "Chancre des agrumes",                "ar": "قرحة الحمضيات",                  "en": "Citrus Canker"},
    "Citrus - Greening / HLB":                      {"fr": "Greening des agrumes",               "ar": "اخضرار الحمضيات",                "en": "Citrus Greening"},
    "Citrus - Healthy":                             {"fr": "Agrumes sains",                      "ar": "حمضيات سليمة",                   "en": "Healthy Citrus"},
    "Wheat - Septoria Leaf Blotch":                 {"fr": "Septoriose du blé",                  "ar": "تبقع سبتوريا في القمح",          "en": "Wheat Septoria Leaf Blotch"},
    "Wheat - Yellow Rust (Puccinia striiformis)":   {"fr": "Rouille jaune du blé",               "ar": "الصدأ الأصفر للقمح",             "en": "Wheat Yellow Rust"},
}

TUNISIA_CLASSES = {
    "Olive - Peacock Spot (Spilocea oleagina)",
    "Olive - Olive Knot (Pseudomonas savastanoi)",
    "Olive - Healthy",
    "Date Palm - Bayoud Disease (Fusarium oxysporum)",
    "Date Palm - Black Scorch (Ceratocystis radicicola)",
    "Date Palm - Healthy",
    "Citrus - Canker (Xanthomonas citri)",
    "Citrus - Greening / HLB",
    "Citrus - Healthy",
    "Wheat - Septoria Leaf Blotch",
    "Wheat - Yellow Rust (Puccinia striiformis)",
}

def translate_label(label: str, locale: str) -> str:
    t = LABEL_TRANSLATIONS.get(label, {})
    return t.get(locale) or t.get("fr") or label

def is_healthy(label: str) -> bool:
    return "healthy" in label.lower() or "Healthy" in label or "sain" in label.lower()

def severity(label: str, confidence: float) -> str:
    if is_healthy(label):
        return "healthy"
    if confidence >= 0.75:
        return "severe"
    if confidence >= 0.50:
        return "moderate"
    return "mild"

# ── Preprocessing ─────────────────────────────────────────────────────────────
def clahe_enhance(img: Image.Image) -> Image.Image:
    """CLAHE-inspired contrast enhancement for real-world leaf photos."""
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.20)
    img = ImageEnhance.Color(img).enhance(1.10)
    return img

def normalize(arr: np.ndarray) -> np.ndarray:
    """ImageNet normalization."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = normalize(arr)
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW

def five_crops(img: Image.Image) -> list[Image.Image]:
    """5-crop TTA: center + 4 corners at 90% scale."""
    w, h = img.size
    cs = int(min(w, h) * 0.90)
    cx, cy = w // 2, h // 2
    coords = [
        (cx - cs//2, cy - cs//2, cx + cs//2, cy + cs//2),  # center
        (0,        0,        cs,      cs     ),              # top-left
        (w - cs,   0,        w,       cs     ),              # top-right
        (0,        h - cs,   cs,      h      ),              # bottom-left
        (w - cs,   h - cs,   w,       h      ),              # bottom-right
    ]
    return [clahe_enhance(img.crop(c)) for c in coords[:TTA_CROPS]]

# ── ONNX Inference ───────────────────────────────────────────────────────────
print(f"[CropNet v4] Loading ONNX model from {MODEL_PATH}...")
t0 = time.time()
_sess_opts = ort.SessionOptions()
_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_sess_opts.intra_op_num_threads = 4
session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=_sess_opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
print(f"[CropNet v4] Ready in {time.time()-t0:.1f}s | {NUM_CLASSES} classes | TTA={TTA_CROPS}")

# ── Feedback store ────────────────────────────────────────────────────────────
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", "/opt/cropnet/logs/feedback.jsonl"))
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_feedback(entry: dict) -> None:
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── Warmup ────────────────────────────────────────────────────────────────────
def _warmup():
    """Run a dummy inference to pre-load ONNX graph into memory."""
    print("[CropNet v4] Running warmup inference...")
    t0 = time.time()
    dummy = Image.new("RGB", (224, 224), color=(34, 139, 34))
    inp = preprocess(dummy)
    for _ in range(3):  # 3 passes to fully JIT the graph
        session.run([OUTPUT_NAME], {INPUT_NAME: inp})
    print(f"[CropNet v4] Warmup done in {time.time()-t0:.1f}s — first request will be fast ⚡")

@asynccontextmanager
async def lifespan(app_: "FastAPI"):
    _warmup()
    yield

app = FastAPI(
    title="CropNet API",
    version="4.0.0",
    description="Tunisia-optimized crop disease detection. 55 classes including olive, date palm, wheat, citrus.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def infer_tta(img: Image.Image) -> tuple[list[dict], int]:
    """Run TTA inference. Returns (top5, crop_agreement)."""
    crops = five_crops(img)
    all_logits: list[np.ndarray] = []

    for crop in crops:
        inp = preprocess(crop)
        logits = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0][0]
        all_logits.append(logits)

    # Average logits, then apply temperature scaling + softmax
    avg_logits = np.mean(all_logits, axis=0) / TEMPERATURE
    probs = softmax(avg_logits)

    # Top-1 vote per crop (majority)
    top1_per_crop = [int(np.argmax(session.run([OUTPUT_NAME], {INPUT_NAME: preprocess(c)})[0][0])) for c in crops]
    vote_counts: dict[int, int] = {}
    for v in top1_per_crop:
        vote_counts[v] = vote_counts.get(v, 0) + 1
    majority_id = max(vote_counts, key=vote_counts.get)
    majority_votes = vote_counts[majority_id]

    # Ranked results
    ranked_ids = np.argsort(probs)[::-1]
    top5 = []
    for idx in ranked_ids[:5]:
        lbl = ID2LABEL.get(int(idx), f"class_{idx}")
        top5.append({
            "class_id":   int(idx),
            "label":      lbl,
            "confidence": float(round(probs[idx], 4)),
            "is_tunisia": lbl in TUNISIA_CLASSES,
        })

    # If majority vote differs from averaged top-1 and is strong, prefer it
    if majority_id != ranked_ids[0] and majority_votes >= 3:
        majority_lbl = ID2LABEL.get(majority_id, f"class_{majority_id}")
        majority_conf = float(round(probs[majority_id], 4))
        top5.insert(0, {
            "class_id":   majority_id,
            "label":      majority_lbl,
            "confidence": majority_conf,
            "is_tunisia": majority_lbl in TUNISIA_CLASSES,
        })
        top5 = top5[:5]

    return top5, majority_votes

# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG image")
    locale: str       = Field("fr",  description="Response language: ar | fr | en")
    crop_hint: str    = Field("",    description="Optional crop type hint (e.g. tomato, olive)")
    scan_id: str      = Field("",    description="Optional scan ID for feedback correlation")

class FeedbackRequest(BaseModel):
    scan_id:        str  = Field(..., description="Scan ID from predict response")
    correct_label:  str  = Field(..., description="Correct disease label (raw class name)")
    user_confirmed: bool = Field(True, description="True = model was right, False = correction")

class ClassResult(BaseModel):
    label_raw:      str
    label:          str
    confidence:     float
    confidence_pct: float
    is_tunisia:     bool

class PredictResponse(BaseModel):
    scan_id:        str
    disease:        str
    disease_en:     str
    disease_raw:    str
    confidence:     float
    confidence_pct: float
    severity:       str
    is_healthy:     bool
    is_tunisia:     bool
    top5:           list[ClassResult]
    crop_agreement: str
    inference_ms:   int
    model_version:  str = "cropnet-v4-ViT-55cls-TTA5"
    below_threshold: bool

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        str(MODEL_PATH),
        "classes":      NUM_CLASSES,
        "tta_crops":    TTA_CROPS,
        "version":      "4.0.0",
        "tunisia_classes": len(TUNISIA_CLASSES),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _key: str = Security(require_api_key)):
    import uuid
    # Decode image
    try:
        img_bytes = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    t0 = time.time()
    top5_raw, crop_agreement = infer_tta(img)
    inf_ms = int((time.time() - t0) * 1000)

    top = top5_raw[0]
    top_label    = top["label"]
    top_conf     = top["confidence"]
    below_thresh = top_conf < CONF_THRESH
    scan_id      = req.scan_id or str(uuid.uuid4())

    locale = req.locale if req.locale in ("ar", "fr", "en") else "fr"

    top5_out = [
        ClassResult(
            label_raw=      r["label"],
            label=          translate_label(r["label"], locale),
            confidence=     r["confidence"],
            confidence_pct= round(r["confidence"] * 100, 1),
            is_tunisia=     r["is_tunisia"],
        )
        for r in top5_raw
    ]

    # Log prediction for auto-learning dataset
    save_feedback({
        "scan_id":        scan_id,
        "timestamp":      time.time(),
        "predicted":      top_label,
        "confidence":     top_conf,
        "crop_hint":      req.crop_hint,
        "locale":         locale,
        "below_threshold": below_thresh,
        "top5":           [r["label"] for r in top5_raw],
        "crop_agreement": crop_agreement,
        "user_confirmed": None,   # filled later via /feedback
        "correct_label":  None,
    })

    return PredictResponse(
        scan_id=        scan_id,
        disease=        translate_label(top_label, locale),
        disease_en=     translate_label(top_label, "en"),
        disease_raw=    top_label,
        confidence=     top_conf,
        confidence_pct= round(top_conf * 100, 1),
        severity=       severity(top_label, top_conf),
        is_healthy=     is_healthy(top_label),
        is_tunisia=     top_label in TUNISIA_CLASSES,
        top5=           top5_out,
        crop_agreement= f"{crop_agreement}/{TTA_CROPS}",
        inference_ms=   inf_ms,
        below_threshold=below_thresh,
    )

@app.post("/feedback")
def feedback(req: FeedbackRequest, _key: str = Security(require_api_key)):
    """
    Collect user feedback — was the model right or wrong?
    Builds the labeled dataset for future fine-tuning.
    """
    save_feedback({
        "scan_id":        req.scan_id,
        "timestamp":      time.time(),
        "user_confirmed": req.user_confirmed,
        "correct_label":  req.correct_label,
        "type":           "feedback",
    })
    return {"status": "ok", "message": "Feedback recorded. Thank you 🌾"}

@app.get("/feedback/stats")
def feedback_stats(_key: str = Security(require_api_key)):
    """Return feedback dataset stats."""
    if not FEEDBACK_PATH.exists():
        return {"total": 0, "confirmed": 0, "corrections": 0, "accuracy": None}

    total = confirmed = corrections = 0
    with open(FEEDBACK_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("type") == "feedback":
                    total += 1
                    if entry.get("user_confirmed"):
                        confirmed += 1
                    else:
                        corrections += 1
            except Exception:
                pass

    accuracy = round(confirmed / total * 100, 1) if total > 0 else None
    return {
        "total_feedback": total,
        "confirmed":      confirmed,
        "corrections":    corrections,
        "real_world_accuracy": f"{accuracy}%" if accuracy else "no data",
    }
