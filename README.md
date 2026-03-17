# CropNet API 🌾

Multi-model crop disease diagnosis for Tunisian agriculture.

> **⚠️ Advisory Only** — CropNet provides AI-generated agricultural guidance. All predictions are advisory and should be verified by a qualified agronomist before making critical decisions. Low-confidence results are explicitly flagged. Do not use this system as a sole basis for treatment decisions.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Client      │────▶│  General      │────▶│  Crop Family     │
│  (photo)     │     │  Model (v4)   │     │  Detection       │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                              ┌──────────┐  ┌──────────┐  ┌──────────┐
                              │  Olive    │  │  Wheat   │  │  Citrus  │
                              │Specialist │  │Specialist│  │Specialist│
                              └──────────┘  └──────────┘  └──────────┘
```

**General model** provides broad coverage (89 classes, DINOv2-B backbone, 85.8% val accuracy). It acts as both a classifier and a router — the top-k predictions are used to infer the crop family.

**Specialist models** provide higher accuracy for specific Tunisian crops. Routing is threshold-gated: specialists override the general model only when confident enough.

### Specialist Models — All Deployed ✅

| Tier | Crop | Classes | Val Accuracy | Status |
|------|------|---------|-------------|--------|
| **T1** | 🫒 Olive | 3 (healthy, aculus olearius, peacock spot) | **99.63%** | ✅ Live |
| **T1** | 🌴 Date Palm | 3 (brown spots, healthy, white scale) | **100.0%** | ✅ Live |
| **T1** | 🌾 Wheat | 3 (healthy, septoria, stripe rust) | **100.0%** | ✅ Live |
| **T1** | 🍊 Citrus | 5 (black spot, canker, greening, healthy, melanose) | **98.44%** | ✅ Live |
| **T2** | 🍅 Tomato | 10 (bacterial spot, blights, mold, viruses, etc.) | **99.95%** | ✅ Live |
| **T2** | 🌶️ Pepper | 2 (bacterial spot, healthy) | **100.0%** | ✅ Live |
| **T2** | 🍉 Watermelon | 4 (anthracnose, downy mildew, healthy, mosaic virus) | **100.0%** | ✅ Live |

All specialists use **DINOv2-B** backbone with progressive unfreezing, MixUp augmentation, and cosine LR schedule. ONNX exported (~331 MB each).

### Tunisia Focus

CropNet prioritizes crops important to Tunisian agriculture. 19 disease labels are tagged as Tunisia-relevant. The advisory system provides localized recommendations in Arabic, French, and English.

## Endpoints

### `GET /health`

No auth required. Returns API status, loaded models, thresholds, version.

### `POST /predict`

Requires `X-API-Key` header.

**Request:**
```json
{
  "image_base64": "<base64-encoded JPEG/PNG>",
  "locale": "ar",
  "crop_hint": "olive",
  "scan_id": "optional-client-id"
}
```

**Response:**
```json
{
  "status": "ok",
  "request_id": "uuid",
  "language": "ar",
  "crop_detected": "olive",
  "primary_prediction": {
    "label": "Olive___olive_peacock_spot",
    "display_name": "عين الطاووس في الزيتون",
    "confidence": 0.82,
    "calibrated_confidence": null
  },
  "top_predictions": [...],
  "model_used": "cropnet-olive-v1",
  "model_type": "specialist",
  "routing_reason": "Specialist 'olive' confident (85% ≥ threshold 45%); matched via top-k label",
  "is_low_confidence": false,
  "confidence_level": "high",
  "advisory_only": true,
  "recommended_action": "رش مبيد فطري نحاسي. أزل الأوراق المصابة وأتلفها.",
  "disclaimer": "تشخيص ذكاء اصطناعي للاسترشاد فقط. استشر مهندساً زراعياً للقرارات المهمة.",
  "image_quality_warnings": [],
  "severity": "moderate",
  "is_healthy": false,
  "is_tunisia": true,
  "inference_ms": 250
}
```

### `POST /predict/upload`

Multipart form upload variant. Same response schema.

### `POST /feedback`

Submit user correction for a previous prediction.

### `GET /feedback/stats`

Feedback accuracy statistics.

### `GET /models`

List all models (general + specialists) with status.

## Routing Logic

1. Run general model → get top-k predictions with TTA (5-crop)
2. Scan top-k for crop family match (not just top-1)
3. If crop_hint is provided, use it as override
4. If a specialist is enabled and loaded for that crop:
   - Run specialist model
   - Specialist overrides general if conf ≥ per-specialist threshold
   - If specialist is close to general (≥80%), prefer specialist (crop-specific knowledge)
   - Otherwise fall back to general
5. `routing_reason` always explains the decision

All routing is deterministic. Same image → same path.

## Confidence & Calibration

Four threshold levels:

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `GENERAL_THRESHOLD` | 0.40 | Below this = "low confidence" |
| `SPECIALIST_THRESHOLD` | 0.45 | Specialist must exceed to override |
| `LOW_CONF_THRESHOLD` | 0.25 | Below this = "very low" warning |
| `ADVISORY_ONLY_THRESHOLD` | 0.15 | Below this = refuse assertion |

Calibration: temperature scaling (T=1.4) is applied during inference. Per-model Platt scaling is supported via `metadata.json` but not yet trained. The `calibrated_confidence` field is included in responses (currently `null` until post-hoc calibration is available).

## Localization

Three languages supported: **Arabic (ar)**, **French (fr)**, **English (en)**.

Localized content:
- Disease display names (45+ labels)
- Recommended actions per disease
- Advisory messages (very_low / low / moderate / high)
- Disclaimer text
- Image quality warnings

Fallback: English if locale unknown.

## Environment Variables

### Required
| Variable | Description |
|----------|-------------|
| `CROPNET_API_KEY` | API authentication key |

### Model Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `/opt/cropnet/models` | Model artifacts directory |
| `IMG_SIZE` | `256` | Input image size |
| `TTA_CROPS` | `5` | TTA crop count |
| `TEMPERATURE` | `1.4` | Softmax temperature scaling |
| `ONNX_THREADS` | `4` | ONNX Runtime thread count |

### Thresholds
| Variable | Default |
|----------|---------|
| `GENERAL_THRESHOLD` | `0.40` |
| `SPECIALIST_THRESHOLD` | `0.45` |
| `LOW_CONF_THRESHOLD` | `0.25` |
| `ADVISORY_ONLY_THRESHOLD` | `0.15` |

### Per-Specialist Thresholds
| Variable | Default |
|----------|---------|
| `SPECIALIST_THRESHOLD_OLIVE` | `0.45` |
| `SPECIALIST_THRESHOLD_DATE_PALM` | `0.45` |
| `SPECIALIST_THRESHOLD_WHEAT` | `0.45` |
| `SPECIALIST_THRESHOLD_CITRUS` | `0.45` |
| `SPECIALIST_THRESHOLD_TOMATO` | `0.45` |
| `SPECIALIST_THRESHOLD_PEPPER` | `0.50` |
| `SPECIALIST_THRESHOLD_WATERMELON` | `0.50` |

### Specialist Enable Flags
| Variable | Default |
|----------|---------|
| `ENABLE_SPECIALIST_OLIVE` | `true` |
| `ENABLE_SPECIALIST_DATE_PALM` | `true` |
| `ENABLE_SPECIALIST_WHEAT` | `true` |
| `ENABLE_SPECIALIST_CITRUS` | `true` |
| `ENABLE_SPECIALIST_TOMATO` | `false` |
| `ENABLE_SPECIALIST_PEPPER` | `false` |
| `ENABLE_SPECIALIST_WATERMELON` | `false` |

## Training

### Prerequisites

```bash
pip install -r training/requirements.txt
# Requires: NVIDIA GPU, CUDA 12+, ~24GB VRAM for DINOv2-B
```

### Train a specialist

```bash
# Prepare data: ImageFolder layout (root/class_name/image.jpg)
python training/olive/train.py --data-dir /data/olive --output-dir /output/olive
```

### Evaluate

```bash
python training/olive/eval.py \
  --model /output/olive/cropnet-olive-v1.onnx \
  --labels /output/olive/labels.json \
  --data-dir /data/olive/val \
  --output-dir /output/olive
```

Produces: `eval.json` (top-1, top-3, macro F1, weighted F1, per-class P/R/F1), `confusion_matrix.json`.

### Export ONNX (standalone)

```bash
python training/olive/export_onnx.py \
  --checkpoint /output/olive/best.pt \
  --num-classes 4 \
  --output /output/olive/cropnet-olive-v1.onnx
```

### Data quality utilities

```bash
# Class distribution
python training/common/distribution.py --dir /data/olive --output dist.json

# Near-duplicate detection
python training/common/dedup.py --dir /data/olive --dry-run

# Train/val leakage check
python training/common/leakage.py --train-dir /data/olive/train --val-dir /data/olive/val

# Dataset manifest
python training/common/manifests.py --dir /data/olive --output manifest.json

# Train/val split
python training/common/split.py --src /data/olive_raw --dst /data/olive --val-ratio 0.1
```

### Training artifacts (per run)

| File | Contents |
|------|----------|
| `best.pt` | Best checkpoint (PyTorch state dict) |
| `cropnet-{crop}-v1.onnx` | ONNX export |
| `labels.json` | id2label mapping |
| `metadata.json` | Training config, lineage, calibration params, date, seed |
| `report.json` | Training summary with per-epoch history |
| `eval.json` | Full evaluation metrics (after running eval.py) |
| `confusion_matrix.json` | NxN confusion matrix |

## Local Development

```bash
# Install API deps
pip install -r api/requirements.txt

# Run locally (needs model files in MODELS_DIR)
MODELS_DIR=./models FEEDBACK_DIR=./logs CROPNET_API_KEY=dev uvicorn api.main:app --reload --port 8001

# Run tests
pip install pytest
python -m pytest tests/ -v
```

## Production Deployment

```bash
# Using the install script (sets up venv, systemd, .env)
sudo bash scripts/install.sh

# Or manually:
cp -r api/ /opt/cropnet/
cp scripts/cropnet.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable cropnet && systemctl start cropnet

# Health check
curl http://127.0.0.1:8001/health
```

### Model file layout (production)

```
/opt/cropnet/models/
├── general/
│   ├── cropnet-general-v1.onnx     # or legacy: cropnet_v2.onnx at parent level
│   ├── labels.json
│   └── metadata.json
├── olive/
│   ├── cropnet-olive-v1.onnx
│   ├── labels.json
│   └── metadata.json
├── wheat/
│   └── ...
└── ...
```

The API falls back to legacy flat layout (`models/cropnet_v2.onnx` + `models/labels.json`) if the clean structure isn't found.

## Repository Structure

```
cropnet-api/
├── api/                        # FastAPI application
│   ├── main.py                 # App entry, routes
│   ├── config.py               # All configuration
│   ├── auth.py                 # API key auth
│   ├── schemas.py              # Pydantic models
│   ├── routing.py              # General → specialist routing
│   ├── feedback.py             # JSONL feedback log
│   ├── requirements.txt
│   ├── inference/
│   │   ├── general.py          # ONNX model wrapper + TTA
│   │   ├── specialists.py      # Auto-load specialist models
│   │   ├── preprocess.py       # Image validation + preprocessing
│   │   ├── metadata.py         # Labels, translations, Tunisia tags
│   │   ├── calibration.py      # Thresholds, confidence assessment
│   │   └── quality.py          # Image quality warnings
│   └── locales/
│       ├── ar.json             # Arabic translations + actions
│       ├── fr.json             # French
│       └── en.json             # English
├── training/
│   ├── common/                 # Shared training code
│   │   ├── train_loop.py       # DINOv2-B fine-tuning loop
│   │   ├── datasets.py         # ImageFolder dataset + sampling
│   │   ├── transforms.py       # Augmentation pipelines
│   │   ├── losses.py           # MixUp
│   │   ├── metrics.py          # Top-1/3, F1, confusion matrix
│   │   ├── utils.py            # Model building, ONNX export
│   │   ├── split.py            # Train/val splitter
│   │   ├── dedup.py            # Perceptual hash dedup
│   │   ├── distribution.py     # Class distribution report
│   │   ├── leakage.py          # Data leakage checks
│   │   ├── manifests.py        # Dataset manifest
│   │   └── reports.py          # Markdown report generator
│   ├── general/                # General model config + scripts
│   ├── olive/                  # Olive specialist
│   ├── date_palm/              # Date palm specialist
│   ├── wheat/
│   ├── citrus/
│   ├── tomato/
│   ├── pepper/
│   ├── watermelon/
│   └── requirements.txt        # Training deps (GPU)
├── models/                     # Model artifacts (gitignored)
├── tests/                      # Pytest test suite (122 tests)
├── scripts/
│   ├── install.sh              # Production installer
│   └── cropnet.service         # systemd unit
└── README.md
```

## Limitations

- **No specialist models trained yet.** Tier 1 and Tier 2 specialists are fully scaffolded but require GPU training runs with curated datasets. Only the general model is production-ready.
- **Calibration is temperature-only.** Post-hoc calibration (Platt scaling) hooks exist but are not yet fitted to held-out data. `calibrated_confidence` is currently `null`.
- **Deduplication is perceptual-hash based.** It catches near-identical images but not semantically similar ones (e.g., same leaf from different angles). This is documented honestly.
- **Leakage checks are file-level.** We check filename overlap and byte-identical duplicates across train/val splits, but cannot detect augmentation leakage or semantic duplicates without embedding databases.
- **89 class labels include 25 junk entries** from the source dataset (numeric strings, folder names). These are filtered at inference time but the index mapping is preserved for model compatibility.
- **Advisory only.** This system does not provide medical-grade or certified agricultural diagnoses. All outputs include disclaimers in the user's language.

## Safety Disclaimer

CropNet is an **advisory tool** for agricultural guidance. It should not be used as the sole basis for crop treatment decisions. Always consult a qualified agronomist or agricultural extension officer before applying treatments based on AI predictions.

All predictions include:
- Explicit confidence levels (high / moderate / low / very_low)
- Localized disclaimers
- Recommended actions from agricultural best practices
- Image quality warnings when input photos are poor

Low-confidence predictions are clearly marked. The system prefers honest uncertainty over false certainty.

---

Built for [Fellah](https://fellah.tn) — empowering Tunisian farmers with AI.
