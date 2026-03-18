# CropNet API 🌾

Multi-model crop disease diagnosis for Tunisian agriculture. **1 general model + 8 specialist models**, all running ONNX inference on a single VPS.

> **⚠️ Advisory Only** — CropNet provides AI-generated agricultural guidance. All predictions are advisory and should be verified by a qualified agronomist before making critical decisions. Low-confidence results are explicitly flagged. Do not use this system as a sole basis for treatment decisions.

## Architecture

```
                              ┌──────────────────────┐
                              │      Client App       │
                              │  (Fellah Web/Mobile)  │
                              └──────────┬───────────┘
                                         │ POST /predict
                                         ▼
                              ┌──────────────────────┐
                              │    CropNet API v5     │
                              │   FastAPI + Uvicorn   │
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────┬┴┬────────────────────┐
                    ▼                    ▼                      ▼
           ┌────────────────┐  ┌─────────────────┐    ┌────────────────┐
           │  Preprocessing  │  │  General Model   │    │  Localization   │
           │  256×256 + TTA  │  │  DINOv2-B (v4)   │    │  AR / FR / EN   │
           └────────────────┘  │  89 classes        │    └────────────────┘
                               │  85.8% val acc     │
                               └────────┬──────────┘
                                        │ top-k predictions
                                        ▼
                               ┌─────────────────┐
                               │  Crop Family     │
                               │  Router          │
                               │  (CROP_FAMILIES) │
                               └────────┬────────┘
                                        │ match?
                    ┌───────┬───────┬───┴───┬───────┬───────┬───────┬───────┐
                    ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
                 ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
                 │🫒    ││🌴    ││🌾    ││🍊    ││🍑    ││🍅    ││🌶️    ││🍉    │
                 │Olive ││D.Palm││Wheat ││Citrus││Peach ││Tomato││Pepper││W.melon│
                 │99.6% ││100%  ││100%  ││98.4% ││100%  ││99.9% ││100%  ││100%  │
                 └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
                  Tier 1   Tier 1  Tier 1  Tier 1  Tier 1  Tier 2  Tier 2  Tier 2
```

### How It Works

1. **Photo in** → 256×256 resize + 5-crop TTA (center + 4 corners at 90% scale)
2. **General model** classifies across 89 diseases/conditions
3. **Router** scans top-k predictions for crop family keywords (e.g. "Olive___" → olive specialist)
4. **Specialist** runs if available and enabled → overrides general if confidence ≥ threshold
5. **GPT-4o-mini** generates treatment advice using the CropNet diagnosis
6. **Response** includes disease, confidence, severity, treatment, prevention — all localized

### Specialist Models

| Tier | Crop | Classes | Images | Val Accuracy | Base Model |
|------|------|---------|--------|-------------|------------|
| **T1** | 🫒 Olive | 3 — healthy, aculus olearius, peacock spot | 2,720 | **99.63%** | DINOv2-B |
| **T1** | 🌴 Date Palm | 3 — brown spots, healthy, white scale | 2,631 | **100.0%** | DINOv2-B |
| **T1** | 🌾 Wheat | 3 — healthy, septoria, stripe rust | 407 | **100.0%** | DINOv2-B |
| **T1** | 🍊 Citrus | 5 — black spot, canker, greening, healthy, melanose | 6,394 | **98.44%** | DINOv2-B |
| **T1** | 🍑 Peach | 2 — bacterial spot, healthy | 3,566 | **100.0%** | DINOv2-B |
| **T2** | 🍅 Tomato | 10 — bacterial spot, early/late blight, leaf mold, viruses, etc. | 18,345 | **99.95%** | DINOv2-B |
| **T2** | 🌶️ Pepper | 2 — bacterial spot, healthy | 3,901 | **100.0%** | DINOv2-B |
| **T2** | 🍉 Watermelon | 4 — anthracnose, downy mildew, healthy, mosaic virus | 1,155 | **100.0%** | DINOv2-B |

**Total: 35,619 training images across 32 specialist classes.**

All specialists trained with:
- **Progressive unfreezing**: Frozen backbone → unfreeze last 4 blocks (epoch 5, lr×0.1) → full unfreeze (epoch 10, lr×0.01)
- **MixUp augmentation** (α=0.3) during frozen phases
- **Cosine LR schedule** with warmup
- **Label smoothing** (0.1)
- ONNX exported (~331 MB each, ~347 MB general)

### Planned Specialists (Awaiting Datasets)

| Crop | Status | Notes |
|------|--------|-------|
| 🌰 Almond | ⏳ No dataset | No public leaf disease dataset — needs field collection or partnership with Tunisian institutes |
| 🥜 Pistachio | ⏳ No dataset | Only fruit classification datasets available on Kaggle/Mendeley |

### Tunisia Focus

Tier 1 specialists cover Tunisia's most important crops: **olive** (world's 4th producer), **date palm** (Deglet Nour), **wheat** (staple cereal), **citrus** (Cap Bon), and **peach** (northern Tunisia). 19 of the general model's disease labels are tagged as Tunisia-relevant.

## API Endpoints

### `GET /health`
No auth. Returns API version, loaded models, thresholds, uptime.

### `POST /predict`
Requires `X-API-Key` header.

```json
// Request
{
  "image_base64": "<base64 JPEG/PNG>",
  "locale": "ar",           // ar | fr | en
  "crop_hint": "olive",     // optional — force specialist
  "scan_id": "client-uuid"  // optional — for tracking
}

// Response
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
  "disclaimer": "...",
  "severity": "moderate",
  "is_healthy": false,
  "is_tunisia": true,
  "inference_ms": 250
}
```

### `POST /predict/upload`
Multipart form upload variant. Same response schema.

### `POST /feedback`
Submit user correction for a previous prediction. Stored as JSONL for retraining.

### `GET /feedback/stats`
Feedback accuracy statistics.

### `GET /models`
List all loaded models (general + specialists) with status, tier, class count.

## Routing Logic

```
1. Run general model → top-k predictions (with 5-crop TTA)
2. Scan top-k labels for crop family match (CROP_FAMILIES dict)
3. If crop_hint provided → use as override
4. If specialist found, enabled, and loaded:
   a. Run specialist on same image
   b. If specialist_conf ≥ per-model threshold → use specialist
   c. If specialist_conf ≥ 80% of general_conf → prefer specialist (crop-specific knowledge)
   d. Otherwise → fall back to general
5. routing_reason always explains the decision
```

All routing is **deterministic** — same image → same path. No randomness.

## Confidence & Calibration

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `GENERAL_THRESHOLD` | 0.40 | Below = "low confidence" warning |
| `SPECIALIST_THRESHOLD` | 0.45 | Specialist must exceed to override general |
| `LOW_CONF_THRESHOLD` | 0.25 | Below = "very low" warning |
| `ADVISORY_ONLY_THRESHOLD` | 0.15 | Below = refuse to assert diagnosis |

**Temperature scaling** (T=1.4) softens overconfident logits. Per-model Platt scaling is supported via `metadata.json` but not yet fitted. `calibrated_confidence` is included in responses (currently `null`).

## Localization

Three languages: **Arabic (ar)**, **French (fr)**, **English (en)**.

Localized: disease display names, recommended actions, advisory messages, disclaimers, image quality warnings. Fallback: English.

## Environment Variables

### Required
| Variable | Description |
|----------|-------------|
| `CROPNET_API_KEY` | API authentication key |

### Model Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `/opt/cropnet/models` | Model artifacts directory |
| `IMG_SIZE` | `256` | Input image size (must match training) |
| `TTA_CROPS` | `5` | TTA crop count (1 = disabled) |
| `TEMPERATURE` | `1.4` | Softmax temperature scaling |
| `ONNX_THREADS` | `4` | ONNX Runtime thread count |

### Per-Specialist Config
Each specialist has enable flag and threshold override:
```bash
ENABLE_SPECIALIST_OLIVE=true          # T1: enabled by default
SPECIALIST_THRESHOLD_OLIVE=0.45

ENABLE_SPECIALIST_DATE_PALM=true
ENABLE_SPECIALIST_WHEAT=true
ENABLE_SPECIALIST_CITRUS=true
ENABLE_SPECIALIST_PEACH=true

ENABLE_SPECIALIST_TOMATO=false        # T2: disabled by default
SPECIALIST_THRESHOLD_TOMATO=0.45
ENABLE_SPECIALIST_PEPPER=false
SPECIALIST_THRESHOLD_PEPPER=0.50
ENABLE_SPECIALIST_WATERMELON=false
SPECIALIST_THRESHOLD_WATERMELON=0.50
```

## Training

### Prerequisites
- NVIDIA GPU with ~24GB VRAM (tested on A10G)
- CUDA 12+
- Python 3.11 (via Miniforge/Conda recommended for GLIBC compatibility)
- PyTorch 2.5+ via `conda-forge` channel (bundles its own CUDA/GLIBC libs)

```bash
pip install -r training/requirements.txt
```

### Train a specialist
```bash
# Data: ImageFolder layout → root/class_name/image.jpg
python training/olive/train.py \
  --data-dir /data/datasets/olive \
  --output-dir /data/output/olive
```

Training uses progressive unfreezing:
- **Epochs 1-4**: Backbone frozen, only classifier head trains (lr=3e-4)
- **Epochs 5-9**: Last 4 backbone blocks unfrozen (lr×0.1)
- **Epochs 10+**: Full model unfrozen (lr×0.01)

### Evaluate
```bash
python training/olive/eval.py \
  --model /output/olive/cropnet-olive-v1.onnx \
  --labels /output/olive/labels.json \
  --data-dir /data/olive/val \
  --output-dir /output/olive
```

Produces: `eval.json` (top-1, top-3, macro/weighted F1, per-class P/R/F1), `confusion_matrix.json`.

### Export ONNX
```bash
python training/olive/export_onnx.py \
  --checkpoint /output/olive/best.pt \
  --num-classes 3 \
  --output /output/olive/cropnet-olive-v1.onnx
```

### Data Quality Utilities
```bash
python training/common/distribution.py --dir /data/olive --output dist.json    # Class distribution
python training/common/dedup.py --dir /data/olive --dry-run                     # Near-duplicate detection
python training/common/leakage.py --train-dir /data/train --val-dir /data/val   # Train/val leakage
python training/common/split.py --src /data/raw --dst /data/split --val-ratio 0.1
```

### Training Artifacts (per run)
| File | Contents |
|------|----------|
| `best.pt` | Best checkpoint (PyTorch state dict) |
| `cropnet-{crop}-v1.onnx` | ONNX export (~331 MB) |
| `labels.json` | id → label mapping |
| `metadata.json` | Config, lineage, calibration params, date, seed |
| `report.json` | Training summary with per-epoch metrics |

## Local Development

```bash
pip install -r api/requirements.txt

# Run locally (needs ONNX model files)
MODELS_DIR=./models FEEDBACK_DIR=./logs CROPNET_API_KEY=dev \
  uvicorn api.main:app --reload --port 8001

# Run tests (122 tests)
pip install pytest
python -m pytest tests/ -v
```

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guide.

```bash
# Quick deploy
sudo bash scripts/install.sh

# Health check
curl http://127.0.0.1:8001/health
```

### Production Model Layout
```
/opt/cropnet/models/
├── general/
│   ├── cropnet-general-v1.onnx   (347 MB)
│   ├── labels.json               (89 classes)
│   └── metadata.json
├── olive/
│   ├── cropnet-olive-v1.onnx     (331 MB)
│   ├── labels.json               (3 classes)
│   └── metadata.json
├── date_palm/
├── wheat/
├── citrus/
├── peach/
├── tomato/
├── pepper/
└── watermelon/
    ├── cropnet-watermelon-v1.onnx
    ├── labels.json
    └── metadata.json
```

Total on disk: **~3 GB** (9 ONNX models). Startup time: ~35-45s (sequential model loading + warmup inference).

## Repository Structure

```
cropnet-api/
├── api/                          # FastAPI application
│   ├── main.py                   # App entry + route handlers
│   ├── config.py                 # All config (specialists, thresholds, families)
│   ├── auth.py                   # API key authentication
│   ├── schemas.py                # Pydantic request/response models
│   ├── routing.py                # General → specialist routing engine
│   ├── feedback.py               # JSONL feedback logging
│   ├── requirements.txt
│   ├── inference/
│   │   ├── general.py            # ONNX model wrapper + 5-crop TTA
│   │   ├── specialists.py        # Auto-load specialist ONNX models
│   │   ├── preprocess.py         # Image validation + preprocessing
│   │   ├── metadata.py           # Labels, translations, Tunisia tags
│   │   ├── calibration.py        # Threshold logic, confidence levels
│   │   └── quality.py            # Image quality warnings
│   └── locales/
│       ├── ar.json               # Arabic disease names + actions
│       ├── fr.json               # French
│       └── en.json               # English
├── training/
│   ├── common/                   # Shared training code
│   │   ├── train_loop.py         # DINOv2-B fine-tuning with progressive unfreezing
│   │   ├── datasets.py           # ImageFolder + balanced sampling
│   │   ├── transforms.py         # Augmentation pipelines
│   │   ├── losses.py             # MixUp implementation
│   │   ├── metrics.py            # Top-k accuracy, F1, confusion matrix
│   │   ├── utils.py              # Model building, ONNX export helpers
│   │   ├── split.py              # Train/val stratified splitter
│   │   ├── dedup.py              # Perceptual hash dedup
│   │   ├── distribution.py       # Class balance report
│   │   ├── leakage.py            # Train/val leak detection
│   │   └── reports.py            # Markdown report generator
│   ├── general/                  # General model (v4)
│   ├── olive/                    # 🫒 Olive specialist
│   ├── date_palm/                # 🌴 Date palm
│   ├── wheat/                    # 🌾 Wheat
│   ├── citrus/                   # 🍊 Citrus (4-source merged dataset)
│   ├── peach/                    # 🍑 Peach
│   ├── tomato/                   # 🍅 Tomato
│   ├── pepper/                   # 🌶️ Pepper
│   ├── watermelon/               # 🍉 Watermelon
│   ├── almond/                   # 🌰 Almond (scaffolded, awaiting data)
│   ├── pistachio/                # 🥜 Pistachio (scaffolded, awaiting data)
│   └── requirements.txt          # GPU training deps
├── models/                       # Model artifacts (gitignored)
├── tests/                        # 122 pytest tests
├── scripts/
│   ├── install.sh                # Production installer
│   └── cropnet.service           # systemd unit file
├── DEPLOYMENT.md                 # Deployment guide
└── README.md
```

## Training Data Sources

| Specialist | Source | Notes |
|-----------|--------|-------|
| General (v4) | [PlantVillage (augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) | 89 classes after junk filtering |
| Olive | Custom Kaggle collection | 3 classes, 2,720 images |
| Date Palm | Custom Kaggle collection | 3 classes, 2,631 images |
| Wheat | PlantVillage wheat subset | 3 classes, 407 images |
| Citrus | 4 merged sources: PlantVillage orange + [sourabh2001](https://www.kaggle.com/datasets/sourabh2001/citrus-leaves-dataset) + [jonathansilva2020](https://www.kaggle.com/datasets/jonathansilva2020/dataset-for-classification-of-citrus-diseases) + [myprojectdictionary](https://www.kaggle.com/datasets/myprojectdictionary/citrus-leaf-disease-image) | 5 classes, 6,394 images |
| Peach | PlantVillage peach subset | 2 classes, 3,566 images |
| Tomato | PlantVillage tomato subset | 10 classes, 18,345 images |
| Pepper | PlantVillage pepper subset | 2 classes, 3,901 images |
| Watermelon | [nirmalsankalana](https://www.kaggle.com/datasets/nirmalsankalana/watermelon-disease-dataset) | 4 classes, 1,155 images |

## Known Limitations

- **General model routing weakness**: Wheat and citrus sometimes misroute because the general model's top-k labels don't always contain the expected crop prefix. This is a general model quality issue, not a routing bug.
- **Calibration is temperature-only**: Post-hoc Platt scaling hooks exist but aren't yet fitted. `calibrated_confidence` is `null`.
- **89 class labels include 25 junk entries** from PlantVillage (numeric strings, folder artifacts). Filtered at inference time; index mapping preserved for compatibility.
- **Small specialist datasets**: Wheat (407 images) may overfit. Monitor field performance.
- **No almond/pistachio datasets available**: Exhaustive search across Kaggle, Mendeley, Roboflow, PlantVillage — none found.

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI + Uvicorn |
| Inference | ONNX Runtime (CPU, no PyTorch dependency) |
| Training | PyTorch 2.5 + DINOv2-B (facebook/dinov2-base) |
| Image Processing | Pillow + NumPy |
| Production | systemd + 2 Uvicorn workers |
| Testing | pytest (122 tests) |

## Safety

All predictions include:
- Explicit confidence levels (high / moderate / low / very_low)
- Localized disclaimers in AR/FR/EN
- Recommended actions from agricultural best practices
- Image quality warnings when input photos are poor
- `advisory_only: true` flag on every response

The system **prefers honest uncertainty over false certainty**. Low-confidence results are clearly marked with explanations.

## Fellah Integration

CropNet is the primary classifier in Fellah's crop scan pipeline:

```
User selects crop + uploads photo
  → Fellah sends POST /predict with image + crop_hint + locale
  → CropNet returns diagnosis (2-5s on VPS)
  → Fellah checks confidence:
      ≥35% → text-only GPT-4o-mini for treatment advice (cheap, no image)
      <35% → GPT-4o-mini Vision fallback (sends image, expensive)
```

**Key integration details:**
- `crop_hint` from dropdown is critical — general model can misroute (e.g. olive→citrus)
- Fellah timeout: 12s (inference 2-5s with TTA_CROPS=3)
- Response mapping: Fellah reads `primary_prediction.label`, `primary_prediction.confidence`, `top_predictions`
- API key must be Vercel env `plain` type (not `encrypted`)
- Crop dropdown is **required** before scanning — ensures crop_hint always sent

**Known limitations:**
- Specialists trained on PlantVillage **leaf images only** — fruit diseases not recognized
- General model confuses similar leaf shapes (olive↔citrus) — always use crop_hint
- 1 Uvicorn worker on 11GB VPS + 4GB swap; ~6GB RSS with 9 models loaded

---

Built for [Fellah](https://app.fellah.tn) 🌾 — empowering Tunisian farmers with AI.
