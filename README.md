# CropNet API 🌾

AI-powered crop disease detection API for **Fellah** — the Tunisian agriculture platform.

## Model: CropNet v4

| Spec | Value |
|------|-------|
| Base model | DINOv2-B (facebook/dinov2-base) |
| Val accuracy | **85.82537999293037%** |
| Classes | 89 |
| Export format | ONNX (opset 17) |
| Inference | ONNX Runtime + 5-crop TTA |
| Model size | ~331M |

### Training details
- **139,668 images** from 13 datasets (PlantVillage, Kaggle olive/date palm, CGIAR, GitHub)
- Tunisia-specific: olive (3 disease types), date palm (4 types), wheat, citrus
- WeightedRandomSampler for class balance
- MixUp augmentation (alpha=0.4)
- 16 epochs, cosine LR decay (3e-4 to 1e-6)
- Progressive unfreezing (head only, last 4 blocks, full)
- Label smoothing 0.1
- FP16 mixed precision on NVIDIA A10G

### Training lineage
| Version | Base | Accuracy | Classes | Notes |
|---------|------|----------|---------|-------|
| v2 | ViT-B/16 | 99.3% | 55 | PlantVillage only, overfit |
| v3 | PlantDiseaseDetectorVit2 | ~77% | 89 | +Mediterranean data |
| v4 | DINOv2-B | 85.82537999293037% | 89 | +MixUp, +WeightedSampler, +progressive unfreeze |

### Datasets
| Dataset | Source | Images | Focus |
|---------|--------|--------|-------|
| PlantVillage | HuggingFace | 54,303 | 38 crop-disease classes |
| PlantVillage (brandon) | HuggingFace | 43,456 | Alternate split |
| olive_zeytin | Kaggle | 954 | Turkish olive (healthy/diseased) |
| olive_habib | Kaggle | 2,742 | Olive (Aculus/Peacock Spot) |
| olive_tacna | GitHub | ~500 | Olive (Fumagina/Virosis/Deficiency) |
| date_palm_hadjer | Kaggle | 5,262 | Date palm diseases |
| palm_brown_spot | Kaggle | 951 | Palm brown leaf spot |
| cassava_disease | HuggingFace | ~10k | Cassava leaf diseases |
| mango_disease | HuggingFace | ~4k | Mango leaf diseases |
| pomegranate_disease | HuggingFace | ~1.5k | Pomegranate classification |
| agri_pests_diseases | HuggingFace | ~15k | Agricultural pests |
| plant_disease_ayerr | HuggingFace | 194 | Mixed plant diseases |
| saon110 | HuggingFace | ~3k | Plant diseases |

## API

### Stack
- **FastAPI** + uvicorn
- **ONNX Runtime** (CPU, no PyTorch needed in production)
- **5-crop TTA** (center + 4 corners at 90% scale)
- **Temperature scaling** T=1.4 for calibrated confidence
- **Confidence threshold** 0.40 (below falls back to GPT Vision)
- **Localization** AR/FR/EN

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | None | Health + model info |
| POST | /predict | API key | Predict crop disease from image |
| POST | /feedback | API key | Submit label correction |

### Deploy
```bash
curl -fsSL https://raw.githubusercontent.com/OtayNacef/cropnet-api/main/install.sh | bash
```

### Environment
```env
CROPNET_API_KEY=your-api-key
MODEL_PATH=/opt/cropnet/models/cropnet_v2.onnx
LABELS_PATH=/opt/cropnet/models/labels.json
```

## Architecture
```
/opt/cropnet/
  api/
    main.py          # FastAPI app
    inference.py     # ONNX Runtime + TTA
    locales/         # AR/FR/EN translations
  models/
    cropnet_v2.onnx  # Active model (replaced on deploy)
    labels.json      # Class labels + metadata
  .env               # API key + paths
  venv/              # Python virtualenv
```

## License
Private — Fellah project. (c) 2026 Nacef Otay.
