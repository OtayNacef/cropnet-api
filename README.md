# CropNet API v4 🌾

**Tunisia-optimized crop disease detection API.**  
ViT model · 55 classes · ONNX Runtime · 5-crop TTA · AR/FR/EN

---

## Classes

**38 PlantVillage** (tomato, potato, corn, grape, apple, pepper, peach, strawberry, citrus, squash…)  
**+11 Tunisia-specific:**
- 🫒 Olive — Peacock Spot, Olive Knot, Healthy
- 🌴 Date Palm — Bayoud Disease, Black Scorch, Healthy
- 🍋 Citrus — Canker, Greening (HLB), Healthy
- 🌾 Wheat — Septoria Leaf Blotch, Yellow Rust

---

## Quick Start

```bash
# Install (production)
sudo bash scripts/install.sh

# Or run locally
pip install -r app/requirements.txt
MODEL_PATH=./models/cropnet_v2.onnx LABELS_PATH=./models/labels.json \
  uvicorn app.main:app --host 0.0.0.0 --port 8001
```

---

## API Usage

### Health check
```bash
curl https://api.cropnet.fellah.tn/health
```

### Predict
```bash
curl -X POST https://api.cropnet.fellah.tn/predict \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64 encoded image>",
    "locale": "ar",
    "crop_hint": "tomato"
  }'
```

### Response
```json
{
  "disease": "اللفحة المبكرة للطماطم",
  "disease_en": "Tomato Early Blight",
  "disease_raw": "Tomato___Early_blight",
  "confidence": 0.8731,
  "confidence_pct": 87.3,
  "severity": "severe",
  "is_healthy": false,
  "is_tunisia": false,
  "below_threshold": false,
  "top5": [...],
  "crop_agreement": "4/5",
  "inference_ms": 120,
  "model_version": "cropnet-v4-ViT-55cls-TTA5"
}
```

---

## Architecture

```
Farmer photo
     │
     ▼
[Preprocessing]
  CLAHE enhance → 5-crop TTA → ImageNet normalize
     │
     ▼
[ViT ONNX Inference]
  55 classes · temperature scaling · averaged logits
     │
     ▼
[Majority vote]
  Top-5 results · AR/FR/EN translation · severity score
     │
     ▼
[Response]
```

---

## Why ONNX Runtime?

- No PyTorch / CUDA required on server
- 3-5x faster inference vs HuggingFace pipeline
- ~40-80ms per image (5-crop TTA)
- Runs on CPU with 4 threads

---

## Real-World Accuracy Notes

PlantVillage models (lab conditions) perform differently on real farm photos. Mitigations applied:
- **CLAHE** contrast/sharpness enhancement for outdoor/shadow images
- **5-crop TTA** handles off-center, partial leaf captures
- **Temperature scaling (T=1.4)** prevents overconfident predictions on novel inputs
- **`below_threshold` flag** signals to caller when to trigger GPT-4 Vision fallback

---

## Deployment

```
VPS (185.209.229.49)
 └─ Nginx (443 SSL) → uvicorn (127.0.0.1:8001)
      └─ 2 workers, uvloop, httptools
```

See `nginx/cropnet.conf` and `scripts/cropnet.service`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/opt/cropnet/models/cropnet_v2.onnx` | ONNX model path |
| `LABELS_PATH` | `/opt/cropnet/models/labels.json` | Labels JSON |
| `CROPNET_API_KEY` | *(empty = no auth)* | API key for `/predict` |
| `CONF_THRESHOLD` | `0.40` | Min confidence to consider reliable |
| `TTA_CROPS` | `5` | Number of TTA crops (1=off, 5=max) |
| `TEMPERATURE` | `1.4` | Softmax temperature (higher=softer) |
| `IMG_SIZE` | `224` | Input resolution |

---

Built for [Fellah](https://app.fellah.tn) 🌾
