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

---

## Training Datasets

CropNet v3 is trained on a merged dataset of **15+ sources** covering real field conditions, lab images, and Tunisia/Mediterranean-specific crops.

### Base Dataset

| Dataset | Images | Classes | Source |
|---|---|---|---|
| PlantVillage | ~54,000 | 38 | [dpdl-benchmark/plant_village](https://huggingface.co/datasets/dpdl-benchmark/plant_village) |
| PlantVillage (augmented) | ~87,000 | 38 | [BrandonFors/Plant-Diseases-PlantVillage-Dataset](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) |

> Hughes, D.P. & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060

### Mediterranean / Tunisia Crops

| Dataset | Crop | Diseases | Source |
|---|---|---|---|
| Wheat Field Disease | Wheat | Yellow Rust, Brown Rust, Septoria, Powdery Mildew, Healthy | [Zenodo 7307816](https://zenodo.org/records/7307816) — **real field conditions** |
| Wheat Fungi Diseases (WFD) | Wheat | Leaf Rust, Stem Rust, Septoria, Mildew | [wfd.sysbio.ru](https://wfd.sysbio.ru/) |
| Wheat Plant Diseases | Wheat | Stripe Rust, Leaf Rust, Healthy | [gtsaidata/Wheat-Plant-Diseases-Dataset](https://huggingface.co/datasets/gtsaidata/Wheat-Plant-Diseases-Dataset) |
| Olive Leaf Image Dataset | Olive | Peacock Spot, Aculus Olearius, Healthy | [Kaggle — habibulbasher01644](https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset) |
| Date Palm Leaf Diseases | Date Palm | Black Scorch, Fusarium Wilt, Rachis Blight, Leaf Spots, Pest (Parlatoria) | [PMC 2024 — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340924008965) |
| Grape Leaf Disease | Grape | Black Rot, Esca, Leaf Blight, Healthy | [adamkatchee/grape-leaf-disease-augmented-dataset](https://huggingface.co/datasets/adamkatchee/grape-leaf-disease-augmented-dataset) |
| Citrus Disease | Citrus | Canker, Greening (HLB), Healthy | [mohitsha03/citrus-disease-classification](https://huggingface.co/datasets/mohitsha03/citrus-disease-classification) |

### Additional Sources

| Dataset | Crops | Source |
|---|---|---|
| Plant Disease Image Dataset | Multi-crop | [gtsaidata/Plant-Disease-Image-Dataset](https://huggingface.co/datasets/gtsaidata/Plant-Disease-Image-Dataset) |
| Plant Disease Recognition | Multi-crop | [NouRed/plant-disease-recognition](https://huggingface.co/datasets/NouRed/plant-disease-recognition) |
| Bangladesh Multi-Crop | Banana, Wheat, Rice | [Saon110/bd-crop-vegetable-plant-disease-dataset](https://huggingface.co/datasets/Saon110/bd-crop-vegetable-plant-disease-dataset) |
| Tomato Disease Detection | Tomato | [Amani-Djole/tomato-disease](https://huggingface.co/datasets/Amani-Djole/tomato-disease) |
| Maize Disease Detection | Maize | [Abonia1/maize-disease-detection](https://huggingface.co/datasets/Abonia1/maize-disease-detection) |
| Potato Disease Detection | Potato | [Abonia1/potato-disease-detection](https://huggingface.co/datasets/Abonia1/potato-disease-detection) |

### Base Model

| Model | Architecture | Pretrained On | Source |
|---|---|---|---|
| PlantDiseaseDetectorVit2 | Vision Transformer (ViT-Base/16) | ImageNet → PlantVillage | [Abhiram4/PlantDiseaseDetectorVit2](https://huggingface.co/Abhiram4/PlantDiseaseDetectorVit2) |

### Why Real Field Images Matter

Standard PlantVillage images are taken in controlled lab conditions (white backgrounds, isolated leaves). Real farm photos have shadows, partial leaves, soil, and variable lighting. CropNet v3 specifically includes datasets collected **in-field** (Zenodo wheat, WFD) to improve real-world accuracy.

---

## Citation

If you use CropNet in your research or product:

```bibtex
@software{cropnet2026,
  title   = {CropNet API — Tunisia-Optimized Crop Disease Detection},
  author  = {Fellah (fellah.tn)},
  year    = {2026},
  url     = {https://github.com/OtayNacef/cropnet-api},
  note    = {ViT fine-tuned on 15+ datasets, 55+ classes, ONNX Runtime}
}
```

---

Built for [Fellah](https://app.fellah.tn) 🌾
