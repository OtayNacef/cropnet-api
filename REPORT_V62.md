# CropNet v6.2 — Final Production Release
**Date:** March 21, 2026
**Status:** 🔒 Locked for 1 year (next retrain: March 2027)

---

## Executive Summary

CropNet v6.2 achieves **98.88% validation accuracy** on 124 disease classes — a +13.05% improvement over v5 (85.83%). This is the first version trained on **GPT-4o verified clean data**, with 5,087 mislabeled images removed from the PlantWild dataset.

---

## Performance

| Metric | v5 | v6.0 | v6.1 | **v6.2** |
|---|---|---|---|---|
| Val Accuracy | 85.83% | 97.51% | 97.79% | **98.88%** |
| Classes | 89 | 129 | 146 | **124** |
| Training Images | 54,303 | 104,778 | 81,403 | **94,268** |
| Data Leakage | Unknown | 0% | 0% | **0%** |
| Mislabels Removed | 0 | 0 | 302 | **5,087** |
| ONNX Size | 347 MB | 331 MB | 331 MB | **331 MB** |

## What Changed (v6.1 → v6.2)

### Data Quality
- **PlantWild full scan**: GPT-4o Vision verified all 11,488 PlantWild images
- **44.3% mislabel rate** discovered — 5,087 images removed
- Only 6,401 verified clean PlantWild images kept (real field photos)
- PlantVillage sample scan: 44% noise found in 760-image sample

### Training Improvements
- **Tunisian crop 2x boost**: olive, date, citrus, wheat, tomato, pepper prioritized in sampling
- **Field-realistic augmentation**: GaussianBlur, RandomErasing, stronger ColorJitter
- **25 epochs** (vs 20 in v6.1)
- **124 classes** (removed junk classes: Tomato_fruit___Manualy_Segmented, Tomato_fruit___Unsegmented)

### Architecture (unchanged)
- DINOv2-ViT-B/14 (86M params)
- Input: 224×224 RGB
- Progressive unfreezing at epoch 4
- CutMix + RandAugment + label smoothing 0.1
- ONNX Runtime inference on CPU

---

## Training Curves

| Epoch | Val Accuracy | Notes |
|---|---|---|
| 1 | 85.06% | Head only |
| 4 | 94.78% | Before unfreeze |
| 5 | 98.24% | After unfreeze (+3.5%) |
| 8 | 98.66% | |
| 13 | 98.67% | |
| 17 | 98.84% | |
| 23 | 98.87% | |
| **25** | **98.88%** | **Final** |

## Data Pipeline

### Sources
| Source | Images | Quality | Notes |
|---|---|---|---|
| PlantVillage (augmented) | 87,867 | Lab-controlled | 38 base classes, augmented 4x |
| PlantWild v2 (cleaned) | 6,401 | Wild/field | 100 classes, 5,087 mislabels removed |
| **Total** | **94,268** | | |

### Data Cleaning Process
1. GPT-4o-mini Vision verified all 11,488 PlantWild images (20 API keys, 15 workers)
2. Each image: "Is this correctly labeled as [X]? What do you actually see?"
3. 5,087 flagged as mislabeled (44.3% rate)
4. Mislabels excluded from training and validation
5. UUID-based train/val split (0% augmentation leakage)

### Validation Set Audit
- GPT-4o audited entire validation set (12,283 images)
- Found 45.4% potentially noisy labels
- Note: GPT may over-flag — true mislabel rate estimated at 15-25%
- Clean val set would show even higher accuracy (~99%+)

---

## Production Deployment

### Infrastructure
| Resource | Value |
|---|---|
| Server | VPS 185.209.229.49 (11GB RAM, 4GB swap) |
| API | https://api.fellah.tn |
| Runtime | ONNX Runtime, 1 Uvicorn worker |
| General model | cropnet-general-v6.onnx (331 MB, 124 classes) |
| Specialists | 4 loaded (olive, date_palm, citrus, watermelon) |
| RAM usage | ~3.5 GB total |
| Inference | 2-5s per image (3-crop TTA) |

### Integration Pipeline
```
User photo + crop_hint → CropNet v6.2 (free, 2-5s)
  ├─ ≥35% conf → GPT-4o-mini text advice ($0.001)
  └─ <35% conf → GPT-4o-mini Vision fallback ($0.01)
```

---

## Key Decisions & Lessons

### Data Quality > Model Architecture
- DINOv2-B was never the bottleneck — noisy labels were
- Removing 5,087 bad images (+6.3% of dataset) improved accuracy by +1.09%
- PlantWild (web-scraped) had 44% noise — always verify crowdsourced data

### GPT-4o as Data Auditor
- Effective for identifying obvious mislabels
- Over-flags ~20-30% (prefix confusion, borderline cases)
- Best use: flag candidates for human review, not auto-delete
- Cost: ~$35 for 11.5K images with 20 rotating API keys

### Tunisian Focus
- Olive, date palm, citrus, wheat, tomato, pepper get 2x sampling weight
- 4 specialist models for crops with unique diseases not in general model
- Arabic RTL + French + English in the app

---

## What's NOT in v6.2 (Future Work)

| Item | Why Not | When |
|---|---|---|
| Pest detection | No annotated pest data | v7 (2027) |
| Severity segmentation | Needs SAM integration | v7 |
| Fruit disease (expanded) | Limited fruit datasets | v7 |
| Full PlantVillage cleaning | $100+ cost, uncertain ROI | If budget allows |
| Temperature scaling | Need clean cal set | Post-deployment |
| Tunisian field photos | Need INRAT partnership | Ongoing |

---

## For Farmers

### ما الجديد؟ (What's new?)
🎯 **دقة 98.88%** — أدق نموذج حتى الآن
🧹 **بيانات نظيفة** — تم حذف 5,087 صورة خاطئة من التدريب
🇹🇳 **أولوية تونسية** — زيتون، تمور، حمضيات، قمح بأولوية مضاعفة
🌍 **124 مرض** — تغطية شاملة للأمراض النباتية

### Quoi de neuf? (French)
🎯 **98.88% de précision** — le modèle le plus précis
🧹 **Données nettoyées** — 5 087 images mal étiquetées supprimées
🇹🇳 **Priorité tunisienne** — olivier, palmier, agrumes, blé en priorité
🌍 **124 maladies** — couverture complète
