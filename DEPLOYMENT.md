# CropNet Deployment Guide

## Production Server
- VPS: 185.209.229.49 (Contabo)
- Service: systemd `cropnet`
- API: http://127.0.0.1:8001 (behind nginx)
- Runtime: ONNX Runtime (CPU only, no PyTorch)

## Model Files
```
/opt/cropnet/models/
├── general/
│   └── cropnet-general-v1.onnx (347 MB, 89 classes)
├── olive/
│   ├── cropnet-olive-v1.onnx (331 MB, 3 classes)
│   ├── labels.json
│   └── metadata.json
├── date_palm/
│   ├── cropnet-date_palm-v1.onnx (331 MB, 3 classes)
│   ├── labels.json
│   └── metadata.json
├── wheat/
│   ├── cropnet-wheat-v1.onnx (331 MB, 3 classes)
│   ├── labels.json
│   └── metadata.json
├── citrus/
│   ├── cropnet-citrus-v1.onnx (331 MB, 5 classes)
│   ├── labels.json
│   └── metadata.json
├── tomato/
│   ├── cropnet-tomato-v1.onnx (331 MB, 10 classes)
│   ├── labels.json
│   └── metadata.json
├── pepper/
│   ├── cropnet-pepper-v1.onnx (331 MB, 2 classes)
│   ├── labels.json
│   └── metadata.json
└── watermelon/
    ├── cropnet-watermelon-v1.onnx (331 MB, 4 classes)
    ├── labels.json
    └── metadata.json
```

## Training Infrastructure
- GPU1: ec2-user@54.92.129.192 (A10G 24GB) — olive, wheat, tomato, pepper
- GPU2: ec2-user@54.81.251.93 (A10G 24GB) — date_palm, citrus, watermelon
- Conda env: /data/miniforge3 + /data/cropnet/env (Python 3.11)
- PyTorch 2.5.1 via conda-forge (GLIBC 2.26 workaround)

## Training Results (2026-03-17)

| Model | Val Acc | Epochs | Dataset | GPU |
|-------|---------|--------|---------|-----|
| Olive | 99.63% | 15 | 2,720 imgs (3 cls) | GPU1 |
| Date Palm | 100.0% | 15 | 2,631 imgs (3 cls) | GPU2 |
| Wheat | 100.0% | 15 | 407 imgs (3 cls) | GPU1 |
| Citrus | 98.44% | 15 | 6,394 imgs (5 cls, 4 merged datasets) | GPU2 |
| Tomato | 99.95% | 15 | 18,345 imgs (10 cls, PlantVillage) | GPU1 |
| Pepper | 100.0% | 15 | 3,901 imgs (2 cls, PlantVillage) | GPU1 |
| Watermelon | 100.0% | 15 | 1,155 imgs (4 cls, Kaggle) | GPU2 |

## Updating a Specialist
1. Train on GPU: `python3 training/{crop}/train.py --data-dir <path> --output-dir <out>`
2. Export ONNX: `python3 training/{crop}/export_onnx.py --checkpoint <out>/best.pt --num-classes N --output <out>/cropnet-{crop}-v1.onnx`
3. SCP to prod: `scp <out>/cropnet-{crop}-v1.onnx root@185.209.229.49:/opt/cropnet/models/{crop}/`
4. Copy labels.json + metadata.json
5. Restart: `systemctl restart cropnet`

## Service Commands
```bash
systemctl status cropnet
systemctl restart cropnet
journalctl -u cropnet -f
curl http://127.0.0.1:8001/health
```

## Startup Time
~35-45s with all 8 models (1 general + 7 specialists, ~2.7 GB total)
