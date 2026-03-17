# CropNet Deployment Guide

## Production Server

| Field | Value |
|-------|-------|
| VPS | 185.209.229.49 (Contabo) |
| Service | systemd `cropnet` |
| API | `http://127.0.0.1:8001` (behind nginx) |
| Runtime | ONNX Runtime (CPU only, no PyTorch in prod) |
| Workers | 2 Uvicorn workers |
| Python | venv at `/opt/cropnet/venv/` |
| Version | **v5.0.0** |
| Startup time | ~35-45 seconds (9 ONNX models, ~3 GB) |

## Model Files (Production)

```
/opt/cropnet/models/
├── general/
│   ├── cropnet-general-v1.onnx     (347 MB, 89 classes, DINOv2-B v4)
│   ├── labels.json
│   └── metadata.json
├── olive/
│   ├── cropnet-olive-v1.onnx       (331 MB, 3 classes, 99.63% val acc)
│   ├── labels.json
│   └── metadata.json
├── date_palm/
│   ├── cropnet-date_palm-v1.onnx   (331 MB, 3 classes, 100% val acc)
│   ├── labels.json
│   └── metadata.json
├── wheat/
│   ├── cropnet-wheat-v1.onnx       (331 MB, 3 classes, 100% val acc)
│   ├── labels.json
│   └── metadata.json
├── citrus/
│   ├── cropnet-citrus-v1.onnx      (331 MB, 5 classes, 98.44% val acc)
│   ├── labels.json
│   └── metadata.json
├── peach/
│   ├── cropnet-peach-v1.onnx       (331 MB, 2 classes, 100% val acc)
│   ├── labels.json
│   └── metadata.json
├── tomato/
│   ├── cropnet-tomato-v1.onnx      (331 MB, 10 classes, 99.95% val acc)
│   ├── labels.json
│   └── metadata.json
├── pepper/
│   ├── cropnet-pepper-v1.onnx      (331 MB, 2 classes, 100% val acc)
│   ├── labels.json
│   └── metadata.json
└── watermelon/
    ├── cropnet-watermelon-v1.onnx   (331 MB, 4 classes, 100% val acc)
    ├── labels.json
    └── metadata.json
```

**Total disk: ~3 GB** (1 general + 8 specialists).

## Service Commands

```bash
# Status & logs
systemctl status cropnet
journalctl -u cropnet -f

# Restart (after model updates)
systemctl restart cropnet

# Health check (wait ~45s after restart)
curl -s http://127.0.0.1:8001/health | python3 -m json.tool

# Quick model count check
curl -s http://127.0.0.1:8001/health | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'{len(d[\"specialists_loaded\"])} specialists loaded')
for k,v in d['specialists_loaded'].items(): print(f'  {k}: Tier {v[\"tier\"]}')
"
```

## Systemd Unit

```ini
# /etc/systemd/system/cropnet.service
[Unit]
Description=CropNet API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/cropnet
EnvironmentFile=/opt/cropnet/.env
ExecStart=/opt/cropnet/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8001 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Updating a Specialist Model

### 1. Train on GPU
```bash
ssh ec2-user@<gpu-ip>
cd /data/cropnet/repo
python3 training/{crop}/train.py \
  --data-dir /data/datasets/{crop} \
  --output-dir /data/cropnet/output/{crop}
```

### 2. Export ONNX
```bash
python3 training/{crop}/export_onnx.py \
  --checkpoint /data/cropnet/output/{crop}/best.pt \
  --num-classes N \
  --output /data/cropnet/output/{crop}/cropnet-{crop}-v1.onnx
```

### 3. Deploy to production
```bash
# From GPU instance → prod VPS
scp /data/cropnet/output/{crop}/cropnet-{crop}-v1.onnx \
  root@185.209.229.49:/opt/cropnet/models/{crop}/
scp /data/cropnet/output/{crop}/labels.json \
  root@185.209.229.49:/opt/cropnet/models/{crop}/
scp /data/cropnet/output/{crop}/metadata.json \
  root@185.209.229.49:/opt/cropnet/models/{crop}/
```

### 4. Enable & restart
```bash
# On prod VPS
echo "ENABLE_SPECIALIST_{CROP}=true" >> /opt/cropnet/.env  # if new
systemctl restart cropnet
sleep 45
curl -s http://127.0.0.1:8001/health  # verify specialist loaded
```

## Adding a New Specialist

1. **Create training dir**: `mkdir training/{crop}` with `config.yaml`, `train.py`, `eval.py`, `export_onnx.py`
2. **Add to `api/config.py`**:
   - Add entry in `SPECIALISTS` dict (tier, threshold, description)
   - Add label prefix mapping in `CROP_FAMILIES` dict
3. **Train, export, deploy** (steps above)
4. **Update prod `.env`**: `ENABLE_SPECIALIST_{CROP}=true`
5. **Restart service**

## Training Infrastructure

| Instance | IP | GPU | Trained |
|----------|-----|-----|---------|
| GPU1 | `ec2-user@54.92.129.192` | NVIDIA A10G 24GB | olive, wheat, tomato, pepper, peach |
| GPU2 | `ec2-user@54.81.251.93` | NVIDIA A10G 24GB | date_palm, citrus, watermelon |

Both instances:
- Amazon Linux 2, CUDA 12.2
- Miniforge at `/data/miniforge3`
- Conda env at `/data/cropnet/env` (Python 3.11, PyTorch 2.5.1 via conda-forge)
- Repo at `/data/cropnet/repo/`
- Data at `/data/datasets/`
- NVMe 232GB at `/data`

**⚠️ Stop GPU instances when not training** — they burn money when idle.

## Training Results (2026-03-17 & 2026-03-18)

| Model | Val Acc | Epochs | Dataset Size | Classes | GPU | Date |
|-------|---------|--------|-------------|---------|-----|------|
| General (v4) | 85.83% | 12 | PlantVillage full | 89 | GPU1 | 2026-03-16 |
| 🫒 Olive | 99.63% | 15 | 2,720 | 3 | GPU1 | 2026-03-17 |
| 🌴 Date Palm | 100.0% | 15 | 2,631 | 3 | GPU2 | 2026-03-17 |
| 🌾 Wheat | 100.0% | 15 | 407 | 3 | GPU1 | 2026-03-17 |
| 🍊 Citrus | 98.44% | 15 | 6,394 | 5 | GPU2 | 2026-03-17 |
| 🍑 Peach | 100.0% | 15 | 3,566 | 2 | GPU1 | 2026-03-18 |
| 🍅 Tomato | 99.95% | 15 | 18,345 | 10 | GPU1 | 2026-03-17 |
| 🌶️ Pepper | 100.0% | 15 | 3,901 | 2 | GPU1 | 2026-03-17 |
| 🍉 Watermelon | 100.0% | 15 | 1,155 | 4 | GPU2 | 2026-03-17 |

## Integration with Fellah

CropNet is called by the Fellah web app at `apps/web/app/api/crop-scan/route.ts`:

```
User photo → Fellah API → CropNet /predict (4s timeout)
                                ↓
                    If CropNet conf ≥ 0.50:
                      → GPT-4o-mini text-only (treatment advice)
                    If CropNet conf < 0.50 or unavailable:
                      → GPT-4o-mini Vision (full analysis)
```

Env vars on Vercel:
- `CROPNET_API_URL` — CropNet base URL
- `CROPNET_API_KEY` — API authentication key

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Service won't start | Model expects 256×256 but `IMG_SIZE=224` in `.env` | Set `IMG_SIZE=256` in `/opt/cropnet/.env` |
| Specialist not loading | ONNX file missing or wrong path | Check `/opt/cropnet/models/{crop}/cropnet-{crop}-v1.onnx` exists |
| Slow startup (~60s+) | Loading 9 models sequentially | Normal — wait for all models to load |
| OOM during training | Batch size too large for A10G | Reduce to `batch_size=32` for datasets >5k images |
| GLIBC error on Amazon Linux 2 | System GLIBC 2.26 too old | Use conda-forge `pytorch-gpu` (bundles its own libs) |
