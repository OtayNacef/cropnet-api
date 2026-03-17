"""
Shared training loop — used by all crop-specific train.py scripts.

Produces artifacts:
  - best.pt (checkpoint)
  - cropnet-{crop}-v1.onnx (ONNX export)
  - labels.json (id2label map)
  - metadata.json (training config, lineage, metrics)
  - report.json (training summary)

Usage:
    from training.common.train_loop import run
    run(config_path="training/olive/config.yaml", data_dir="/data/olive", output_dir="/output/olive")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

from .datasets import ImageFolderDataset, class_weights
from .losses import mixup_data
from .metrics import Meter
from .transforms import train_transform, val_transform
from .utils import build_dinov2_classifier, export_onnx, save_labels, unfreeze_all, unfreeze_last_n


def run(config_path: str, data_dir: str, output_dir: str, seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop = cfg["crop"]
    img_size = cfg["img_size"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]

    print(f"\n{'='*60}")
    print(f"CropNet Training: {crop}")
    print(f"Device: {device} | Epochs: {epochs} | BS: {batch_size} | LR: {lr}")
    print(f"Seed: {seed} | Config: {config_path}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    data_root = Path(data_dir)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    auto_split = False

    if not train_dir.exists():
        train_dir = data_root
        val_dir = None

    train_ds = ImageFolderDataset(train_dir, train_transform(img_size))
    classes = train_ds.classes
    num_classes = len(classes)
    print(f"Train: {len(train_ds)} images, {num_classes} classes")
    for i, c in enumerate(classes):
        count = sum(1 for _, l in train_ds.samples if l == i)
        print(f"  [{i:3d}] {c}: {count}")

    if val_dir and val_dir.exists():
        val_ds = ImageFolderDataset(val_dir, val_transform(img_size))
    else:
        n_train = int(0.9 * len(train_ds))
        n_val = len(train_ds) - n_train
        train_ds, val_ds = random_split(train_ds, [n_train, n_val])
        auto_split = True
        print(f"Auto-split: {n_train} train / {n_val} val")

    # Weighted sampling for class imbalance
    if hasattr(train_ds, "samples"):
        weights = class_weights(train_ds)
        sampler = WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=cfg.get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_dinov2_classifier(cfg["base_model"], num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=cfg.get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history: list[dict] = []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # Progressive unfreezing
        if epoch == cfg.get("unfreeze_partial_epoch", 999):
            unfreeze_last_n(model, 4)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.1, weight_decay=cfg.get("weight_decay", 0.01))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)
        if epoch == cfg.get("unfreeze_full_epoch", 999):
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.01, weight_decay=cfg.get("weight_decay", 0.01))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)

        # Train
        model.train()
        loss_meter = Meter()
        correct = total = 0
        alpha = cfg.get("mixup_alpha", 0)
        use_mixup = alpha > 0 and epoch <= cfg.get("unfreeze_full_epoch", epochs)

        for imgs, labels in tqdm(train_loader, desc=f"E{epoch}/{epochs} train"):
            imgs, labels = imgs.to(device), labels.to(device)
            if use_mixup:
                imgs, la, lb, lam = mixup_data(imgs, labels, alpha)
                logits = model(imgs)
                loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()
            loss_meter.update(loss.item(), imgs.size(0))
            _, pred = logits.max(1)
            correct += pred.eq(labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_acc = correct / total * 100 if total else 0

        # Val
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, pred = model(imgs).max(1)
                vc += pred.eq(labels).sum().item()
                vt += imgs.size(0)
        val_acc = vc / vt * 100 if vt else 0

        lr_now = scheduler.get_last_lr()[0]
        print(f"  E{epoch}: loss={loss_meter.avg:.4f} train={train_acc:.1f}% val={val_acc:.1f}% lr={lr_now:.2e}")

        history.append({"epoch": epoch, "loss": round(loss_meter.avg, 4), "train_acc": round(train_acc, 2), "val_acc": round(val_acc, 2), "lr": lr_now})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out / "best.pt")
            print(f"  ★ new best = {val_acc:.2f}%")

    # ── Export ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Best val accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    onnx_name = f"cropnet-{crop}-v1.onnx"
    export_onnx(model, img_size, out / onnx_name, device)
    save_labels(classes, out / "labels.json")

    # Metadata (lineage, config, version tracking)
    metadata = {
        "crop": crop,
        "version": "v1",
        "base_model": cfg["base_model"],
        "best_val_acc": round(best_acc, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "img_size": img_size,
        "label_smoothing": cfg.get("label_smoothing", 0),
        "mixup_alpha": cfg.get("mixup_alpha", 0),
        "num_classes": num_classes,
        "classes": classes,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "auto_split": auto_split,
        "device": str(device),
        "config_path": config_path,
        "data_dir": data_dir,
        "onnx_file": onnx_name,
        "calibration": {"method": "temperature", "temperature": 1.4, "threshold": 0.45},
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Report
    report = {
        "crop": crop,
        "best_val_acc": round(best_acc, 2),
        "epochs": epochs,
        "num_classes": num_classes,
        "classes": classes,
        "date": time.strftime("%Y-%m-%d"),
        "history": history,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Artifacts: {onnx_name}, labels.json, metadata.json, report.json, best.pt")
    print(f"✅ {crop} training complete")
