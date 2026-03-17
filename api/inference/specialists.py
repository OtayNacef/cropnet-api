"""Specialist model loader — same engine as general, separate instances."""
from __future__ import annotations

from pathlib import Path

from .general import GeneralModel   # same class, different weights
from .metadata import load_id2label
from ..config import SPECIALISTS, specialist_labels_path, specialist_onnx_path


def load_available_specialists() -> dict[str, GeneralModel]:
    """Scan SPECIALISTS registry; load those whose ONNX file exists on disk."""
    loaded: dict[str, GeneralModel] = {}
    for key, cfg in SPECIALISTS.items():
        if not cfg["enabled"]:
            continue
        onnx = specialist_onnx_path(key)
        labels_p = specialist_labels_path(key)
        if not onnx.exists() or not labels_p.exists():
            print(f"[CropNet:specialist:{key}] ONNX or labels missing — skipped")
            continue
        try:
            id2label = load_id2label(labels_p)
            model = GeneralModel(onnx, id2label, img_size=cfg["img_size"], name=f"specialist-{key}")
            model.warmup(n=1)
            loaded[key] = model
        except Exception as e:
            print(f"[CropNet:specialist:{key}] failed to load: {e}")
    return loaded
