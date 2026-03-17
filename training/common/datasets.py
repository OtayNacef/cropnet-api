"""Reusable dataset classes for CropNet training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """Loads images from root/class_name/image.ext structure."""

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.classes: list[str] = sorted(d.name for d in root.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for cls_dir in root.iterdir():
            if not cls_dir.is_dir() or cls_dir.name not in self.class_to_idx:
                continue
            idx = self.class_to_idx[cls_dir.name]
            for p in cls_dir.iterdir():
                if p.suffix.lower() in self.EXTS:
                    self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def class_weights(dataset: ImageFolderDataset) -> torch.Tensor:
    """Inverse-frequency sample weights for WeightedRandomSampler."""
    counts = np.zeros(len(dataset.classes))
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    w = 1.0 / (counts + 1)
    return torch.tensor([w[lbl] for _, lbl in dataset.samples], dtype=torch.float32)
