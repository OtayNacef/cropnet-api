"""Training utilities: model building, unfreezing, ONNX export."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def build_dinov2_classifier(base: str, num_classes: int, device: torch.device) -> nn.Module:
    """Build a DINOv2 classifier with frozen backbone."""
    from transformers import Dinov2Model

    backbone = Dinov2Model.from_pretrained(base)
    hidden = backbone.config.hidden_size

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.3), nn.Linear(hidden, num_classes))
            for p in self.backbone.parameters():
                p.requires_grad = False

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0])

    return Classifier().to(device)


def unfreeze_last_n(model: nn.Module, n: int):
    for block in list(model.backbone.encoder.layer)[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    print(f"  Unfroze last {n} blocks")


def unfreeze_all(model: nn.Module):
    for p in model.backbone.parameters():
        p.requires_grad = True
    print("  Unfroze entire backbone")


def export_onnx(model: nn.Module, img_size: int, path: Path, device: torch.device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model, dummy, str(path), opset_version=17,
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"  ONNX → {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


def save_labels(classes: list[str], path: Path):
    data = {"id2label": {str(i): c for i, c in enumerate(classes)}, "num_classes": len(classes)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Labels → {path} ({len(classes)} classes)")
