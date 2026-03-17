"""Loss functions and MixUp helper."""
from __future__ import annotations

import numpy as np
import torch


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """MixUp augmentation: blend pairs of samples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam
