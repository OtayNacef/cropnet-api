"""
DINOv2 attention heatmap extraction.
====================================
Since the production ONNX models only output logits (no attention weights),
this module uses the HuggingFace DINOv2 PyTorch backbone to extract
self-attention from the last transformer layer.

Flow:
  1. Load DINOv2-base backbone (lazy, cached)
  2. Register a forward hook on the last attention layer
  3. Forward pass → capture attention weights
  4. Extract CLS→patch attention, average across heads
  5. Reshape to spatial grid, upsample, apply JET colormap
  6. Blend with original image (alpha overlay)

The PyTorch model is only loaded when explain is requested (lazy init).
This keeps the normal predict path ONNX-only and fast.
"""
from __future__ import annotations

import io
import logging
import base64
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("cropnet.attention")

# ── Lazy globals ──────────────────────────────────────────────────────────────
_backbone = None
_device = None
_torch = None


def _ensure_torch():
    """Import torch lazily so the main API doesn't pay the import cost."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _load_backbone():
    """Load DINOv2-base backbone (once). Returns (model, device)."""
    global _backbone, _device
    if _backbone is not None:
        return _backbone, _device

    torch = _ensure_torch()
    from transformers import Dinov2Model

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    _backbone = model
    _device = device
    log.info(f"DINOv2 attention backbone loaded in {time.time() - t0:.1f}s on {device}")
    return _backbone, _device


def extract_attention(
    img: Image.Image,
    img_size: int = 256,
    head_reduction: str = "mean",
) -> np.ndarray:
    """
    Extract CLS token attention map from the last DINOv2 layer.

    Parameters
    ----------
    img : PIL Image (RGB)
    img_size : resize target (must match training config)
    head_reduction : how to combine heads ("mean", "max")

    Returns
    -------
    attn_map : np.ndarray of shape (H_patches, W_patches), float32 in [0, 1]
    """
    torch = _ensure_torch()
    backbone, device = _load_backbone()

    # Preprocess: same normalization as training
    img_resized = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Hook into last attention layer to capture attention weights
    attn_weights = {}

    def _hook(module, input, output):
        # Dinov2's attention: output is (attn_output, attn_weights)
        # attn_weights shape: (batch, num_heads, seq_len, seq_len)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attn_weights["last"] = output[1].detach().cpu()

    # The attention module in HuggingFace DINOv2:
    # model.encoder.layer[-1].attention.attention
    last_attn = backbone.encoder.layer[-1].attention.attention
    handle = last_attn.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            # output_attentions=True makes HF return attention weights
            outputs = backbone(pixel_values=tensor, output_attentions=True)

        # HuggingFace returns attentions as a tuple, one per layer
        # Each is (batch, num_heads, seq_len, seq_len)
        if outputs.attentions is not None:
            last_attn_weights = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
        elif "last" in attn_weights:
            last_attn_weights = attn_weights["last"][0]
        else:
            raise RuntimeError("Could not extract attention weights from DINOv2")

        # Convert to numpy
        if hasattr(last_attn_weights, "numpy"):
            attn = last_attn_weights.numpy()
        else:
            attn = last_attn_weights.cpu().numpy()

    finally:
        handle.remove()

    # attn shape: (num_heads, seq_len, seq_len)
    # CLS token is index 0. We want CLS → all patches (skip CLS→CLS)
    # seq_len = 1 (CLS) + num_patches
    num_heads = attn.shape[0]
    cls_attn = attn[:, 0, 1:]  # (num_heads, num_patches)

    if head_reduction == "max":
        cls_attn = cls_attn.max(axis=0)
    else:
        cls_attn = cls_attn.mean(axis=0)  # (num_patches,)

    # DINOv2 with patch_size=14, img_size=256: grid = 256//14 = 18
    patch_size = 14
    grid_h = img_size // patch_size
    grid_w = img_size // patch_size
    num_patches_expected = grid_h * grid_w

    # Handle potential mismatch (DINOv2 may include register tokens)
    if cls_attn.shape[0] > num_patches_expected:
        # DINOv2 v2 has 4 register tokens after CLS, before patch tokens
        # Skip register tokens
        cls_attn = cls_attn[-num_patches_expected:]
    elif cls_attn.shape[0] < num_patches_expected:
        log.warning(
            f"Attention map size {cls_attn.shape[0]} < expected {num_patches_expected}, "
            f"padding with zeros"
        )
        padded = np.zeros(num_patches_expected, dtype=np.float32)
        padded[: cls_attn.shape[0]] = cls_attn
        cls_attn = padded

    attn_map = cls_attn.reshape(grid_h, grid_w)

    # Normalize to [0, 1]
    attn_min, attn_max = attn_map.min(), attn_map.max()
    if attn_max - attn_min > 1e-8:
        attn_map = (attn_map - attn_min) / (attn_max - attn_min)
    else:
        attn_map = np.zeros_like(attn_map)

    return attn_map.astype(np.float32)


def generate_heatmap_overlay(
    original_img: Image.Image,
    attn_map: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    output_size: Optional[tuple[int, int]] = None,
) -> Image.Image:
    """
    Generate a heatmap overlay on the original image.

    Parameters
    ----------
    original_img : PIL RGB image
    attn_map : 2D float array in [0, 1] (e.g. 18×18)
    alpha : blending factor (0 = only original, 1 = only heatmap)
    colormap : OpenCV colormap
    output_size : (width, height) for output, defaults to original image size

    Returns
    -------
    blended : PIL RGB image with heatmap overlay
    """
    if output_size is None:
        output_size = original_img.size  # (W, H)

    # Resize original to output size
    orig = original_img.resize(output_size, Image.LANCZOS)
    orig_arr = np.array(orig)  # (H, W, 3) RGB

    # Upsample attention map to output size
    h, w = output_size[1], output_size[0]
    attn_upsampled = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # Clamp after resize
    attn_upsampled = np.clip(attn_upsampled, 0, 1)

    # Apply colormap (expects uint8)
    heatmap = cv2.applyColorMap((attn_upsampled * 255).astype(np.uint8), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    blended = (orig_arr.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def heatmap_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    """Encode a PIL image as base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def explain(
    img: Image.Image,
    img_size: int = 256,
    alpha: float = 0.4,
    output_size: Optional[tuple[int, int]] = None,
) -> dict:
    """
    Full explain pipeline: extract attention → generate heatmap → encode.

    Returns
    -------
    dict with keys:
      - heatmap_base64: base64 JPEG of the overlaid heatmap
      - attention_raw: list of list (the raw attention grid, for debugging)
      - grid_size: [H, W] of the attention grid
      - extraction_ms: time to extract + render
    """
    t0 = time.time()

    attn_map = extract_attention(img, img_size=img_size)
    overlay = generate_heatmap_overlay(
        img, attn_map, alpha=alpha, output_size=output_size or (512, 512)
    )
    b64 = heatmap_to_base64_jpeg(overlay)

    ms = int((time.time() - t0) * 1000)
    grid_h, grid_w = attn_map.shape

    return {
        "heatmap_base64": b64,
        "attention_grid": attn_map.tolist(),
        "grid_size": [grid_h, grid_w],
        "extraction_ms": ms,
    }
