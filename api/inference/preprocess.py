"""Image preprocessing: validation, enhancement, TTA crops."""
from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageEnhance, ImageStat

from ..config import IMG_SIZE, MAX_UPLOAD_BYTES, TTA_CROPS

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_DIM = 64
MAX_DIM = 8192
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Validation ────────────────────────────────────────────────────────────────

class ImageCheck(NamedTuple):
    ok: bool
    issue: str


def validate_image_bytes(raw: bytes) -> ImageCheck:
    if len(raw) > MAX_UPLOAD_BYTES:
        return ImageCheck(False, f"Image too large ({len(raw)//1024//1024}MB, max 20MB)")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception:
        return ImageCheck(False, "Corrupt or unsupported image format")
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    if w < MIN_DIM or h < MIN_DIM:
        return ImageCheck(False, f"Image too small ({w}×{h}, min {MIN_DIM})")
    if w > MAX_DIM or h > MAX_DIM:
        return ImageCheck(False, f"Image too large ({w}×{h}, max {MAX_DIM})")
    return ImageCheck(True, "")


# ── Quality warnings ──────────────────────────────────────────────────────────

def quality_warnings(img: Image.Image) -> list[str]:
    warnings: list[str] = []
    stat = ImageStat.Stat(img)
    brightness = sum(stat.mean) / 3
    if brightness < 30:
        warnings.append("very_dark")
    elif brightness > 240:
        warnings.append("overexposed")
    if sum(stat.stddev) / 3 < 10:
        warnings.append("low_variance")
    return warnings


# ── Enhancement / normalisation ───────────────────────────────────────────────

def clahe_enhance(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.20)
    img = ImageEnhance.Color(img).enhance(1.10)
    return img


def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - _MEAN) / _STD


def preprocess(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Resize → float32 → ImageNet-normalise → NCHW."""
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return normalize(arr).transpose(2, 0, 1)[np.newaxis]


# ── TTA ───────────────────────────────────────────────────────────────────────

def five_crops(img: Image.Image, n: int = TTA_CROPS) -> list[Image.Image]:
    w, h = img.size
    cs = int(min(w, h) * 0.90)
    cx, cy = w // 2, h // 2
    coords = [
        (cx - cs//2, cy - cs//2, cx + cs//2, cy + cs//2),
        (0, 0, cs, cs),
        (w - cs, 0, w, cs),
        (0, h - cs, cs, h),
        (w - cs, h - cs, w, h),
    ]
    return [clahe_enhance(img.crop(c)) for c in coords[:n]]
