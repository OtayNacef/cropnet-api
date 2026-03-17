"""Label loading, translations, and metadata utilities."""
from __future__ import annotations

import json
from pathlib import Path

# Labels known to be training-data leakage (numeric strings, split folder names)
_JUNK = frozenset({
    *(str(i) for i in range(38)),
    "dataset", "date palm data", "diseased", "healthy",
    "test", "train", "valid",
})

TUNISIA_LABELS = frozenset({
    "Olive___Diseased", "Olive___Healthy",
    "Olive___aculus_olearius", "Olive___olive_peacock_spot",
    "Date Palm data", "Palm___Brown_Leaf_Spot",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
})


def load_id2label(path: Path) -> dict[int, str]:
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["id2label"].items()}


def is_junk(label: str) -> bool:
    return label in _JUNK


def is_healthy(label: str) -> bool:
    return "healthy" in label.lower() or "sain" in label.lower()


def severity(label: str, conf: float) -> str:
    if is_healthy(label):
        return "healthy"
    if conf >= 0.75:
        return "severe"
    if conf >= 0.50:
        return "moderate"
    return "mild"


# ── Locale loading ────────────────────────────────────────────────────────────
_locale_cache: dict[str, dict[str, str]] = {}


def _load_locale(lang: str) -> dict[str, str]:
    if lang not in _locale_cache:
        p = Path(__file__).parent.parent / "locales" / f"{lang}.json"
        if p.exists():
            with open(p) as f:
                _locale_cache[lang] = json.load(f)
        else:
            _locale_cache[lang] = {}
    return _locale_cache[lang]


def translate(label: str, lang: str) -> str:
    """Translate a raw label key to a localised display name."""
    table = _load_locale(lang)
    if label in table:
        return table[label]
    # fallback: prettify the raw label
    return label.replace("___", " — ").replace("_", " ")


def recommended_action(label: str, lang: str) -> str:
    """Return a short recommended action for the detected disease."""
    table = _load_locale(lang)
    key = f"action:{label}"
    return table.get(key, "")
