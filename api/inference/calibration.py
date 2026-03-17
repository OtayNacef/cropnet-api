"""
Confidence calibration, threshold logic, and advisory text.

Current implementation: temperature scaling (applied in general.py) + threshold-based
assessment. Per-model calibration parameters can be loaded from metadata.json beside
each ONNX model.

TODO (when calibration training data available):
- Platt scaling per model (logistic regression on held-out logits)
- Store calibration params in metadata.json: {"calibration": {"method": "platt", "a": ..., "b": ...}}
- Apply: calibrated = 1 / (1 + exp(a * logit + b))
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import (
    ADVISORY_ONLY_THRESHOLD,
    GENERAL_THRESHOLD,
    LOW_CONF_THRESHOLD,
    SPECIALIST_THRESHOLD,
)


# ── Per-model calibration metadata ────────────────────────────────────────────

class CalibrationParams:
    """Calibration state for a single model. Currently stores thresholds only."""

    def __init__(self, conf_threshold: float = GENERAL_THRESHOLD, method: str = "none"):
        self.conf_threshold = conf_threshold
        self.method = method  # "none" | "temperature" | "platt" (future)
        self.a: float | None = None  # Platt param
        self.b: float | None = None  # Platt param

    def calibrate(self, raw_conf: float) -> float:
        """Apply calibration transform. Returns calibrated confidence."""
        if self.method == "platt" and self.a is not None and self.b is not None:
            import math
            return 1.0 / (1.0 + math.exp(self.a * raw_conf + self.b))
        # temperature scaling is already applied in inference (TEMPERATURE config)
        return raw_conf

    @classmethod
    def from_metadata(cls, metadata_path: Path, default_threshold: float) -> "CalibrationParams":
        """Load calibration from metadata.json if it exists."""
        params = cls(conf_threshold=default_threshold)
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                cal = meta.get("calibration", {})
                params.method = cal.get("method", "none")
                params.conf_threshold = cal.get("threshold", default_threshold)
                params.a = cal.get("a")
                params.b = cal.get("b")
            except Exception:
                pass
        return params


def load_general_calibration() -> CalibrationParams:
    from ..config import general_metadata_path
    return CalibrationParams.from_metadata(general_metadata_path(), GENERAL_THRESHOLD)


def load_specialist_calibration(key: str) -> CalibrationParams:
    from ..config import SPECIALISTS, specialist_metadata_path
    threshold = SPECIALISTS.get(key, {}).get("conf", SPECIALIST_THRESHOLD)
    return CalibrationParams.from_metadata(specialist_metadata_path(key), threshold)


# ── Confidence assessment ─────────────────────────────────────────────────────

def assess(conf: float) -> str:
    """Return a human-readable confidence level string."""
    if conf >= 0.75:
        return "high"
    if conf >= GENERAL_THRESHOLD:
        return "moderate"
    if conf >= LOW_CONF_THRESHOLD:
        return "low"
    return "very_low"


def is_below_threshold(conf: float, threshold: float = GENERAL_THRESHOLD) -> bool:
    return conf < threshold


def is_advisory_only(conf: float) -> bool:
    """Below advisory threshold = refuse to make any assertion."""
    return conf < ADVISORY_ONLY_THRESHOLD


# ── Localized advisory text ───────────────────────────────────────────────────

_ADVISORY = {
    "very_low": {
        "ar": "❌ لم يتمكن النظام من تحديد المرض بثقة كافية. يرجى التقاط صورة أوضح أو استشارة مختص.",
        "fr": "❌ Le système n'a pas pu identifier avec suffisamment de confiance. Prenez une photo plus nette ou consultez un spécialiste.",
        "en": "❌ The system could not identify the disease with enough confidence. Take a clearer photo or consult a specialist.",
    },
    "low": {
        "ar": "⚠️ مستوى الثقة منخفض. هذه النتيجة استرشادية فقط.",
        "fr": "⚠️ Confiance faible. Résultat indicatif uniquement.",
        "en": "⚠️ Low confidence. This result is advisory only.",
    },
    "moderate": {
        "ar": "ℹ️ تشخيص استرشادي. يرجى التأكد من مختص.",
        "fr": "ℹ️ Diagnostic indicatif. Veuillez confirmer avec un spécialiste.",
        "en": "ℹ️ Advisory diagnosis. Please confirm with a specialist.",
    },
    "high": {
        "ar": "✅ ثقة عالية. تحقق مع مختص للقرارات المهمة.",
        "fr": "✅ Confiance élevée. Vérifiez avec un spécialiste pour les décisions critiques.",
        "en": "✅ High confidence. Verify with a specialist for critical decisions.",
    },
}

_DISCLAIMER = {
    "ar": "تشخيص ذكاء اصطناعي للاسترشاد فقط. استشر مهندساً زراعياً للقرارات المهمة.",
    "fr": "Diagnostic IA à titre indicatif. Consultez un agronome pour les décisions critiques.",
    "en": "AI-generated agricultural guidance. Verify with an agronomist for critical decisions.",
}


def advisory_text(level: str, lang: str) -> str:
    return _ADVISORY.get(level, _ADVISORY["moderate"]).get(lang, _ADVISORY[level]["en"])


def disclaimer(lang: str) -> str:
    return _DISCLAIMER.get(lang, _DISCLAIMER["en"])
