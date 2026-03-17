"""Unit tests for calibration and threshold logic."""
from api.inference.calibration import (
    assess,
    is_below_threshold,
    is_advisory_only,
    advisory_text,
    disclaimer,
    CalibrationParams,
)


class TestAssess:
    def test_high(self):
        assert assess(0.85) == "high"

    def test_moderate(self):
        assert assess(0.50) == "moderate"

    def test_low(self):
        assert assess(0.30) == "low"

    def test_very_low(self):
        assert assess(0.10) == "very_low"

    def test_boundary_high(self):
        assert assess(0.75) == "high"

    def test_boundary_moderate(self):
        assert assess(0.40) == "moderate"


class TestThresholds:
    def test_below_general(self):
        assert is_below_threshold(0.35, 0.40)
        assert not is_below_threshold(0.45, 0.40)

    def test_advisory_only(self):
        assert is_advisory_only(0.10)
        assert not is_advisory_only(0.20)


class TestAdvisoryText:
    def test_all_levels_all_langs(self):
        for level in ("very_low", "low", "moderate", "high"):
            for lang in ("ar", "fr", "en"):
                text = advisory_text(level, lang)
                assert len(text) > 0, f"Empty for {level}/{lang}"

    def test_very_low_has_warning(self):
        assert "❌" in advisory_text("very_low", "en")

    def test_high_has_checkmark(self):
        assert "✅" in advisory_text("high", "en")

    def test_fallback_to_english(self):
        text = advisory_text("moderate", "xx")
        assert len(text) > 0  # falls back


class TestDisclaimer:
    def test_all_langs(self):
        for lang in ("ar", "fr", "en"):
            text = disclaimer(lang)
            assert len(text) > 10

    def test_english_content(self):
        assert "agronomist" in disclaimer("en").lower()

    def test_fallback(self):
        assert len(disclaimer("xx")) > 0


class TestCalibrationParams:
    def test_default_passthrough(self):
        cal = CalibrationParams()
        assert cal.calibrate(0.75) == 0.75

    def test_threshold(self):
        cal = CalibrationParams(conf_threshold=0.5)
        assert cal.conf_threshold == 0.5
