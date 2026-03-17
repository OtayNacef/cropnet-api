"""Unit tests for label metadata and translation."""
from api.inference.metadata import (
    is_junk,
    is_healthy,
    translate,
    severity,
    recommended_action,
    TUNISIA_LABELS,
)


class TestJunkDetection:
    def test_numeric_strings(self):
        for i in range(38):
            assert is_junk(str(i)), f"'{i}' should be junk"

    def test_folder_names(self):
        assert is_junk("train")
        assert is_junk("valid")
        assert is_junk("test")
        assert is_junk("dataset")

    def test_real_labels(self):
        assert not is_junk("Olive___Healthy")
        assert not is_junk("Tomato___Early_blight")
        assert not is_junk("Apple___Apple_scab")


class TestHealthy:
    def test_olive_healthy(self):
        assert is_healthy("Olive___Healthy")

    def test_tomato_healthy(self):
        assert is_healthy("Tomato___healthy")

    def test_diseased(self):
        assert not is_healthy("Olive___Diseased")
        assert not is_healthy("Tomato___Early_blight")


class TestTranslation:
    def test_arabic(self):
        assert translate("Olive___olive_peacock_spot", "ar") == "عين الطاووس في الزيتون"

    def test_french(self):
        assert translate("Tomato___Late_blight", "fr") == "Mildiou de la tomate"

    def test_english(self):
        assert translate("Tomato___Late_blight", "en") == "Tomato Late Blight"

    def test_fallback_prettifies(self):
        result = translate("Unknown___Some_Disease", "en")
        assert "Unknown" in result
        assert "Some Disease" in result or "Some_Disease" in result

    def test_all_locales_have_olive(self):
        for lang in ("ar", "fr", "en"):
            result = translate("Olive___olive_peacock_spot", lang)
            assert result != "Olive___olive_peacock_spot", f"No translation for {lang}"


class TestSeverity:
    def test_healthy(self):
        assert severity("Olive___Healthy", 0.95) == "healthy"

    def test_severe(self):
        assert severity("Olive___Diseased", 0.85) == "severe"

    def test_moderate(self):
        assert severity("Olive___Diseased", 0.60) == "moderate"

    def test_mild(self):
        assert severity("Olive___Diseased", 0.30) == "mild"


class TestRecommendedAction:
    def test_known_disease_en(self):
        action = recommended_action("Olive___olive_peacock_spot", "en")
        assert "fungicide" in action.lower() or "copper" in action.lower()

    def test_known_disease_ar(self):
        action = recommended_action("Olive___olive_peacock_spot", "ar")
        assert len(action) > 0

    def test_unknown_disease(self):
        action = recommended_action("Nonexistent___Disease", "en")
        assert action == ""


class TestTunisiaLabels:
    def test_olive_included(self):
        assert "Olive___olive_peacock_spot" in TUNISIA_LABELS

    def test_tomato_included(self):
        assert "Tomato___Bacterial_spot" in TUNISIA_LABELS

    def test_apple_excluded(self):
        assert "Apple___healthy" not in TUNISIA_LABELS
