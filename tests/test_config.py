"""Unit tests for config module."""
from api.config import (
    infer_crop_family,
    SPECIALISTS,
    GENERAL_THRESHOLD,
    SPECIALIST_THRESHOLD,
    LOW_CONF_THRESHOLD,
    ADVISORY_ONLY_THRESHOLD,
    specialist_onnx_path,
    specialist_labels_path,
    general_onnx_path,
    general_labels_path,
)


class TestCropFamilyInference:
    def test_olive(self):
        assert infer_crop_family("Olive___olive_peacock_spot") == "olive"

    def test_olive_healthy(self):
        assert infer_crop_family("Olive___Healthy") == "olive"

    def test_date_palm(self):
        assert infer_crop_family("Date Palm data") == "date_palm"

    def test_palm(self):
        assert infer_crop_family("Palm___Brown_Leaf_Spot") == "date_palm"

    def test_citrus_via_orange(self):
        assert infer_crop_family("Orange___Haunglongbing_(Citrus_greening)") == "citrus"

    def test_tomato(self):
        assert infer_crop_family("Tomato___Early_blight") == "tomato"

    def test_pepper(self):
        assert infer_crop_family("Pepper,_bell___Bacterial_spot") == "pepper"

    def test_wheat(self):
        assert infer_crop_family("Wheat___Septoria") == "wheat"

    def test_unknown(self):
        assert infer_crop_family("SomethingRandom___Disease") is None

    def test_empty_string(self):
        assert infer_crop_family("") is None


class TestSpecialistRegistry:
    def test_tier1_keys(self):
        tier1 = {k for k, v in SPECIALISTS.items() if v["tier"] == 1}
        assert tier1 == {"olive", "date_palm", "wheat", "citrus"}

    def test_tier2_keys(self):
        tier2 = {k for k, v in SPECIALISTS.items() if v["tier"] == 2}
        assert tier2 == {"tomato", "pepper", "watermelon"}

    def test_all_have_conf(self):
        for key, cfg in SPECIALISTS.items():
            assert "conf" in cfg, f"{key} missing conf threshold"
            assert cfg["conf"] > 0

    def test_all_have_version(self):
        for key, cfg in SPECIALISTS.items():
            assert "version" in cfg

    def test_all_have_img_size(self):
        for key, cfg in SPECIALISTS.items():
            assert cfg["img_size"] > 0


class TestThresholds:
    def test_general_threshold_positive(self):
        assert GENERAL_THRESHOLD > 0

    def test_specialist_above_general(self):
        assert SPECIALIST_THRESHOLD >= GENERAL_THRESHOLD

    def test_low_conf_below_general(self):
        assert LOW_CONF_THRESHOLD < GENERAL_THRESHOLD

    def test_advisory_below_low(self):
        assert ADVISORY_ONLY_THRESHOLD < LOW_CONF_THRESHOLD

    def test_ordering(self):
        assert ADVISORY_ONLY_THRESHOLD < LOW_CONF_THRESHOLD < GENERAL_THRESHOLD


class TestModelPaths:
    def test_general_onnx_name(self):
        p = general_onnx_path()
        assert "cropnet-general-" in p.name

    def test_specialist_onnx_name(self):
        p = specialist_onnx_path("olive")
        assert "cropnet-olive-" in p.name

    def test_specialist_labels(self):
        p = specialist_labels_path("wheat")
        assert p.name == "labels.json"
        assert "wheat" in str(p)

    def test_general_labels(self):
        p = general_labels_path()
        assert p.name == "labels.json"
