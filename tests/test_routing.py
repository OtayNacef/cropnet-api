"""Unit tests for routing logic."""
from unittest.mock import MagicMock
from api.routing import Router, RoutingDecision
from api.inference.general import GeneralModel, InferenceResult, Prediction
from api.inference.calibration import CalibrationParams


def _mock_general(top_k: list[Prediction], agreement: int = 3) -> GeneralModel:
    m = MagicMock(spec=GeneralModel)
    m.predict.return_value = InferenceResult(top_k=top_k, crop_agreement=agreement, tta_crops=5, inference_ms=100)
    return m


def _mock_specialist(top_k: list[Prediction]) -> GeneralModel:
    m = MagicMock(spec=GeneralModel)
    m.predict.return_value = InferenceResult(top_k=top_k, crop_agreement=3, tta_crops=5, inference_ms=50)
    return m


class TestRoutingGeneralOnly:
    def test_no_specialists_returns_general(self):
        gen = _mock_general([Prediction(0, "Tomato___healthy", 0.85)])
        router = Router(gen, {})
        dec = router.route(MagicMock())
        assert dec.model_type == "general"
        assert dec.model_key == "general"
        assert dec.specialist is None

    def test_empty_predictions(self):
        gen = _mock_general([])
        router = Router(gen, {})
        dec = router.route(MagicMock())
        assert dec.model_type == "general"
        assert "No valid predictions" in dec.reason

    def test_unknown_crop_stays_general(self):
        gen = _mock_general([Prediction(0, "SomeRandom___Disease", 0.70)])
        router = Router(gen, {})
        dec = router.route(MagicMock())
        assert dec.model_type == "general"
        assert dec.crop_family is None


class TestRoutingSpecialistSelection:
    def test_specialist_high_confidence_wins(self):
        gen = _mock_general([Prediction(0, "Olive___olive_peacock_spot", 0.60)])
        spec = _mock_specialist([Prediction(0, "Olive___olive_peacock_spot", 0.85)])
        router = Router(gen, {"olive": spec})
        dec = router.route(MagicMock())
        assert dec.model_type == "specialist"
        assert dec.model_key == "olive"
        assert dec.specialist is not None

    def test_specialist_low_confidence_falls_back(self):
        gen = _mock_general([Prediction(0, "Olive___olive_peacock_spot", 0.75)])
        spec = _mock_specialist([Prediction(0, "Olive___olive_peacock_spot", 0.20)])
        router = Router(gen, {"olive": spec})
        dec = router.route(MagicMock())
        assert dec.model_type == "general"
        assert dec.crop_family == "olive"
        assert "low confidence" in dec.reason.lower()

    def test_specialist_competitive_still_used(self):
        """Specialist below threshold but within 80% of general → still preferred."""
        gen = _mock_general([Prediction(0, "Olive___Diseased", 0.50)])
        spec = _mock_specialist([Prediction(0, "Olive___Diseased", 0.42)])
        router = Router(gen, {"olive": spec})
        dec = router.route(MagicMock())
        # 0.42 >= 0.50 * 0.8 = 0.40, so specialist should be used
        assert dec.model_type == "specialist"

    def test_crop_hint_overrides(self):
        gen = _mock_general([Prediction(0, "Tomato___healthy", 0.90)])
        spec = _mock_specialist([Prediction(0, "Olive___Healthy", 0.70)])
        router = Router(gen, {"olive": spec})
        dec = router.route(MagicMock(), crop_hint="olive")
        assert dec.model_type == "specialist"
        assert dec.model_key == "olive"
        assert "crop_hint" in dec.reason


class TestRoutingTopK:
    def test_routes_from_second_prediction(self):
        """Olive appears as #2 in general top-k but should still trigger specialist."""
        gen = _mock_general([
            Prediction(0, "SomeUnknown___thing", 0.40),
            Prediction(1, "Olive___Diseased", 0.35),
        ])
        spec = _mock_specialist([Prediction(0, "Olive___olive_peacock_spot", 0.80)])
        router = Router(gen, {"olive": spec})
        dec = router.route(MagicMock())
        assert dec.model_type == "specialist"
        assert dec.crop_family == "olive"

    def test_routes_from_third_prediction(self):
        gen = _mock_general([
            Prediction(0, "Unknown1", 0.30),
            Prediction(1, "Unknown2", 0.25),
            Prediction(2, "Tomato___Early_blight", 0.20),
        ])
        spec = _mock_specialist([Prediction(0, "Tomato___Early_blight", 0.75)])
        router = Router(gen, {"tomato": spec})
        dec = router.route(MagicMock())
        assert dec.crop_family == "tomato"


class TestRoutingDisabledSpecialist:
    def test_crop_detected_but_no_specialist_loaded(self):
        """Specialist exists in config but not loaded (no ONNX) → general fallback."""
        gen = _mock_general([Prediction(0, "Pepper,_bell___Bacterial_spot", 0.80)])
        router = Router(gen, {})  # no specialists loaded
        dec = router.route(MagicMock())
        assert dec.model_type == "general"
        assert dec.crop_family == "pepper"
        assert "no specialist loaded" in dec.reason.lower()


class TestRoutingReason:
    def test_reason_always_present(self):
        gen = _mock_general([Prediction(0, "Tomato___healthy", 0.90)])
        router = Router(gen, {})
        dec = router.route(MagicMock())
        assert len(dec.reason) > 0

    def test_reason_includes_confidence(self):
        gen = _mock_general([Prediction(0, "Olive___Diseased", 0.60)])
        spec = _mock_specialist([Prediction(0, "Olive___Diseased", 0.80)])
        router = Router(gen, {"olive": spec})
        dec = router.route(MagicMock())
        assert "%" in dec.reason


class TestCalibrationPassthrough:
    def test_calibration_attached(self):
        gen = _mock_general([Prediction(0, "Olive___Diseased", 0.60)])
        spec = _mock_specialist([Prediction(0, "Olive___Diseased", 0.80)])
        gen_cal = CalibrationParams(conf_threshold=0.40)
        spec_cal = CalibrationParams(conf_threshold=0.45)
        router = Router(gen, {"olive": spec}, gen_cal, {"olive": spec_cal})
        dec = router.route(MagicMock())
        assert dec.general_calibration is gen_cal
        assert dec.specialist_calibration is spec_cal
