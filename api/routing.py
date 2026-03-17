"""
Model router: general → infer crop from top-k → specialist (if available & passes threshold).

Routing rules:
1. Run general model first, get top-k predictions
2. Scan top-k (not just top-1) for crop family match
3. If a specialist is enabled and loaded for that crop, run it
4. Specialist overrides general ONLY if specialist confidence ≥ specialist threshold
5. Otherwise fall back to general result
6. routing_reason always explains the decision
7. Deterministic: same image → same routing path
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .config import SPECIALISTS, infer_crop_family
from .inference.calibration import CalibrationParams
from .inference.general import GeneralModel, InferenceResult


@dataclass
class RoutingDecision:
    general: InferenceResult
    specialist: InferenceResult | None
    model_key: str           # "general" or specialist key like "olive"
    model_type: str          # "general" | "specialist"
    crop_family: str | None
    reason: str
    general_calibration: CalibrationParams | None = None
    specialist_calibration: CalibrationParams | None = None


class Router:
    def __init__(
        self,
        general: GeneralModel,
        specialists: dict[str, GeneralModel],
        general_cal: CalibrationParams | None = None,
        specialist_cals: dict[str, CalibrationParams] | None = None,
    ):
        self.general = general
        self.specialists = specialists
        self.general_cal = general_cal
        self.specialist_cals = specialist_cals or {}

    def route(self, img: Image.Image, crop_hint: str = "", top_k_scan: int = 5) -> RoutingDecision:
        gen = self.general.predict(img, top_k=top_k_scan)

        if not gen.top_k:
            return RoutingDecision(
                gen, None, "general", "general", None,
                "No valid predictions from general model",
                self.general_cal, None,
            )

        # ── Step 1: Determine crop family ─────────────────────────────────

        family: str | None = None
        match_source: str = ""

        # 1a. crop_hint override (user explicitly told us the crop)
        if crop_hint:
            h = crop_hint.strip().lower().replace(" ", "_")
            for key in self.specialists:
                if h == key or h in key or key in h:
                    family = key
                    match_source = f"crop_hint='{crop_hint}'"
                    break

        # 1b. Scan top-k general predictions for crop family
        if not family:
            for pred in gen.top_k[:top_k_scan]:
                f = infer_crop_family(pred.label)
                if f and f in self.specialists:
                    family = f
                    match_source = f"top-k label '{pred.label}' (conf={pred.confidence:.2%})"
                    break
                elif f:
                    # crop detected but no specialist loaded — note it
                    family = f
                    match_source = f"top-k label '{pred.label}' (no specialist loaded)"
                    break

        # ── Step 2: Run specialist if available ───────────────────────────

        if family and family in self.specialists:
            spec_cfg = SPECIALISTS[family]
            spec_model = self.specialists[family]
            spec_cal = self.specialist_cals.get(family)
            spec = spec_model.predict(img)

            if spec.top_k:
                spec_top_conf = spec.top_k[0].confidence
                gen_top_conf = gen.top_k[0].confidence
                spec_threshold = spec_cfg["conf"]

                # Specialist wins if it passes its own threshold
                if spec_top_conf >= spec_threshold:
                    return RoutingDecision(
                        gen, spec, family, "specialist", family,
                        f"Specialist '{family}' confident ({spec_top_conf:.0%} ≥ threshold {spec_threshold:.0%}); "
                        f"matched via {match_source}",
                        self.general_cal, spec_cal,
                    )

                # Specialist below threshold but still better than general? Still use it
                # (crop-specific model knows more about that crop even at lower confidence)
                if spec_top_conf >= gen_top_conf * 0.8:
                    return RoutingDecision(
                        gen, spec, family, "specialist", family,
                        f"Specialist '{family}' below threshold ({spec_top_conf:.0%} < {spec_threshold:.0%}) "
                        f"but competitive with general ({gen_top_conf:.0%}); matched via {match_source}",
                        self.general_cal, spec_cal,
                    )

                # Fall back to general
                return RoutingDecision(
                    gen, spec, "general", "general", family,
                    f"Specialist '{family}' low confidence ({spec_top_conf:.0%} < {spec_threshold:.0%}), "
                    f"general preferred ({gen_top_conf:.0%}); matched via {match_source}",
                    self.general_cal, spec_cal,
                )

            return RoutingDecision(
                gen, spec, "general", "general", family,
                f"Specialist '{family}' returned no predictions, falling back to general",
                self.general_cal, None,
            )

        # ── Step 3: No specialist available ───────────────────────────────

        if family:
            reason = f"Crop '{family}' detected via {match_source} but no specialist loaded — general model only"
        else:
            reason = "Could not determine crop family from top-k predictions — general model only"

        return RoutingDecision(gen, None, "general", "general", family, reason, self.general_cal, None)
