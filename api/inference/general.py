"""General (broad) ONNX model wrapper."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ..config import ONNX_THREADS, TEMPERATURE, TTA_CROPS
from .metadata import is_junk
from .preprocess import five_crops, preprocess


@dataclass
class Prediction:
    class_id: int
    label: str
    confidence: float


@dataclass
class InferenceResult:
    top_k: list[Prediction] = field(default_factory=list)
    crop_agreement: int = 0
    tta_crops: int = TTA_CROPS
    inference_ms: int = 0


class GeneralModel:
    """ONNX-backed general crop-disease classifier with TTA."""

    def __init__(self, onnx_path: Path, id2label: dict[int, str], img_size: int = 256, name: str = "general"):
        self.name = name
        self.img_size = img_size
        self.id2label = id2label
        self.num_classes = len(id2label)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = ONNX_THREADS

        t0 = time.time()
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[CropNet:{name}] loaded {onnx_path.name} in {time.time()-t0:.1f}s | {self.num_classes} cls")

    def warmup(self, n: int = 3) -> None:
        dummy = Image.new("RGB", (self.img_size, self.img_size), (34, 139, 34))
        inp = preprocess(dummy, self.img_size)
        for _ in range(n):
            self.session.run([self.output_name], {self.input_name: inp})
        print(f"[CropNet:{self.name}] warmup done")

    def _forward(self, img: Image.Image) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: preprocess(img, self.img_size)})[0][0]

    def predict(self, img: Image.Image, top_k: int = 5) -> InferenceResult:
        t0 = time.time()
        crops = five_crops(img)
        logits_list = [self._forward(c) for c in crops]
        top1_ids = [int(np.argmax(l)) for l in logits_list]

        avg = np.mean(logits_list, axis=0) / TEMPERATURE
        exp = np.exp(avg - np.max(avg))
        probs = exp / exp.sum()

        # majority vote
        votes: dict[int, int] = {}
        for v in top1_ids:
            votes[v] = votes.get(v, 0) + 1
        maj_id = max(votes, key=votes.get)  # type: ignore
        maj_votes = votes[maj_id]

        ranked = np.argsort(probs)[::-1]
        preds: list[Prediction] = []
        for idx in ranked:
            lbl = self.id2label.get(int(idx), f"class_{idx}")
            if is_junk(lbl):
                continue
            preds.append(Prediction(int(idx), lbl, float(round(probs[idx], 4))))
            if len(preds) >= top_k:
                break

        # boost majority vote if it disagrees with averaged top-1
        if preds and maj_id != preds[0].class_id and maj_votes >= 3:
            lbl = self.id2label.get(maj_id, f"class_{maj_id}")
            if not is_junk(lbl):
                preds.insert(0, Prediction(maj_id, lbl, float(round(probs[maj_id], 4))))
                preds = preds[:top_k]

        return InferenceResult(preds, maj_votes, TTA_CROPS, int((time.time()-t0)*1000))
