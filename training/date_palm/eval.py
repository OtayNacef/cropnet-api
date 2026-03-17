#!/usr/bin/env python3
"""Evaluate CropNet ONNX model. Produces: top-1, top-3, macro F1, weighted F1, confusion matrix, per-class P/R/F1."""
import argparse, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from training.common.metrics import full_eval_report, save_eval_report

_MEAN = np.array([.485,.456,.406], dtype=np.float32)
_STD  = np.array([.229,.224,.225], dtype=np.float32)

def preprocess(img, sz=256):
    return ((np.array(img.resize((sz,sz), Image.LANCZOS), dtype=np.float32)/255 - _MEAN) / _STD).transpose(2,0,1)[np.newaxis]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to ONNX model")
    p.add_argument("--labels", required=True, help="Path to labels.json")
    p.add_argument("--data-dir", required=True, help="ImageFolder val directory")
    p.add_argument("--output-dir", default=None, help="Where to save eval artifacts")
    p.add_argument("--img-size", type=int, default=256)
    a = p.parse_args()

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    inp_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    with open(a.labels) as f:
        id2l = {int(k): v for k, v in json.load(f)["id2label"].items()}

    true_labels, pred_labels, top3_preds = [], [], []
    classes_seen = set()

    for cls_dir in sorted(Path(a.data_dir).iterdir()):
        if not cls_dir.is_dir(): continue
        classes_seen.add(cls_dir.name)
        for img_path in tqdm(sorted(cls_dir.glob("*")), desc=cls_dir.name, leave=False):
            if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".webp"}: continue
            try:
                img = Image.open(img_path).convert("RGB")
                logits = sess.run([out_name], {inp_name: preprocess(img, a.img_size)})[0][0]
                ranked = np.argsort(logits)[::-1]
                top1 = id2l.get(int(ranked[0]), f"class_{ranked[0]}")
                top3 = [id2l.get(int(ranked[i]), f"class_{ranked[i]}") for i in range(min(3, len(ranked)))]
                true_labels.append(cls_dir.name)
                pred_labels.append(top1)
                top3_preds.append(top3)
            except Exception as e:
                print(f"  SKIP {img_path.name}: {e}")

    classes = sorted(classes_seen | set(pred_labels))
    report = full_eval_report(true_labels, pred_labels, top3_preds, classes)

    print(f"\n{'='*60}")
    print(f"Samples: {report['total_samples']}  Classes: {report['num_classes']}")
    print(f"Top-1: {report['top1_accuracy']}%  Top-3: {report['top3_accuracy']}%")
    print(f"Macro F1: {report['macro_f1']}  Weighted F1: {report['weighted_f1']}")

    if a.output_dir:
        save_eval_report(report, Path(a.output_dir))
        # Save confusion matrix as separate artifact for visualization
        cm_path = Path(a.output_dir) / "confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump({"classes": classes, "matrix": report["confusion_matrix"]}, f)
        print(f"  Confusion matrix → {cm_path}")

if __name__ == "__main__": main()
