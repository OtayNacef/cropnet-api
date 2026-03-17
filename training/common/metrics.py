"""
Evaluation metrics: top-1, top-3, macro F1, weighted F1, per-class P/R/F1, confusion matrix.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


class Meter:
    """Running average accumulator."""
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def per_class_metrics(true_labels: list[str], pred_labels: list[str]) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, F1."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for t, p in zip(true_labels, pred_labels):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    all_classes = sorted(set(true_labels) | set(pred_labels))
    result = {}
    for cls in all_classes:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        result[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp[cls] + fn[cls],
        }
    return result


def macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    """Macro F1: unweighted average of per-class F1."""
    vals = [v["f1"] for v in per_class.values()]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def weighted_f1(per_class: dict[str, dict[str, float]]) -> float:
    """Weighted F1: weighted by class support."""
    total_support = sum(v["support"] for v in per_class.values())
    if total_support == 0:
        return 0.0
    return round(sum(v["f1"] * v["support"] for v in per_class.values()) / total_support, 4)


def confusion_matrix(true_labels: list[str], pred_labels: list[str], classes: list[str]) -> list[list[int]]:
    """Build NxN confusion matrix. Rows = true, cols = predicted."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1
    return matrix


def top_k_accuracy(true_labels: list[str], top_k_preds: list[list[str]], k: int) -> float:
    """Top-k accuracy: fraction where true label appears in top-k predictions."""
    correct = 0
    for t, preds in zip(true_labels, top_k_preds):
        if t in preds[:k]:
            correct += 1
    return round(correct / len(true_labels) * 100, 2) if true_labels else 0.0


def full_eval_report(
    true_labels: list[str],
    pred_labels: list[str],
    top3_preds: list[list[str]],
    classes: list[str],
) -> dict:
    """Produce a complete evaluation report dict."""
    pc = per_class_metrics(true_labels, pred_labels)
    top1 = round(sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels) * 100, 2) if true_labels else 0.0
    top3 = top_k_accuracy(true_labels, top3_preds, 3)

    return {
        "total_samples": len(true_labels),
        "num_classes": len(classes),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "macro_f1": macro_f1(pc),
        "weighted_f1": weighted_f1(pc),
        "per_class": pc,
        "confusion_matrix": confusion_matrix(true_labels, pred_labels, classes),
        "classes": classes,
    }


def save_eval_report(report: dict, output_dir: Path, name: str = "eval"):
    """Save evaluation report as JSON artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Eval report → {path}")
    print(f"  Top-1: {report['top1_accuracy']}%  Top-3: {report['top3_accuracy']}%  Macro-F1: {report['macro_f1']}  Weighted-F1: {report['weighted_f1']}")
    return path
