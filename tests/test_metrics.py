"""Unit tests for evaluation metrics."""
from training.common.metrics import (
    per_class_metrics, macro_f1, weighted_f1,
    confusion_matrix, top_k_accuracy, full_eval_report,
)


class TestPerClassMetrics:
    def test_perfect(self):
        true = ["A", "A", "B", "B"]
        pred = ["A", "A", "B", "B"]
        pc = per_class_metrics(true, pred)
        assert pc["A"]["precision"] == 1.0
        assert pc["A"]["recall"] == 1.0
        assert pc["A"]["f1"] == 1.0

    def test_all_wrong(self):
        true = ["A", "A"]
        pred = ["B", "B"]
        pc = per_class_metrics(true, pred)
        assert pc["A"]["recall"] == 0.0
        assert pc["B"]["precision"] == 0.0

    def test_support(self):
        true = ["A", "A", "A", "B"]
        pred = ["A", "A", "B", "B"]
        pc = per_class_metrics(true, pred)
        assert pc["A"]["support"] == 3
        assert pc["B"]["support"] == 1


class TestMacroF1:
    def test_perfect(self):
        pc = per_class_metrics(["A", "B"], ["A", "B"])
        assert macro_f1(pc) == 1.0

    def test_half(self):
        pc = per_class_metrics(["A", "A", "B", "B"], ["A", "B", "A", "B"])
        f = macro_f1(pc)
        assert 0 < f < 1


class TestWeightedF1:
    def test_perfect(self):
        pc = per_class_metrics(["A", "B"], ["A", "B"])
        assert weighted_f1(pc) == 1.0


class TestConfusionMatrix:
    def test_2x2(self):
        true = ["A", "A", "B", "B"]
        pred = ["A", "B", "B", "B"]
        cm = confusion_matrix(true, pred, ["A", "B"])
        assert cm[0][0] == 1  # A→A
        assert cm[0][1] == 1  # A→B
        assert cm[1][1] == 2  # B→B


class TestTopKAccuracy:
    def test_top1(self):
        true = ["A", "B", "C"]
        preds = [["A", "B"], ["C", "B"], ["C", "A"]]
        # A matches, B→C no, C matches top1 = 2/3
        assert top_k_accuracy(true, preds, 1) == 66.67

    def test_top3(self):
        true = ["A", "B"]
        preds = [["X", "Y", "A"], ["B", "X", "Y"]]
        assert top_k_accuracy(true, preds, 3) == 100.0


class TestFullReport:
    def test_complete_report(self):
        true = ["A", "A", "B", "B", "C"]
        pred = ["A", "A", "B", "A", "C"]
        top3 = [["A", "B", "C"]] * 5
        report = full_eval_report(true, pred, top3, ["A", "B", "C"])
        assert report["total_samples"] == 5
        assert report["num_classes"] == 3
        assert "top1_accuracy" in report
        assert "top3_accuracy" in report
        assert "macro_f1" in report
        assert "weighted_f1" in report
        assert "per_class" in report
        assert "confusion_matrix" in report
        assert len(report["confusion_matrix"]) == 3
