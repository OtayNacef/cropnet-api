"""Unit tests for feedback persistence."""
import json
import os
from pathlib import Path

from api import feedback
from api.config import FEEDBACK_DIR


class TestFeedback:
    def setup_method(self):
        """Clean feedback file before each test."""
        p = FEEDBACK_DIR / "feedback.jsonl"
        if p.exists():
            p.unlink()
        feedback.init()

    def test_init_creates_file(self):
        assert (FEEDBACK_DIR / "feedback.jsonl").exists()

    def test_append_and_read(self):
        feedback.append({"type": "scan", "scan_id": "test-1", "predicted": "Olive___Healthy"})
        feedback.append({"type": "feedback", "scan_id": "test-1", "user_confirmed": True, "correct_label": "Olive___Healthy"})
        p = FEEDBACK_DIR / "feedback.jsonl"
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["scan_id"] == "test-1"

    def test_stats_empty(self):
        s = feedback.stats()
        assert s["total"] == 0
        assert s["accuracy"] is None

    def test_stats_confirmed(self):
        feedback.append({"type": "feedback", "scan_id": "a", "user_confirmed": True})
        feedback.append({"type": "feedback", "scan_id": "b", "user_confirmed": True})
        feedback.append({"type": "feedback", "scan_id": "c", "user_confirmed": False})
        s = feedback.stats()
        assert s["total"] == 3
        assert s["confirmed"] == 2
        assert s["corrections"] == 1
        assert "66.7%" in s["accuracy"]

    def test_stats_ignores_non_feedback(self):
        feedback.append({"type": "scan", "scan_id": "x"})
        s = feedback.stats()
        assert s["total"] == 0  # scan entries don't count
