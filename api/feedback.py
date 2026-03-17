"""Feedback persistence — append-only JSONL log."""
from __future__ import annotations

import json
from pathlib import Path

from .config import FEEDBACK_DIR

_PATH = FEEDBACK_DIR / "feedback.jsonl"


def init() -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _PATH.exists():
        _PATH.touch()


def append(entry: dict) -> None:
    with open(_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def stats() -> dict:
    if not _PATH.exists():
        return {"total": 0, "confirmed": 0, "corrections": 0, "accuracy": None}
    total = confirmed = corrections = 0
    with open(_PATH) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("type") == "feedback":
                    total += 1
                    if e.get("user_confirmed"):
                        confirmed += 1
                    else:
                        corrections += 1
            except Exception:
                pass
    acc = round(confirmed / total * 100, 1) if total else None
    return {"total": total, "confirmed": confirmed, "corrections": corrections, "accuracy": f"{acc}%" if acc else None}
