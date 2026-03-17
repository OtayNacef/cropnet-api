"""Shared test fixtures for CropNet API tests."""
import os
import sys
import json
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test-friendly env vars BEFORE importing api modules
os.environ["MODELS_DIR"] = "/tmp/cropnet-test-models"
os.environ["FEEDBACK_DIR"] = tempfile.mkdtemp(prefix="cropnet-feedback-")
os.environ["CROPNET_API_KEY"] = "test-key-12345"
os.environ["ENABLE_SPECIALIST_OLIVE"] = "false"
os.environ["ENABLE_SPECIALIST_DATE_PALM"] = "false"
os.environ["ENABLE_SPECIALIST_WHEAT"] = "false"
os.environ["ENABLE_SPECIALIST_CITRUS"] = "false"
os.environ["ENABLE_SPECIALIST_TOMATO"] = "false"
os.environ["ENABLE_SPECIALIST_PEPPER"] = "false"
os.environ["ENABLE_SPECIALIST_WATERMELON"] = "false"


@pytest.fixture
def test_labels_dir():
    """Create a temporary directory with a labels.json file."""
    d = Path("/tmp/cropnet-test-models/general")
    d.mkdir(parents=True, exist_ok=True)
    labels = {
        "id2label": {
            "0": "Tomato___healthy",
            "1": "Tomato___Early_blight",
            "2": "Olive___olive_peacock_spot",
            "3": "Olive___Healthy",
            "4": "Orange___Haunglongbing_(Citrus_greening)",
        },
        "num_classes": 5,
    }
    with open(d / "labels.json", "w") as f:
        json.dump(labels, f)
    return d


@pytest.fixture
def feedback_dir():
    """Return a clean temp feedback directory."""
    d = Path(os.environ["FEEDBACK_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    # Clean any previous test data
    for f in d.iterdir():
        f.unlink()
    return d
