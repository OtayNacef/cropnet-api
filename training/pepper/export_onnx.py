#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX."""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pathlib import Path
from training.common.utils import export_onnx, build_dinov2_classifier
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default="facebook/dinov2-base")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dinov2_classifier(a.base_model, a.num_classes, device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=True))
    export_onnx(model, 256, Path(a.output), device)

if __name__ == "__main__": main()
