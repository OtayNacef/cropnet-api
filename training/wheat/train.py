#!/usr/bin/env python3
"""Train CropNet wheat model."""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from training.common.train_loop import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    run(config_path=os.path.join(os.path.dirname(__file__), "config.yaml"), data_dir=a.data_dir, output_dir=a.output_dir)
