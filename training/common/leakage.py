#!/usr/bin/env python3
"""
Data leakage risk checks.

Checks for:
- Train/val file name overlap (same filename in both splits)
- Train/val exact byte duplicates (MD5)
- Class name anomalies (numeric-only, suspiciously short)

Limitations:
- Does NOT detect near-duplicate images across splits (use dedup.py for perceptual hashing)
- Does NOT detect augmentation leakage (same base image augmented differently in train+val)
- File-level checks only; content-based leak detection would need a feature embedding DB
"""
import argparse, hashlib
from pathlib import Path
from collections import Counter


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description="Check for train/val data leakage risks")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    a = p.parse_args()

    train_root, val_root = Path(a.train_dir), Path(a.val_dir)
    issues = []

    # 1. Filename overlap
    train_names = {f.name for f in train_root.rglob("*") if f.is_file()}
    val_names = {f.name for f in val_root.rglob("*") if f.is_file()}
    overlap = train_names & val_names
    if overlap:
        issues.append(f"⚠️  {len(overlap)} filenames appear in both train and val (e.g. {list(overlap)[:5]})")
    else:
        print("✅ No filename overlap between train/val")

    # 2. MD5 duplicates across splits
    print("Computing MD5 hashes...")
    train_hashes = {}
    for f in train_root.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            train_hashes[md5(f)] = f
    dup_count = 0
    for f in val_root.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            h = md5(f)
            if h in train_hashes:
                dup_count += 1
                if dup_count <= 5:
                    issues.append(f"  LEAK: {f} == {train_hashes[h]}")
    if dup_count:
        issues.append(f"⚠️  {dup_count} byte-identical files across train/val")
    else:
        print("✅ No byte-identical duplicates across splits")

    # 3. Class name anomalies
    all_classes = set()
    for d in [train_root, val_root]:
        for sub in d.iterdir():
            if sub.is_dir():
                all_classes.add(sub.name)
    suspicious = [c for c in all_classes if c.isdigit() or len(c) < 3 or c.lower() in {"test", "train", "valid", "dataset"}]
    if suspicious:
        issues.append(f"⚠️  Suspicious class names: {suspicious}")
    else:
        print("✅ No suspicious class names")

    if issues:
        print(f"\n{'='*60}")
        print(f"LEAKAGE RISKS FOUND ({len(issues)}):")
        for i in issues:
            print(f"  {i}")
    else:
        print("\n✅ All leakage checks passed")


if __name__ == "__main__":
    main()
