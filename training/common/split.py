#!/usr/bin/env python3
"""Split an ImageFolder dataset into train/val (default 90/10)."""
import argparse, shutil, random
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source ImageFolder directory")
    p.add_argument("--dst", required=True, help="Destination (will create train/ and val/)")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    random.seed(a.seed)
    src, dst = Path(a.src), Path(a.dst)
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir(): continue
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * a.val_ratio))
        val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]
        for split, files in [("train", train_imgs), ("val", val_imgs)]:
            out = dst / split / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out / f.name)
        print(f"  {cls_dir.name}: {len(train_imgs)} train / {len(val_imgs)} val")

if __name__ == "__main__": main()
