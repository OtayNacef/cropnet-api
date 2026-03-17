#!/usr/bin/env python3
"""Deduplicate images by perceptual hash (removes near-duplicates)."""
import argparse, hashlib
from pathlib import Path
from PIL import Image

def phash(img: Image.Image, size: int = 8) -> str:
    img = img.convert("L").resize((size, size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p > avg else "0" for p in pixels)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    seen: dict[str, Path] = {}
    removed = 0
    for img_path in sorted(Path(a.dir).rglob("*")):
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".webp"}: continue
        try:
            h = phash(Image.open(img_path))
            if h in seen:
                print(f"  DUP: {img_path} == {seen[h]}")
                if not a.dry_run:
                    img_path.unlink()
                removed += 1
            else:
                seen[h] = img_path
        except Exception:
            pass
    print(f"{'Would remove' if a.dry_run else 'Removed'} {removed} duplicates")

if __name__ == "__main__": main()
