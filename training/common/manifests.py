#!/usr/bin/env python3
"""Generate a manifest JSON listing all images, classes, and counts."""
import argparse, json
from pathlib import Path
from collections import Counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--output", default="manifest.json")
    a = p.parse_args()
    root = Path(a.dir)
    counts = Counter()
    files = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir(): continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}:
                counts[cls_dir.name] += 1
                files.append({"class": cls_dir.name, "path": str(f.relative_to(root))})
    manifest = {"total": len(files), "classes": len(counts), "per_class": dict(counts.most_common()), "files": files}
    with open(a.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {len(files)} images, {len(counts)} classes → {a.output}")

if __name__ == "__main__": main()
