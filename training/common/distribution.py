#!/usr/bin/env python3
"""Class distribution report. Prints per-class counts, imbalance ratio, and min/max."""
import argparse, json
from pathlib import Path
from collections import Counter


def analyze(root: Path) -> dict:
    counts = Counter()
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        counts[cls_dir.name] = n

    if not counts:
        return {"error": "No classes found"}

    total = sum(counts.values())
    sorted_counts = counts.most_common()
    largest = sorted_counts[0]
    smallest = sorted_counts[-1]
    imbalance_ratio = largest[1] / smallest[1] if smallest[1] > 0 else float("inf")

    return {
        "total_images": total,
        "num_classes": len(counts),
        "per_class": dict(sorted_counts),
        "largest_class": {"name": largest[0], "count": largest[1]},
        "smallest_class": {"name": smallest[0], "count": smallest[1]},
        "imbalance_ratio": round(imbalance_ratio, 2),
        "mean_per_class": round(total / len(counts), 1),
        "median_per_class": sorted(counts.values())[len(counts) // 2],
    }


def main():
    p = argparse.ArgumentParser(description="Class distribution report")
    p.add_argument("--dir", required=True, help="ImageFolder root directory")
    p.add_argument("--output", default=None, help="Save JSON report")
    a = p.parse_args()

    report = analyze(Path(a.dir))
    if "error" in report:
        print(report["error"])
        return

    print(f"\nDataset: {a.dir}")
    print(f"Total: {report['total_images']} images / {report['num_classes']} classes")
    print(f"Imbalance ratio: {report['imbalance_ratio']}x (largest/smallest)")
    print(f"Largest:  {report['largest_class']['name']} ({report['largest_class']['count']})")
    print(f"Smallest: {report['smallest_class']['name']} ({report['smallest_class']['count']})")
    print(f"Mean: {report['mean_per_class']} | Median: {report['median_per_class']}")
    print(f"\nPer-class:")
    for cls, count in report["per_class"].items():
        bar = "█" * min(50, int(count / report["largest_class"]["count"] * 50))
        print(f"  {cls:40s} {count:5d} {bar}")

    if a.output:
        with open(a.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved → {a.output}")


if __name__ == "__main__":
    main()
