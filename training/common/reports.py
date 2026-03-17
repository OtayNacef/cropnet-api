#!/usr/bin/env python3
"""Generate a markdown training report from report.json + eval.json."""
import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Output dir with report.json and optionally eval.json")
    a = p.parse_args()
    d = Path(a.dir)
    report = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else {}
    ev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}

    lines = [f"# CropNet Training Report: {report.get('crop', '?')}", ""]
    lines.append(f"- **Date**: {report.get('date', '?')}")
    lines.append(f"- **Best val accuracy**: {report.get('best_val_acc', '?'):.2f}%")
    lines.append(f"- **Epochs**: {report.get('epochs', '?')}")
    lines.append(f"- **Classes**: {len(report.get('classes', []))}")
    if ev:
        lines.append(f"- **Eval accuracy**: {ev.get('accuracy', '?')}%")
        lines.append(f"- **Eval images**: {ev.get('total', '?')}")
    lines.append(f"\n## Classes\n")
    for c in report.get("classes", []):
        lines.append(f"- {c}")
    print("\n".join(lines))

if __name__ == "__main__": main()
