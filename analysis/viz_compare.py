"""Phase 10 — Baseline vs Ours comparison bar chart.

Reads results/eval_all.json and produces a grouped bar chart of
Accuracy / Weighted-F1 / Macro-F1 per condition (A, B, C, D).
Also outputs a class-wise F1 delta bar chart (B - A).

Usage:
    python erc/viz_compare.py --out results/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

CONDITION_NAMES = {
    "A_baseline":    "(A) Baseline TelME",
    "B_ours":        "(B) + Gaze (ours)",
    "C_random_gaze": "(C) Random gaze",
    "D_zero_gaze":   "(D) Zero gaze",
}


def main(args: argparse.Namespace) -> int:
    eval_path = _ROOT / "results" / "eval_all.json"
    if not eval_path.exists():
        print(f"[ERROR] {eval_path} not found.\n"
              "Run scripts/eval_all.py first.", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with eval_path.open() as f:
        results = json.load(f)

    present = [(k, CONDITION_NAMES.get(k, k)) for k in CONDITION_NAMES if k in results]
    if not present:
        print("[ERROR] No recognised condition keys found in eval_all.json.",
              file=sys.stderr)
        return 1

    # ----- Grouped bar chart: Acc / W-F1 / M-F1 -----
    metrics  = ["accuracy", "weighted_f1", "macro_f1"]
    x        = np.arange(len(metrics))
    width    = 0.8 / len(present)
    offsets  = np.linspace(-(len(present) - 1) * width / 2,
                            (len(present) - 1) * width / 2, len(present))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label) in enumerate(present):
        vals = [results[key].get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Weighted-F1", "Macro-F1"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline vs Ours — Overall Metrics")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(str(out_dir / "compare_overall.png"), dpi=120)
    plt.close(fig)
    print("Saved: compare_overall.png")

    # ----- Class-wise F1 delta: B - A -----
    if "A_baseline" in results and "B_ours" in results:
        cf1_a = results["A_baseline"].get("class_f1", {})
        cf1_b = results["B_ours"].get("class_f1", {})
        deltas = [cf1_b.get(e, 0) - cf1_a.get(e, 0) for e in EMOTION_LABELS]
        colors = ["green" if d >= 0 else "red" for d in deltas]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(EMOTION_LABELS, deltas, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Class-wise F1 Delta: (B) Ours − (A) Baseline")
        ax.set_ylabel("ΔF1")
        ax.set_xlabel("Emotion")
        plt.tight_layout()
        fig.savefig(str(out_dir / "compare_classwise_delta.png"), dpi=120)
        plt.close(fig)
        print("Saved: compare_classwise_delta.png")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comparison figures from eval_all.json.")
    parser.add_argument("--out", default="results/figures/")
    sys.exit(main(parser.parse_args()))
