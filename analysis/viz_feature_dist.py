"""Phase 10 — Gaze feature distribution per emotion class.

Generates box-plots of each gaze feature (R^6) grouped by emotion label.

Usage:
    python erc/viz_feature_dist.py --split train --out results/figures/
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
FEAT_NAMES = ['p_face', 'p_out', 'switch_rate', 'entropy', 'target_count', 'p_center']


def _load_gaze_features(split: str) -> dict:
    path = _ROOT / "features" / "gaze" / f"{split}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}\n"
                                "Run features/gaze/build_features.py first.")
    with path.open("rb") as f:
        return pickle.load(f)


def _load_labels(split: str) -> dict:
    """Return {(dialogue_id, utterance_id): emotion_label_str}."""
    import csv
    csv_name = f"{split}_sent_emo.csv"
    csv_path = _ROOT / "data" / "MELD.Raw" / csv_name
    if not csv_path.exists():
        csv_path = _ROOT / "data" / "MELD.Raw" / f"{split}_meld_emo.csv"
    labels = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = int(row["Dialogue_ID"])
            u = int(row["Utterance_ID"])
            labels[(d, u)] = row["Emotion"].lower()
    return labels


def main(args: argparse.Namespace) -> int:
    split   = args.split
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = _load_gaze_features(split)
    try:
        labels = _load_labels(split)
    except FileNotFoundError as e:
        print(f"[WARNING] {e}\nSkipping label-split plots.")
        labels = {}

    # Build DataFrame
    rows = []
    for (d, u), vec in features.items():
        emotion = labels.get((d, u), "unknown")
        row = {"emotion": emotion, **{FEAT_NAMES[i]: float(vec[i])
                                      for i in range(len(FEAT_NAMES))}}
        rows.append(row)
    df = pd.DataFrame(rows)

    # Filter to known emotions
    df = df[df["emotion"].isin(EMOTION_LABELS)]

    # One figure per feature
    for feat in FEAT_NAMES:
        fig, ax = plt.subplots(figsize=(9, 4))
        data_by_emo = [df.loc[df["emotion"] == e, feat].values
                       for e in EMOTION_LABELS]
        ax.boxplot(data_by_emo, labels=EMOTION_LABELS, patch_artist=True,
                   notch=False, showfliers=False)
        ax.set_title(f"[{split}] {feat} by emotion")
        ax.set_ylabel(feat)
        ax.set_xlabel("Emotion")
        plt.tight_layout()
        fig.savefig(str(out_dir / f"{split}_{feat}_by_emotion.png"), dpi=120)
        plt.close(fig)
        print(f"Saved: {feat}_by_emotion.png")

    # Per-emotion mean bar chart for all features
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(FEAT_NAMES):
        ax = axes[i]
        means = [df.loc[df["emotion"] == e, feat].mean() for e in EMOTION_LABELS]
        ax.bar(EMOTION_LABELS, means, color="steelblue", alpha=0.8)
        ax.set_title(feat)
        ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
    plt.suptitle(f"Mean gaze features per emotion ({split})", y=1.02)
    plt.tight_layout()
    fig.savefig(str(out_dir / f"{split}_gaze_features_overview.png"), dpi=120)
    plt.close(fig)
    print(f"Saved: {split}_gaze_features_overview.png")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot gaze feature distributions per emotion.")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--out",   default="results/figures/")
    sys.exit(main(parser.parse_args()))
