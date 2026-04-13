"""Evaluation metrics for MELD emotion recognition.

Primary metric: Weighted-F1 (matches paper reporting convention).
Secondary: Accuracy, Macro-F1, class-wise F1.
"""

from typing import List, Optional, Sequence
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Optional[List[str]] = None,
) -> dict:
    """Compute full metric suite.

    Returns a dict with keys:
        accuracy        — micro accuracy
        weighted_f1     — primary metric
        macro_f1
        class_f1        — dict {emotion: f1}
        confusion_matrix — list[list[int]]
    """
    if label_names is None:
        label_names = EMOTION_LABELS

    y_true = list(y_true)
    y_pred = list(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted",
                                  zero_division=0))
    macro_f1    = float(f1_score(y_true, y_pred, average="macro",
                                  zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, average=None,
                             labels=list(range(len(label_names))),
                             zero_division=0)
    class_f1 = {label_names[i]: round(float(per_class_f1[i]), 6)
                for i in range(len(label_names))}

    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(label_names)))).tolist()

    return {
        "accuracy":        round(acc, 6),
        "weighted_f1":     round(weighted_f1, 6),
        "macro_f1":        round(macro_f1, 6),
        "class_f1":        class_f1,
        "confusion_matrix": cm,
    }


def save_metrics(metrics: dict, path: str | Path, key: str = "results") -> None:
    """Persist metrics dict to JSON (merges with existing file if present)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open() as f:
            existing = json.load(f)
    else:
        existing = {}
    existing[key] = metrics
    with path.open("w") as f:
        json.dump(existing, f, indent=2)


def aggregate_seeds(results: List[dict]) -> dict:
    """Average metrics over multiple seeds.

    Args:
        results: list of metric dicts (one per seed)

    Returns:
        dict with mean ± std for scalar metrics.
    """
    keys = ["accuracy", "weighted_f1", "macro_f1"]
    out = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        out[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        out[f"{k}_std"]  = round(float(np.std(vals)), 6)

    # Class-wise mean
    if results and "class_f1" in results[0]:
        class_keys = list(results[0]["class_f1"].keys())
        out["class_f1_mean"] = {}
        for ck in class_keys:
            vals = [r["class_f1"][ck] for r in results if ck in r.get("class_f1", {})]
            out["class_f1_mean"][ck] = round(float(np.mean(vals)), 6)

    return out


def paired_ttest(y_true: Sequence[int],
                 pred_a: Sequence[int],
                 pred_b: Sequence[int]) -> dict:
    """Approximate paired-sample t-test on per-sample correctness.

    Returns p-value and whether (B) is significantly better than (A) at p<0.05.
    """
    from scipy import stats
    a_correct = (np.array(pred_a) == np.array(y_true)).astype(float)
    b_correct = (np.array(pred_b) == np.array(y_true)).astype(float)
    t_stat, p_val = stats.ttest_rel(b_correct, a_correct)
    return {
        "t_statistic": round(float(t_stat), 6),
        "p_value":     round(float(p_val), 6),
        "significant_at_0.05": bool(p_val < 0.05),
    }
