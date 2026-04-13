"""Phase 3 — Gaze feature engineering.

Aggregates frame-level Sharingan outputs into utterance-level gaze_vec ∈ R^6.

Features (indices fixed, documented in comments):
  0  p_face      — fraction of valid frames where gaze lands on a face
  1  p_out       — fraction of valid frames with inout < 0.5 (gaze outside)
  2  switch_rate — gaze-target switches per consecutive valid-frame pair
  3  entropy     — Shannon entropy of gaze-point histogram (4×4 grid)
  4  target_count— distinct gaze-target clusters (KMeans k=3 or fewer)
  5  p_center    — fraction of valid frames with gaze in centre [0.3,0.7]²

Normalisation:
  z-score on train set → scaler saved to features/gaze/scaler.pkl
  dev/test use train scaler (load-then-apply)

Usage:
  python pipeline/build_features.py --split train          # fit + transform
  python pipeline/build_features.py --split dev            # transform only
  python pipeline/build_features.py --split test           # transform only
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_GAZE_CACHE = _ROOT / "features" / "cache" / "gaze"
_FEAT_DIR   = _ROOT / "features" / "gaze"
_SCALER_PATH = _FEAT_DIR / "scaler.pkl"

FEAT_DIM = 6


# ---------------------------------------------------------------------------
# Per-utterance feature extraction
# ---------------------------------------------------------------------------

def _compute_gaze_vec(npz_path: Path) -> np.ndarray:
    """Return float32 array of shape (6,) or zeros if invalid."""
    data = np.load(str(npz_path), allow_pickle=False)

    valid_mask      = data["valid_mask"].astype(bool)   # (N,)
    frame_points    = data["frame_points"]               # (N, K, 2)
    frame_inout     = data["frame_inout"]                # (N, K)
    frame_hm_stats  = data["frame_heatmap_stats"]        # (N, K, 4)
    N               = int(data["num_frames"])

    n_valid = int(valid_mask.sum())
    if n_valid < 3:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    # For each frame, pick the primary face (first detected, track_id order)
    # We average over K faces per frame for simplicity.
    # gaze_pts: (N_valid, 2)  inout: (N_valid,)  entropy: (N_valid,)
    v_pts    = []
    v_inout  = []
    v_entrop = []

    for i in range(N):
        if not valid_mask[i]:
            continue
        pts  = frame_points[i]    # (K, 2)
        iout = frame_inout[i]     # (K,)
        hms  = frame_hm_stats[i]  # (K, 4)  mean_x, mean_y, std, entropy

        # Use the highest-confidence face (first in K dimension)
        mask_k = (pts.sum(axis=1) != 0)
        if not mask_k.any():
            continue
        k0 = np.argmax(mask_k)
        v_pts.append(pts[k0])
        v_inout.append(float(iout[k0]))
        v_entrop.append(float(hms[k0, 3]))   # Shannon entropy

    if len(v_pts) < 3:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    pts_arr  = np.array(v_pts,   dtype=np.float64)   # (M, 2)
    inout_arr = np.array(v_inout, dtype=np.float64)   # (M,)
    entrop_arr = np.array(v_entrop, dtype=np.float64) # (M,)
    M = len(pts_arr)

    # 0. p_face: gaze lands on a face  → inout ≥ 0.5
    p_face = float((inout_arr >= 0.5).sum()) / M

    # 1. p_out: gaze outside scene  → inout < 0.5
    p_out  = float((inout_arr < 0.5).sum()) / M

    # 2. switch_rate: consecutive gaze-target changes
    #    Proxy: L2 distance between consecutive gaze points > threshold (0.1)
    if M > 1:
        diffs = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)
        switches = int((diffs > 0.1).sum())
        switch_rate = switches / (M - 1)
    else:
        switch_rate = 0.0

    # 3. entropy: Shannon entropy of gaze-point 4×4 grid histogram
    xs = np.clip(pts_arr[:, 0], 0.0, 1.0 - 1e-9)
    ys = np.clip(pts_arr[:, 1], 0.0, 1.0 - 1e-9)
    grid_x = (xs * 4).astype(int)
    grid_y = (ys * 4).astype(int)
    cell   = grid_y * 4 + grid_x
    counts = np.bincount(cell, minlength=16).astype(float)
    prob   = counts / (counts.sum() + 1e-9)
    entropy = float(-(prob * np.log(prob + 1e-9)).sum())

    # 4. target_count: distinct gaze clusters (KMeans k ≤ 3)
    if M >= 3:
        try:
            from sklearn.cluster import KMeans
            k = min(3, M)
            km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pts_arr)
            # Count clusters with ≥ 2 members
            labels = km.labels_
            unique, cnts = np.unique(labels, return_counts=True)
            target_count = float(int((cnts >= 2).sum()))
        except Exception:
            target_count = 1.0
    else:
        target_count = 1.0

    # 5. p_center: gaze in center [0.3, 0.7]^2
    in_center = ((pts_arr[:, 0] >= 0.3) & (pts_arr[:, 0] <= 0.7) &
                 (pts_arr[:, 1] >= 0.3) & (pts_arr[:, 1] <= 0.7))
    p_center = float(in_center.sum()) / M

    vec = np.array([p_face, p_out, switch_rate, entropy, target_count, p_center],
                   dtype=np.float32)
    return vec


# ---------------------------------------------------------------------------
# Z-score scaler
# ---------------------------------------------------------------------------

class _ZScaler:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, X: np.ndarray) -> "\_ZScaler":
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    split = args.split
    gaze_cache = _GAZE_CACHE / split
    _FEAT_DIR.mkdir(parents=True, exist_ok=True)

    if not gaze_cache.exists():
        print(f"[ERROR] Gaze cache not found: {gaze_cache}\n"
              "Run pipeline/gaze_infer.py first.", file=sys.stderr)
        return 1

    npz_files = sorted(gaze_cache.glob("*.npz"))
    print(f"[{split}] Computing features for {len(npz_files)} utterances…")

    features: dict[tuple, np.ndarray] = {}
    is_valid_flags: dict[tuple, bool] = {}
    zero_count = 0

    for npz_path in npz_files:
        stem = npz_path.stem    # dia3_utt7
        m = re.match(r"dia(\d+)_utt(\d+)", stem)
        if not m:
            continue
        d, u = int(m.group(1)), int(m.group(2))

        vec = _compute_gaze_vec(npz_path)
        key = (d, u)
        features[key] = vec
        is_valid = bool(vec.sum() != 0)
        is_valid_flags[key] = is_valid
        if not is_valid:
            zero_count += 1

    zero_pct = 100 * zero_count / max(len(features), 1)
    print(f"[{split}] zero-vector rate: {zero_pct:.1f}%  "
          f"({zero_count}/{len(features)})")
    if zero_pct > 5:
        print("[WARNING] Zero-vector rate > 5% — check gaze cache quality.")

    # Stack into matrix for normalisation
    keys_ordered = list(features.keys())
    X = np.stack([features[k] for k in keys_ordered], axis=0)   # (M, 6)

    if split == "train":
        scaler = _ZScaler()
        X_norm = scaler.fit_transform(X)
        with _SCALER_PATH.open("wb") as f:
            pickle.dump(scaler, f)
        print(f"[train] Scaler saved to {_SCALER_PATH}")
    else:
        if not _SCALER_PATH.exists():
            print(f"[ERROR] Scaler not found at {_SCALER_PATH}.\n"
                  "Run --split train first.", file=sys.stderr)
            return 1
        with _SCALER_PATH.open("rb") as f:
            scaler = pickle.load(f)
        X_norm = scaler.transform(X)

    # Rebuild normalised dict
    feat_dict: dict[tuple, np.ndarray] = {}
    for key, row in zip(keys_ordered, X_norm):
        feat_dict[key] = row

    out_path = _FEAT_DIR / f"{split}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(feat_dict, f)
    print(f"[{split}] Saved {len(feat_dict)} utterance features → {out_path}")

    # Quick sanity
    sample_key = next(iter(feat_dict))
    assert feat_dict[sample_key].shape == (FEAT_DIM,), \
        f"Unexpected feature shape: {feat_dict[sample_key].shape}"

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build utterance-level gaze features (R^6) from Sharingan cache.")
    parser.add_argument("--split", choices=["train", "dev", "test"],
                        default="train")
    sys.exit(main(parser.parse_args()))
