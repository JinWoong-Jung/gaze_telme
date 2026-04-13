"""Phase 2.4 — Frame-level gaze inference via Sharingan.

Loads a Sharingan checkpoint from sharingan_ckpt/, iterates over
features/cache/faces/{split}/*.npz, and for each utterance runs batched
gaze inference, writing results to features/cache/gaze/{split}/dia{D}_utt{U}.npz.

Expected Sharingan API (from upstream repo):
    model = build_model(cfg)
    gaze_map, gaze_point, inout = model(frame_tensor, head_bbox_tensor)

Output schema per .npz:
  dialogue_id         : int
  utterance_id        : int
  num_frames          : int
  frame_points        : float32 (N, K, 2)   normalised (x, y) in [0,1]
  frame_inout         : float32 (N, K)      P(gaze inside scene)
  frame_heatmap_stats : float32 (N, K, 4)   mean_x, mean_y, std, entropy
  frame_face_bbox     : float32 (N, K, 4)   xyxy from detection stage
  valid_mask          : bool    (N,)

N = num_frames, K = max faces per frame in utterance.
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root & path setup
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_SHARINGAN_ROOT = _ROOT / "sharingan"
_CKPT_ROOT = _ROOT / "sharingan_ckpt"

if str(_SHARINGAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARINGAN_ROOT))


# ---------------------------------------------------------------------------
# Sharingan model loader (adapt to actual repo API)
# ---------------------------------------------------------------------------

def _load_sharingan_model(ckpt_dir: Path, device: str):
    """Load Sharingan model.  Adjust import path to match actual repo layout."""
    try:
        # Attempt to import from the cloned Sharingan repo
        from model import ModelSharingan  # type: ignore
        model = ModelSharingan()
        ckpt_files = sorted(ckpt_dir.glob("*.pth")) + sorted(ckpt_dir.glob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        state = torch.load(str(ckpt_files[0]), map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model.to(device)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import Sharingan model from {_SHARINGAN_ROOT}.\n"
            f"Clone the repo first: git clone <sharingan-repo> sharingan\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Heatmap stats (mean gaze point, std, entropy)
# ---------------------------------------------------------------------------

def _heatmap_stats(heatmap: np.ndarray) -> np.ndarray:
    """heatmap: (H, W) float32 — return (4,): mean_x, mean_y, std, entropy."""
    H, W = heatmap.shape
    h = heatmap.astype(np.float64)
    h = h / (h.sum() + 1e-9)

    ys, xs = np.mgrid[0:H, 0:W]
    mean_x = float((h * xs).sum()) / W
    mean_y = float((h * ys).sum()) / H
    var    = float((h * ((xs / W - mean_x) ** 2 + (ys / H - mean_y) ** 2)).sum())
    std    = float(var ** 0.5)
    entropy = float(-(h * np.log(h + 1e-9)).sum())
    return np.array([mean_x, mean_y, std, entropy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-utterance inference
# ---------------------------------------------------------------------------

def _load_frames(utt_dir: Path, frame_indices: list) -> list:
    """Return list of HxWx3 uint8 BGR frames."""
    import cv2
    frames = []
    for i in frame_indices:
        p = utt_dir / f"frame_{i:03d}.jpg"
        if p.exists():
            frames.append(cv2.imread(str(p)))
        else:
            frames.append(None)
    return frames


@torch.no_grad()
def _infer_utterance(model, face_npz: Path, frames_root: Path,
                     device: str, batch_size: int) -> dict:
    data = np.load(str(face_npz), allow_pickle=False)
    d   = int(data["dialogue_id"])
    u   = int(data["utterance_id"])
    N   = int(data["num_frames"])
    bboxes    = data["bboxes"]     # (N, K, 4)
    valid_mask = data["valid_mask"] # (N,)

    K = bboxes.shape[1]
    stem = f"dia{d}_utt{u}"
    utt_dir = frames_root / stem

    frame_points        = np.zeros((N, K, 2),  dtype=np.float32)
    frame_inout         = np.zeros((N, K),      dtype=np.float32)
    frame_heatmap_stats = np.zeros((N, K, 4),   dtype=np.float32)

    # Process frames in batches
    for batch_start in range(0, N, batch_size):
        batch_idx = range(batch_start, min(batch_start + batch_size, N))
        for i in batch_idx:
            if not valid_mask[i]:
                continue
            frame_path = utt_dir / f"frame_{i:03d}.jpg"
            if not frame_path.exists():
                continue

            import cv2
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue
            H_img, W_img = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            for k in range(K):
                bbox = bboxes[i, k]  # xyxy
                if bbox.sum() == 0:
                    continue
                x1, y1, x2, y2 = bbox
                # Clamp to image bounds
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W_img, int(x2)), min(H_img, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                # Prepare tensors
                frame_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                frame_t = frame_t.unsqueeze(0).to(device)  # (1, 3, H, W)
                bbox_t  = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32).to(device)

                try:
                    gaze_map, gaze_point, inout = model(frame_t, bbox_t)
                except Exception:
                    continue

                # gaze_point: (1, 2) — normalised (x, y)
                pt = gaze_point.squeeze(0).cpu().numpy()
                frame_points[i, k] = pt.astype(np.float32)

                # inout: (1,) or scalar
                io = float(inout.squeeze().cpu().numpy())
                frame_inout[i, k] = io

                # heatmap stats
                if gaze_map is not None:
                    gm = gaze_map.squeeze().cpu().numpy()
                    if gm.ndim == 2:
                        frame_heatmap_stats[i, k] = _heatmap_stats(gm)

        if torch.cuda.is_available() and (batch_start // batch_size + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return {
        "dialogue_id": np.int32(d),
        "utterance_id": np.int32(u),
        "num_frames": np.int32(N),
        "frame_points": frame_points,
        "frame_inout":  frame_inout,
        "frame_heatmap_stats": frame_heatmap_stats,
        "frame_face_bbox": bboxes.astype(np.float32),
        "valid_mask": valid_mask,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(args: argparse.Namespace) -> int:
    root  = _ROOT
    split = args.split
    faces_dir  = root / "features" / "cache" / "faces"  / split
    gaze_dir   = root / "features" / "cache" / "gaze"   / split
    frames_root = root / "data" / "processed" / "frames" / split

    gaze_dir.mkdir(parents=True, exist_ok=True)

    if not faces_dir.exists():
        print(f"[ERROR] Face cache not found: {faces_dir}\n"
              "Run preprocess/detect_faces.py first.", file=sys.stderr)
        return 1

    device = _device()
    print(f"[{split}] Loading Sharingan model from {_CKPT_ROOT}  device={device}")
    model = _load_sharingan_model(_CKPT_ROOT, device)

    face_npzs = sorted(faces_dir.glob("*.npz"))
    print(f"[{split}] {len(face_npzs)} utterances to process")

    missing = []
    ok = skipped = errors = 0
    flush_interval = 500

    for idx, face_npz in enumerate(face_npzs):
        stem = face_npz.stem
        out_path = gaze_dir / f"{stem}.npz"

        if out_path.exists() and args.resume:
            skipped += 1
            continue

        try:
            result = _infer_utterance(model, face_npz, frames_root, device,
                                      args.batch_size)
            np.savez_compressed(str(out_path), **result)
            ok += 1
        except Exception as exc:
            errors += 1
            missing.append(stem)
            print(f"  [ERROR] {stem}: {exc}")

        if torch.cuda.is_available() and (idx + 1) % flush_interval == 0:
            torch.cuda.empty_cache()

    print(f"[{split}] done — ok={ok}  skipped={skipped}  errors={errors}")

    missing_log = root / "features" / "cache" / "gaze" / "missing.txt"
    if missing:
        with missing_log.open("a") as f:
            for s in missing:
                f.write(f"{split}/{s}\n")
        print(f"Missing utterances logged to {missing_log}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Sharingan gaze inference on MELD frames.")
    parser.add_argument("--split", choices=["train", "dev", "test"],
                        default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true",
                        help="Skip utterances whose .npz already exists")
    sys.exit(main(parser.parse_args()))
