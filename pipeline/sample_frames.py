"""Phase 2.2 — Frame sampling from utterance clips.

For each clip in data/processed/clips/{split}/ samples frames at a fixed FPS
(default 6) up to a maximum count (default 32) using uniform spacing, and
writes JPEGs to data/processed/frames/{split}/dia{D}_utt{U}/frame_{i:03d}.jpg.

Uses decord for fast video decoding when available, falls back to OpenCV.
Supports multiprocessing via --num-workers.
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np

try:
    import decord
    _USE_DECORD = True
except ImportError:
    _USE_DECORD = False

try:
    import cv2
    _USE_CV2 = True
except ImportError:
    _USE_CV2 = False

if not _USE_DECORD and not _USE_CV2:
    raise ImportError("Either 'decord' or 'opencv-python' must be installed.")


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------

def _sample_with_decord(clip_path: Path, target_fps: float, max_frames: int):
    """Return list of BGR ndarrays sampled at target_fps, capped at max_frames."""
    import decord
    decord.bridge.set_bridge("numpy")
    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0))
    total = len(vr)
    src_fps = vr.get_avg_fps()
    if src_fps <= 0:
        src_fps = 25.0

    step = max(1, round(src_fps / target_fps))
    indices = list(range(0, total, step))

    if len(indices) > max_frames:
        idx_arr = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
        indices = [indices[i] for i in idx_arr]

    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, RGB)
    return [frame[:, :, ::-1] for frame in frames]   # RGB → BGR for cv2.imwrite


def _sample_with_cv2(clip_path: Path, target_fps: float, max_frames: int):
    """OpenCV fallback."""
    cap = cv2.VideoCapture(str(clip_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, round(src_fps / target_fps))
    indices = set(range(0, total, step))

    if len(indices) > max_frames:
        idx_arr = np.linspace(0, total - 1, max_frames, dtype=int)
        indices = set(idx_arr.tolist())

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def sample_frames(clip_path: Path, target_fps: float, max_frames: int):
    if _USE_DECORD:
        return _sample_with_decord(clip_path, target_fps, max_frames)
    return _sample_with_cv2(clip_path, target_fps, max_frames)


# ---------------------------------------------------------------------------
# Per-clip worker
# ---------------------------------------------------------------------------

def _process_clip(clip_path: Path, out_root: Path, fps: float,
                  max_frames: int, overwrite: bool) -> dict:
    stem = clip_path.stem   # e.g. dia3_utt7
    out_dir = out_root / stem

    if out_dir.exists() and not overwrite:
        # Skip if already fully processed (rough check: has at least 1 frame)
        if any(out_dir.iterdir()):
            return {"clip": stem, "status": "skipped", "n_frames": -1}

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames = sample_frames(clip_path, fps, max_frames)
    except Exception as exc:
        return {"clip": stem, "status": "error", "error": str(exc), "n_frames": 0}

    for i, frame in enumerate(frames):
        fname = out_dir / f"frame_{i:03d}.jpg"
        import cv2 as _cv2
        _cv2.imwrite(str(fname), frame,
                     [_cv2.IMWRITE_JPEG_QUALITY, 95])

    return {"clip": stem, "status": "ok", "n_frames": len(frames)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(args: argparse.Namespace) -> int:
    root = _project_root()
    split = args.split
    clips_dir = root / "data" / "processed" / "clips" / split
    frames_root = root / "data" / "processed" / "frames" / split

    if not clips_dir.exists():
        print(f"[ERROR] Clips dir not found: {clips_dir}\n"
              "Run preprocess/extract_clips.py first.", file=sys.stderr)
        return 1

    clip_paths = sorted(clips_dir.glob("*.mp4"))
    if not clip_paths:
        print(f"[WARNING] No .mp4 clips found in {clips_dir}", file=sys.stderr)
        return 0

    print(f"[{split}] Found {len(clip_paths)} clips. fps={args.fps} "
          f"max_frames={args.max_frames} workers={args.num_workers}")

    worker = partial(_process_clip,
                     out_root=frames_root,
                     fps=float(args.fps),
                     max_frames=args.max_frames,
                     overwrite=args.overwrite)

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = pool.map(worker, clip_paths)
    else:
        results = [worker(p) for p in clip_paths]

    ok = sum(r["status"] == "ok" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    errors = [r for r in results if r["status"] == "error"]

    print(f"[{split}] done: {ok} processed, {skipped} skipped, "
          f"{len(errors)} errors")

    if errors:
        for e in errors:
            print(f"  [ERROR] {e['clip']}: {e.get('error', '')}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample frames from MELD clips at fixed FPS.")
    parser.add_argument("--split", choices=["train", "dev", "test"],
                        default="train")
    parser.add_argument("--fps", type=float, default=6.0,
                        help="Target sampling FPS (default: 6)")
    parser.add_argument("--max-frames", type=int, default=32,
                        help="Max frames per clip (default: 32)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    sys.exit(main(parser.parse_args()))
