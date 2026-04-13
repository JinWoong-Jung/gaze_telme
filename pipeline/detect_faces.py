"""Phase 2.3 — Face detection and tracking per utterance.

For each utterance dir in data/processed/frames/{split}/dia{D}_utt{U}/
detects faces with RetinaFace (facenet-pytorch) and tracks them with
DeepSORT (deep-sort-realtime), then saves per-utterance results to
features/cache/faces/{split}/dia{D}_utt{U}.npz.

Output schema per .npz:
  dialogue_id   : int
  utterance_id  : int
  num_frames    : int
  bboxes        : float32 (N, K_max, 4)  xyxy, zero-padded for missing
  track_ids     : int32   (N, K_max)     -1 = no face / padding
  confs         : float32 (N, K_max)
  valid_mask    : bool    (N,)            True if ≥1 face detected

N = number of sampled frames, K_max = max faces per frame in this utterance.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from facenet_pytorch import MTCNN
    _DETECTOR = "mtcnn"
except ImportError:
    _DETECTOR = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _TRACKER = "deepsort"
except ImportError:
    _TRACKER = None

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _load_detector(device: str):
    if _DETECTOR == "mtcnn":
        from facenet_pytorch import MTCNN
        return MTCNN(keep_all=True, device=device, post_process=False,
                     min_face_size=20, thresholds=[0.6, 0.7, 0.9])
    raise RuntimeError("facenet-pytorch (MTCNN) is required for face detection.")


def _load_tracker():
    if _TRACKER == "deepsort":
        return DeepSort(max_age=10)
    raise RuntimeError("deep-sort-realtime is required for face tracking.")


def _detect_frame(detector, frame_rgb: np.ndarray):
    """Return boxes (K,4) in xyxy format and confs (K,) or empty arrays."""
    boxes, probs = detector.detect(frame_rgb)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    mask = (probs > 0.9)
    return boxes[mask].astype(np.float32), probs[mask].astype(np.float32)


def _track_sequence(tracker, frames_rgb, boxes_per_frame, confs_per_frame):
    """Run DeepSORT over all frames; return track_ids per frame."""
    track_ids_seq = []
    for frame, boxes, confs in zip(frames_rgb, boxes_per_frame, confs_per_frame):
        if len(boxes) == 0:
            tracks = tracker.update_tracks([], frame=frame)
        else:
            # DeepSort expects [[x1,y1,x2,y2, conf], ...]
            detections = [[*b, c] for b, c in zip(boxes.tolist(), confs.tolist())]
            tracks = tracker.update_tracks(detections, frame=frame)

        ids = []
        for t in tracks:
            if t.is_confirmed():
                ids.append(t.track_id)
        track_ids_seq.append(ids)
    return track_ids_seq


# ---------------------------------------------------------------------------
# Per-utterance processing
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _process_utterance(utt_dir: Path, out_path: Path,
                       detector, tracker, device: str) -> dict:
    stem = utt_dir.name    # dia3_utt7
    m = re.match(r"dia(\d+)_utt(\d+)", stem)
    if not m:
        return {"utt": stem, "status": "skip_name"}

    d, u = int(m.group(1)), int(m.group(2))

    frame_paths = sorted(utt_dir.glob("frame_*.jpg"))
    if not frame_paths:
        return {"utt": stem, "status": "no_frames"}

    frames_bgr = [cv2.imread(str(p)) for p in frame_paths]
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr
                  if f is not None]
    N = len(frames_rgb)

    if N == 0:
        return {"utt": stem, "status": "unreadable"}

    boxes_per_frame, confs_per_frame = [], []
    for frame in frames_rgb:
        b, c = _detect_frame(detector, frame)
        boxes_per_frame.append(b)
        confs_per_frame.append(c)

    tracker.delete_all_tracks()
    track_ids_seq = _track_sequence(tracker, frames_rgb,
                                    boxes_per_frame, confs_per_frame)

    K_max = max((len(b) for b in boxes_per_frame), default=0)
    K_max = max(K_max, 1)   # avoid zero-dim arrays

    bboxes    = np.zeros((N, K_max, 4), dtype=np.float32)
    track_ids = np.full((N, K_max), -1, dtype=np.int32)
    confs_arr = np.zeros((N, K_max), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for i, (boxes, confs, tids) in enumerate(
            zip(boxes_per_frame, confs_per_frame, track_ids_seq)):
        K = min(len(boxes), K_max)
        if K > 0:
            bboxes[i, :K]    = boxes[:K]
            confs_arr[i, :K] = confs[:K]
            valid_mask[i]    = True
        for j, tid in enumerate(tids[:K_max]):
            track_ids[i, j] = tid

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        dialogue_id=np.int32(d),
        utterance_id=np.int32(u),
        num_frames=np.int32(N),
        bboxes=bboxes,
        track_ids=track_ids,
        confs=confs_arr,
        valid_mask=valid_mask,
    )
    n_valid = int(valid_mask.sum())
    return {"utt": stem, "status": "ok", "N": N, "valid": n_valid}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if (hasattr(torch.backends, "mps") and
                        torch.backends.mps.is_available()) else "cpu"

    root  = _project_root()
    split = args.split
    frames_root = root / "data" / "processed" / "frames" / split
    cache_dir   = root / "features" / "cache" / "faces" / split

    if not frames_root.exists():
        print(f"[ERROR] Frames dir not found: {frames_root}\n"
              "Run preprocess/sample_frames.py first.", file=sys.stderr)
        return 1

    utt_dirs = sorted(frames_root.iterdir())
    utt_dirs = [d for d in utt_dirs if d.is_dir()]
    print(f"[{split}] {len(utt_dirs)} utterance dirs  device={device}")

    detector = _load_detector(device)
    tracker  = _load_tracker()

    ok = errors = skipped = 0
    for utt_dir in utt_dirs:
        out_path = cache_dir / f"{utt_dir.name}.npz"
        if out_path.exists() and not args.resume:
            skipped += 1
            continue

        res = _process_utterance(utt_dir, out_path, detector, tracker, device)
        if res["status"] == "ok":
            ok += 1
        elif res["status"] in ("skip_name", "no_frames", "unreadable"):
            skipped += 1
        else:
            errors += 1
            print(f"  [ERROR] {res}")

    print(f"[{split}] done — ok={ok}  skipped={skipped}  errors={errors}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and track faces in sampled frames.")
    parser.add_argument("--split", choices=["train", "dev", "test"],
                        default="train")
    parser.add_argument("--resume", action="store_true",
                        help="Skip utterances whose .npz already exists")
    sys.exit(main(parser.parse_args()))
