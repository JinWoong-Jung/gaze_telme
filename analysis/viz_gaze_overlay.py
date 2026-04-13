"""Phase 10 — Visualise gaze points overlaid on video frames.

For each utterance, draws:
  • red circle at predicted gaze point (normalised → pixel)
  • green rectangle for each detected face bbox
  • arrowed line from face centre to gaze point

Usage:
    python erc/viz_gaze_overlay.py --dialogue 0 --utterance 0 --split dev
    python erc/viz_gaze_overlay.py --dialogue 0 --utterance 0 --split dev --out results/figures/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def _load_gaze_npz(split: str, d: int, u: int) -> dict:
    path = _ROOT / "features" / "cache" / "gaze" / split / f"dia{d}_utt{u}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Gaze cache not found: {path}")
    return dict(np.load(str(path), allow_pickle=False))


def _load_frames(split: str, d: int, u: int) -> list:
    frame_dir = _ROOT / "data" / "processed" / "frames" / split / f"dia{d}_utt{u}"
    paths = sorted(frame_dir.glob("frame_*.jpg"))
    return [cv2.imread(str(p)) for p in paths], [p.name for p in paths]


def overlay_frame(frame: np.ndarray, gaze_pt: np.ndarray,
                  face_bboxes: np.ndarray) -> np.ndarray:
    """Draw gaze overlay on a single BGR frame."""
    H, W = frame.shape[:2]
    out = frame.copy()

    # Gaze point (normalised → pixel)
    gx = int(gaze_pt[0] * W)
    gy = int(gaze_pt[1] * H)
    cv2.circle(out, (gx, gy), 8, (0, 0, 255), -1)        # red dot
    cv2.circle(out, (gx, gy), 12, (0, 0, 200), 2)         # ring

    for bbox in face_bboxes:
        if bbox.sum() == 0:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)  # green
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.arrowedLine(out, (cx, cy), (gx, gy), (255, 100, 0), 2,
                        tipLength=0.2)  # blue arrow
    return out


def main(args: argparse.Namespace) -> int:
    d, u = args.dialogue, args.utterance
    split = args.split
    out_dir = Path(args.out) if args.out else (_ROOT / "results" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_gaze_npz(split, d, u)
    frames, fnames = _load_frames(split, d, u)

    pts    = data["frame_points"]     # (N, K, 2)
    bboxes = data["frame_face_bbox"]  # (N, K, 4)
    valid  = data["valid_mask"]       # (N,)

    N = min(len(frames), int(data["num_frames"]))
    written = 0

    for i in range(N):
        frame = frames[i]
        if frame is None:
            continue

        gaze_pt = pts[i, 0] if valid[i] else np.array([0.5, 0.5])
        face_bb = bboxes[i]

        overlay = overlay_frame(frame, gaze_pt, face_bb)
        fname   = f"dia{d}_utt{u}_frame{i:03d}_overlay.jpg"
        cv2.imwrite(str(out_dir / fname), overlay)
        written += 1

    print(f"Wrote {written} overlay frames → {out_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay Sharingan gaze on MELD frames.")
    parser.add_argument("--dialogue",   type=int, required=True)
    parser.add_argument("--utterance",  type=int, required=True)
    parser.add_argument("--split",      default="dev",
                        choices=["train", "dev", "test"])
    parser.add_argument("--out",        default=None,
                        help="Output directory (default: results/figures/)")
    sys.exit(main(parser.parse_args()))
