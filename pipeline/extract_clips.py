"""Phase 2.1 — Utterance clip extraction.

Reads *_sent_emo.csv, maps each row to the pre-split .mp4 clip that already
lives in data/MELD.Raw/{split}_splits_complete/, and symlinks (or copies) it
into data/processed/clips/{split}/dia{D}_utt{U}.mp4.

MELD already ships utterance-level clips, so 'extraction' here is mostly a
rename+copy step.  If you have raw episode files instead, uncomment the
ffmpeg slicing block.
"""

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPLIT_DIR_MAP = {
    "train": "train_splits",
    "dev":   "dev_splits_complete",
    "test":  "test_splits_complete",
}
_CSV_MAP = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_utterances(csv_path: Path):
    """Yield (dialogue_id, utterance_id, src_filename) for every CSV row."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = int(row["Dialogue_ID"])
            u = int(row["Utterance_ID"])
            # MELD csv embeds path like ./dataset/MELD.Raw/train_splits/dia0_utt0.mp4
            video_path = row.get("Video_Path", "")
            fname = Path(video_path).name  # e.g. dia0_utt0.mp4
            yield d, u, fname


def main(args: argparse.Namespace) -> int:
    root = _project_root()
    split = args.split

    raw_dir = root / "data" / "MELD.Raw" / _SPLIT_DIR_MAP[split]
    csv_path = root / "data" / "MELD.Raw" / _CSV_MAP[split]
    out_dir  = root / "data" / "processed" / "clips" / split
    missing_csv = root / "data" / "processed" / "missing_clips.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    missing = []
    ok = 0

    for d, u, fname in iter_utterances(csv_path):
        src = raw_dir / fname
        canonical = f"dia{d}_utt{u}.mp4"
        dst = out_dir / canonical

        if dst.exists() and not args.overwrite:
            ok += 1
            continue

        if not src.exists():
            # Try the canonical name directly (some splits already use it)
            alt = raw_dir / canonical
            if alt.exists():
                src = alt
            else:
                missing.append({"split": split, "dialogue_id": d,
                                 "utterance_id": u, "expected": str(src)})
                continue

        if args.symlink:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

        ok += 1

    print(f"[{split}] processed {ok} clips, {len(missing)} missing")

    if missing:
        missing_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not missing_csv.exists()
        with missing_csv.open("a", newline="") as f:
            fieldnames = ["split", "dialogue_id", "utterance_id", "expected"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerows(missing)
        print(f"Missing clips logged to {missing_csv}")

    return 0 if not missing else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organise MELD utterance clips into data/processed/clips/")
    parser.add_argument("--split", choices=["train", "dev", "test"],
                        default="train")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink instead of copy (saves disk space)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process already-present destination files")
    sys.exit(main(parser.parse_args()))
