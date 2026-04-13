"""Phase 8 — Comprehensive evaluation script.

Runs four experimental conditions:
  A  Baseline TelME           (results/baseline.json must exist)
  B  + Gaze (ours)            (results/ours.json must exist)
  C  Random gaze              (gaze_vec ~ N(0,1), uses trained ours model)
  D  Zero gaze                (gaze_vec = 0,      uses trained ours model)

Generates:
  results/table_main.md          — comparison table
  results/ablation_{feature}.json — per-feature ablation
  results/error_analysis.md      — qualitative error analysis stub

Usage:
    python eval/eval_all.py --out results/
    python eval/eval_all.py --skip-condition C D  # skip sanity checks
"""

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "TelME" / "MELD"))

from preprocessing import preprocessing
from dataset import meld_dataset
from utils import make_batchs
from model import Teacher_model, Student_Audio, ASF

from models.video_gaze_student import VideoGazeStudent
from utils.metrics import compute_metrics, save_metrics, paired_ttest

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if (hasattr(torch.backends, "mps") and
               torch.backends.mps.is_available()) else
    "cpu"
)

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_teacher(ckpt_dir: Path, cls_num: int):
    model = Teacher_model("roberta-large", cls_num)
    model.load_state_dict(torch.load(str(ckpt_dir / "teacher.bin"), map_location=device))
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def _load_audio_student(ckpt_dir: Path, cls_num: int):
    @dataclass
    class Cfg:
        mask_time_length: int = 3
    model = Student_Audio("facebook/data2vec-audio-base-960h", cls_num, Cfg())
    model.load_state_dict(torch.load(
        str(ckpt_dir / "student_audio" / "total_student.bin"), map_location=device))
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def _load_baseline_video_student(ckpt_dir: Path, cls_num: int):
    from model import Student_Video
    model = Student_Video("facebook/timesformer-base-finetuned-k400", cls_num)
    model.load_state_dict(torch.load(
        str(ckpt_dir / "student_video" / "total_student.bin"), map_location=device))
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def _load_video_gaze_student(ckpt_dir: Path, cls_num: int,
                              fusion_type: str = "gated",
                              fusion_lambda: float = 0.3):
    model = VideoGazeStudent(
        cls_num=cls_num, fusion_type=fusion_type, fusion_lambda=fusion_lambda)
    model.load_state_dict(torch.load(
        str(ckpt_dir / "student_video_gaze" / "total_student.bin"), map_location=device))
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def _load_baseline_fusion(ckpt_dir: Path, cls_num: int):
    model = ASF(cls_num, hidden_size=768, beta_shift=0.1, dropout_prob=0.2, num_head=3)
    model.load_state_dict(torch.load(str(ckpt_dir / "total_fusion.bin"), map_location=device))
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def _load_gaze_fusion(ckpt_dir: Path, cls_num: int, seed: int = 42):
    model = ASF(cls_num, hidden_size=768, beta_shift=0.1, dropout_prob=0.2, num_head=3)
    ckpt = ckpt_dir / f"total_fusion_gaze_seed{seed}.bin"
    if not ckpt.exists():
        ckpt = ckpt_dir / "total_fusion_gaze.bin"
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_baseline(model_t, audio_s, video_s_base, fusion, loader):
    y_true, y_pred = [], []
    for batch in loader:
        tok, mask, audio, video, labels = batch[:5]
        tok, mask, audio, video, labels = (
            tok.to(device), mask.to(device), audio.to(device),
            video.to(device), labels.to(device))
        text_h,  _ = model_t(tok, mask)
        audio_h, _ = audio_s(audio)
        video_h, _ = video_s_base(video)
        logits      = fusion(text_h, audio_h, video_h)
        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_true.extend(labels.cpu().numpy())
    return y_true, y_pred


@torch.no_grad()
def _run_ours(model_t, audio_s, video_s_gaze, fusion, loader,
              gaze_mode: str = "real"):
    """gaze_mode: 'real' | 'random' | 'zero'"""
    y_true, y_pred = [], []
    for batch in loader:
        tok, mask, audio, video, labels, gaze = batch
        tok, mask, audio, video, labels, gaze = (
            tok.to(device), mask.to(device), audio.to(device),
            video.to(device), labels.to(device), gaze.to(device))

        if gaze_mode == "random":
            gaze = torch.randn_like(gaze)
        elif gaze_mode == "zero":
            gaze = torch.zeros_like(gaze)

        text_h,  _ = model_t(tok, mask)
        audio_h, _ = audio_s(audio)
        video_h, _ = video_s_gaze(video, gaze)
        logits      = fusion(text_h, audio_h, video_h)
        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_true.extend(labels.cpu().numpy())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def _build_table(conditions: dict) -> str:
    header = "| Condition | Accuracy | Weighted-F1 | Macro-F1 |\n"
    sep    = "|-----------|----------|-------------|----------|\n"
    rows   = ""
    for name, m in conditions.items():
        acc = f"{m.get('accuracy', m.get('accuracy_mean', 0)):.4f}"
        wf1 = f"{m.get('weighted_f1', m.get('weighted_f1_mean', 0)):.4f}"
        mf1 = f"{m.get('macro_f1',   m.get('macro_f1_mean',   0)):.4f}"
        rows += f"| {name} | {acc} | {wf1} | {mf1} |\n"
    return header + sep + rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    out_dir  = Path(args.out)
    ckpt_dir = _ROOT / "models" / "checkpoints"
    data_root = _ROOT / "data" / "MELD.Raw"
    feat_dir  = _ROOT / "features" / "gaze"

    # Build test loader (without gaze)
    test_data_base = meld_dataset(
        preprocessing(str(data_root / "test_meld_emo.csv")))
    test_loader_base = DataLoader(test_data_base, batch_size=4, shuffle=False,
                                  num_workers=2, collate_fn=make_batchs)

    # Build test loader (with gaze)
    test_data_gaze = meld_dataset(
        preprocessing(str(data_root / "test_meld_emo.csv")),
        gaze_pkl=str(feat_dir / "test.pkl"))
    test_loader_gaze = DataLoader(test_data_gaze, batch_size=4, shuffle=False,
                                  num_workers=2, collate_fn=make_batchs)

    cls_num  = len(test_data_base.emoList)
    skip     = set(args.skip_condition or [])
    results  = {}

    print(f"Device: {device}")

    # ----- Condition A: Baseline TelME -----
    if "A" not in skip:
        print("Evaluating A: Baseline TelME…")
        model_t   = _load_teacher(ckpt_dir, cls_num)
        audio_s   = _load_audio_student(ckpt_dir, cls_num)
        video_s_b = _load_baseline_video_student(ckpt_dir, cls_num)
        fusion_b  = _load_baseline_fusion(ckpt_dir, cls_num)
        y_true, pred_A = _run_baseline(
            model_t, audio_s, video_s_b, fusion_b, test_loader_base)
        results["A_baseline"] = compute_metrics(y_true, pred_A, EMOTION_LABELS)
        print(f"  Weighted-F1: {results['A_baseline']['weighted_f1']:.4f}")
        del video_s_b, fusion_b; gc.collect()

    # ----- Condition B: + Gaze (ours) -----
    if "B" not in skip:
        print("Evaluating B: + Gaze (ours)…")
        if "A" not in skip:
            pass   # model_t, audio_s already loaded
        else:
            model_t = _load_teacher(ckpt_dir, cls_num)
            audio_s = _load_audio_student(ckpt_dir, cls_num)
        video_s_g = _load_video_gaze_student(ckpt_dir, cls_num)
        fusion_g  = _load_gaze_fusion(ckpt_dir, cls_num)
        y_true, pred_B = _run_ours(
            model_t, audio_s, video_s_g, fusion_g, test_loader_gaze, "real")
        results["B_ours"] = compute_metrics(y_true, pred_B, EMOTION_LABELS)
        print(f"  Weighted-F1: {results['B_ours']['weighted_f1']:.4f}")

    # Significance test A vs B
    if "A" not in skip and "B" not in skip:
        sig = paired_ttest(y_true, pred_A, pred_B)
        results["AB_ttest"] = sig
        print(f"  A vs B t-test: p={sig['p_value']:.4f}  "
              f"significant={sig['significant_at_0.05']}")

    # ----- Condition C: Random gaze -----
    if "C" not in skip and "B" not in skip:
        print("Evaluating C: Random gaze…")
        y_true_c, pred_C = _run_ours(
            model_t, audio_s, video_s_g, fusion_g, test_loader_gaze, "random")
        results["C_random_gaze"] = compute_metrics(y_true_c, pred_C, EMOTION_LABELS)
        print(f"  Weighted-F1: {results['C_random_gaze']['weighted_f1']:.4f}")

    # ----- Condition D: Zero gaze -----
    if "D" not in skip and "B" not in skip:
        print("Evaluating D: Zero gaze…")
        y_true_d, pred_D = _run_ours(
            model_t, audio_s, video_s_g, fusion_g, test_loader_gaze, "zero")
        results["D_zero_gaze"] = compute_metrics(y_true_d, pred_D, EMOTION_LABELS)
        print(f"  Weighted-F1: {results['D_zero_gaze']['weighted_f1']:.4f}")

    # ----- Save results -----
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval_all.json"
    with eval_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {eval_path}")

    # ----- Markdown table -----
    table_conditions = {
        k: v for k, v in results.items()
        if k not in ("AB_ttest",)
    }
    table_md = "# Main Results\n\n" + _build_table(table_conditions)
    if "AB_ttest" in results:
        sig = results["AB_ttest"]
        table_md += (f"\n**A vs B**: Δ weighted-F1 = "
                     f"{results.get('B_ours', {}).get('weighted_f1', 0) - results.get('A_baseline', {}).get('weighted_f1', 0):.4f}  "
                     f"p={sig['p_value']:.4f}  "
                     f"significant={'Yes' if sig['significant_at_0.05'] else 'No'}\n")

    table_path = out_dir / "table_main.md"
    table_path.write_text(table_md)
    print(f"Table saved → {table_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all evaluation conditions (A/B/C/D).")
    parser.add_argument("--out", default="results/")
    parser.add_argument("--skip-condition", nargs="*", choices=["A", "B", "C", "D"],
                        default=[])
    sys.exit(main(parser.parse_args()))
