"""Phase 7 — ASF Fusion training with gaze-injected video student.

Loads the trained VideoGazeStudent ckpt and the frozen teacher/audio-student,
then trains the ASF fusion head.  Structure is identical to TelME fusion.py
except video_s is replaced with VideoGazeStudent and batch unpacking includes
gaze_vec.

Run from project root:
    python train/train_fusion.py --config configs/phase7_fusion.yaml
    python train/train_fusion.py --config configs/phase7_fusion.yaml --seed 7
    python train/train_fusion.py --config configs/phase7_fusion.yaml --seed 2024
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "TelME" / "MELD"))

from preprocessing import preprocessing
from dataset import meld_dataset
from utils import make_batchs
from model import Teacher_model, Student_Audio, ASF
from meld_kd import Logit_Loss, Feature_Loss

from models.video_gaze_student import VideoGazeStudent
from utils.logger import Logger
from utils.seed import set_seed

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if (hasattr(torch.backends, "mps") and
               torch.backends.mps.is_available()) else
    "cpu"
)
use_amp = (device.type == "cuda")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/phase7_fusion.yaml")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Override seed (for multi-seed runs)")
    parser.add_argument("--no-wandb",   action="store_true")
    return parser.parse_args()


def _load_config(path: str) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_model(model, path: Path, name: str = "total_fusion_gaze.bin") -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / name)


def _evaluate(model_t, audio_s, video_s, fusion, loader, device) -> tuple:
    fusion.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            tok, mask, audio, video, labels, gaze = batch
            tok, mask, audio, video, labels, gaze = (
                tok.to(device), mask.to(device), audio.to(device),
                video.to(device), labels.to(device), gaze.to(device))

            text_h,  _ = model_t(tok, mask)
            audio_h, _ = audio_s(audio)
            video_h, _ = video_s(video, gaze)
            logits      = fusion(text_h, audio_h, video_h)

            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_true, y_pred


def _metrics(y_true, y_pred) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    _, _, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    _, _, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {"accuracy": round(acc, 6),
            "weighted_f1": round(float(wf1), 6),
            "macro_f1": round(float(mf1), 6)}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_one_seed(cfg: dict, seed: int, logger: Logger) -> dict:
    set_seed(seed)

    data_root = _ROOT / "data" / "MELD.Raw"
    feat_dir  = _ROOT / "features" / "gaze"
    ckpt_dir  = _ROOT / "models" / "checkpoints"

    @dataclass
    class AudioCfg:
        mask_time_length: int = 3

    train_data = meld_dataset(
        preprocessing(str(data_root / "train_meld_emo.csv")),
        gaze_pkl=str(feat_dir / "train.pkl"))
    dev_data = meld_dataset(
        preprocessing(str(data_root / "dev_meld_emo.csv")),
        gaze_pkl=str(feat_dir / "dev.pkl"))
    test_data = meld_dataset(
        preprocessing(str(data_root / "test_meld_emo.csv")),
        gaze_pkl=str(feat_dir / "test.pkl"))

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,
                              num_workers=2, collate_fn=make_batchs)
    dev_loader   = DataLoader(dev_data,   batch_size=bs, shuffle=False,
                              num_workers=2, collate_fn=make_batchs)
    test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False,
                              num_workers=2, collate_fn=make_batchs)

    cls_num = len(train_data.emoList)

    # Teacher (frozen)
    model_t = Teacher_model(cfg["models"]["text_model"], cls_num)
    model_t.load_state_dict(torch.load(
        str(ckpt_dir / "teacher.bin"), map_location=device))
    for p in model_t.parameters():
        p.requires_grad = False
    model_t = model_t.to(device).eval()

    # Audio student (frozen)
    audio_s = Student_Audio(cfg["models"]["audio_model"], cls_num, AudioCfg())
    audio_s.load_state_dict(torch.load(
        str(ckpt_dir / "student_audio" / "total_student.bin"), map_location=device))
    for p in audio_s.parameters():
        p.requires_grad = False
    audio_s = audio_s.to(device).eval()

    # VideoGazeStudent (frozen — loaded from Phase 6 ckpt)
    g_cfg = cfg.get("gaze", {})
    video_s = VideoGazeStudent(
        video_model=cfg["models"]["video_model"],
        cls_num=cls_num,
        fusion_type=g_cfg.get("fusion_type", "gated"),
        fusion_lambda=float(g_cfg.get("fusion_lambda", 0.3)),
    )
    video_s.load_state_dict(torch.load(
        str(ckpt_dir / "student_video_gaze" / "total_student.bin"),
        map_location=device))
    for p in video_s.parameters():
        p.requires_grad = False
    video_s = video_s.to(device).eval()

    # ASF fusion (trainable)
    tr_cfg = cfg["training"]
    asf_cfg = tr_cfg  # same dict
    fusion = ASF(
        clsNum=cls_num,
        hidden_size=int(asf_cfg.get("asf_hidden", 768)),
        beta_shift=float(asf_cfg.get("asf_beta_shift", 0.1)),
        dropout_prob=float(asf_cfg.get("asf_dropout", 0.2)),
        num_head=int(asf_cfg.get("asf_num_head", 3)),
    ).to(device)

    lr = float(tr_cfg["learning_rate"])
    epochs = int(tr_cfg["epochs"])
    max_grad_norm = float(tr_cfg.get("max_grad_norm", 10.0))
    n_steps  = len(train_data) * epochs
    n_warmup = int(n_steps * float(tr_cfg.get("warmup_ratio", 0.1)))

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup, n_steps)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_loss   = nn.CrossEntropyLoss()

    save_path = ckpt_dir
    best_dev_f1, best_test_m = 0.0, {}

    for epoch in tqdm(range(epochs), desc=f"[seed={seed}]"):
        fusion.train()
        for batch in train_loader:
            optimizer.zero_grad()
            tok, mask, audio, video, labels, gaze = batch
            tok, mask, audio, video, labels, gaze = (
                tok.to(device), mask.to(device), audio.to(device),
                video.to(device), labels.to(device), gaze.to(device))

            with torch.cuda.amp.autocast(enabled=use_amp):
                text_h,  _ = model_t(tok, mask)
                audio_h, _ = audio_s(audio)
                video_h, _ = video_s(video, gaze)
                logits      = fusion(text_h, audio_h, video_h)
                loss        = ce_loss(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()

        y_true, y_pred = _evaluate(model_t, audio_s, video_s, fusion,
                                   dev_loader, device)
        _, _, dev_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted")
        dev_f1 = float(dev_f1)
        logger.log({"epoch": epoch, "dev_weighted_f1": dev_f1,
                    "seed": seed}, step=epoch)
        print(f"  epoch={epoch}  dev_f1={dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            _save_model(fusion, save_path, f"total_fusion_gaze_seed{seed}.bin")
            y_true_t, y_pred_t = _evaluate(model_t, audio_s, video_s, fusion,
                                           test_loader, device)
            best_test_m = _metrics(y_true_t, y_pred_t)
            best_test_m["seed"] = seed
            best_test_m["best_epoch"] = epoch
            print(f"    ↳ test weighted-F1: {best_test_m['weighted_f1']:.4f}")

    return best_test_m


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = parse_args()
    cfg  = _load_config(args.config)

    if args.epochs:     cfg["training"]["epochs"]        = args.epochs
    if args.batch_size: cfg["training"]["batch_size"]    = args.batch_size
    if args.lr:         cfg["training"]["learning_rate"] = args.lr

    seeds = [int(args.seed)] if args.seed else [int(s) for s in cfg["training"].get("seeds", [42])]

    log_cfg = cfg.get("logging", {})
    all_results = []

    for seed in seeds:
        run_name = f"phase7_fusion_gaze_seed{seed}"
        logger = Logger(
            project=cfg.get("project", {}).get("name", "gaze_telme"),
            run_name=run_name,
            log_dir=str(_ROOT / "logs" / "phase7"),
            use_wandb=(not args.no_wandb) and log_cfg.get("use_wandb", False),
            config={**cfg, "seed": seed},
        )
        result = train_one_seed(cfg, seed, logger)
        all_results.append(result)
        logger.finish()

    # Aggregate
    from utils.metrics import aggregate_seeds
    summary = aggregate_seeds(all_results)
    summary["per_seed"] = all_results

    results_path = _ROOT / "results" / "ours.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump({"fusion_gaze": summary}, f, indent=2)
    print(f"\nFinal results ({len(seeds)} seeds): {summary}")
    print(f"Saved to {results_path}")
