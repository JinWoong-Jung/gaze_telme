"""Phase 6 — Train VideoGazeStudent via knowledge distillation.

Extends TelME's student.py to use the gaze-injected VideoGazeStudent.
Adds gaze_vec from the batch (requires gaze-augmented dataset) and feeds it
to the model.  All other distillation losses (Logit_Loss + Feature_Loss) are
kept identical to TelME original.

Run from project root:
    python train/train_student_gaze.py --config configs/phase5_video_gaze.yaml
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project root & imports
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "TelME" / "MELD"))

from preprocessing import preprocessing
from dataset import meld_dataset
from utils import make_batchs
from model import Teacher_model
from meld_kd import Logit_Loss, Feature_Loss

from models.video_gaze_student import VideoGazeStudent
from utils.logger import Logger
from utils.seed import set_seed

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda"  if torch.cuda.is_available() else
    "mps"   if (hasattr(torch.backends, "mps") and
                torch.backends.mps.is_available()) else
    "cpu"
)
use_amp = (device.type == "cuda")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_video_gaze.yaml")
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--seed",          type=int,   default=None)
    parser.add_argument("--fusion-type",   type=str,   default=None,
                        choices=["add", "concat_proj", "gated"])
    parser.add_argument("--fusion-lambda", type=float, default=None)
    parser.add_argument("--no-wandb",      action="store_true")
    return parser.parse_args()


def _load_config(path: str) -> dict:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=False)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _ce_kd_loss(logit_s, logit_t, hidden_s, hidden_t, labels, alpha=1.0, beta=1.0):
    ce  = nn.CrossEntropyLoss()(logit_s, labels)
    kd  = Logit_Loss().to(logit_s.device)(logit_s, logit_t)
    feat = Feature_Loss().to(logit_s.device)(hidden_s, hidden_t)
    return ce + alpha * kd + beta * feat


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _save_model(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "total_student.bin")


def train(cfg: dict, logger: Logger) -> None:
    set_seed(cfg["training"]["seed"])

    data_root  = _ROOT / "data" / "MELD.Raw"
    feat_dir   = _ROOT / "features" / "gaze"
    ckpt_dir   = _ROOT / "models" / "checkpoints"

    train_pkl = str(feat_dir / "train.pkl")
    dev_pkl   = str(feat_dir / "dev.pkl")
    test_pkl  = str(feat_dir / "test.pkl")

    @dataclass
    class AudioConfig:
        mask_time_length: int = 3

    train_data = meld_dataset(
        preprocessing(str(data_root / "train_meld_emo.csv")), gaze_pkl=train_pkl)
    dev_data   = meld_dataset(
        preprocessing(str(data_root / "dev_meld_emo.csv")),   gaze_pkl=dev_pkl)
    test_data  = meld_dataset(
        preprocessing(str(data_root / "test_meld_emo.csv")),  gaze_pkl=test_pkl)

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
    model_t.load_state_dict(
        torch.load(str(ckpt_dir / "teacher.bin"), map_location=device))
    for p in model_t.parameters():
        p.requires_grad = False
    model_t = model_t.to(device).eval()

    # VideoGazeStudent
    g_cfg = cfg.get("gaze", {})
    model_s = VideoGazeStudent(
        video_model=cfg["models"]["video_model"],
        cls_num=cls_num,
        fusion_type=g_cfg.get("fusion_type", "gated"),
        fusion_lambda=float(g_cfg.get("fusion_lambda", 0.3)),
        gaze_in_dim=int(g_cfg.get("feat_dim", 6)),
        gaze_hidden=int(g_cfg.get("projector_hidden", 128)),
        gaze_dropout=float(g_cfg.get("projector_dropout", 0.1)),
    ).to(device)

    tr_cfg = cfg["training"]
    lr     = float(tr_cfg["learning_rate"])
    epochs = int(tr_cfg["epochs"])
    alpha  = float(tr_cfg.get("alpha", 1.0))
    beta   = float(tr_cfg.get("beta",  1.0))
    max_grad_norm = float(tr_cfg.get("max_grad_norm", 10.0))

    n_steps   = len(train_data) * epochs
    n_warmup  = int(n_steps * float(tr_cfg.get("warmup_ratio", 0.1)))
    optimizer = torch.optim.AdamW(model_s.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup, n_steps)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_path = ckpt_dir / "student_video_gaze"
    best_dev_f1, best_test_metrics = 0.0, {}

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model_s.train()
        for batch in train_loader:
            optimizer.zero_grad()
            (tok, mask, audio, video, labels, gaze) = batch
            tok, mask, video, labels, gaze = (
                tok.to(device), mask.to(device),
                video.to(device), labels.to(device), gaze.to(device))

            with torch.cuda.amp.autocast(enabled=use_amp):
                hidden_t, logit_t = model_t(tok, mask)
                hidden_s, logit_s = model_s(video, gaze)
                loss = _ce_kd_loss(logit_s, logit_t, hidden_s, hidden_t,
                                   labels, alpha=alpha, beta=beta)

            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()

        # Dev evaluation
        dev_f1 = _evaluate_f1(model_s, dev_loader, device, use_gaze=True)
        logger.log({"epoch": epoch, "dev_weighted_f1": dev_f1}, step=epoch)
        print(f"Epoch {epoch}  dev_f1={dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            _save_model(model_s, save_path)
            best_test_metrics = _full_eval(model_s, test_loader, device, use_gaze=True)
            logger.log({"best_test_weighted_f1": best_test_metrics["weighted_f1"],
                        "best_epoch": epoch}, step=epoch)
            print(f"  ↳ New best — test weighted-F1: "
                  f"{best_test_metrics['weighted_f1']:.4f}")

    # Save results
    results_path = _ROOT / "results" / "ours_student.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump({"video_gaze_student": best_test_metrics}, f, indent=2)
    print(f"Results saved to {results_path}")
    logger.finish()


def _evaluate_f1(model, loader, device, use_gaze: bool) -> float:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            if use_gaze:
                tok, mask, audio, video, labels, gaze = batch
                video, labels, gaze = video.to(device), labels.to(device), gaze.to(device)
                _, logit = model(video, gaze)
            else:
                tok, mask, audio, video, labels = batch
                video, labels = video.to(device), labels.to(device)
                _, logit = model(video, torch.zeros(video.size(0), 6, device=device))
            y_pred.extend(logit.argmax(1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return float(f1)


def _full_eval(model, loader, device, use_gaze: bool) -> dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            if use_gaze:
                tok, mask, audio, video, labels, gaze = batch
                video, labels, gaze = video.to(device), labels.to(device), gaze.to(device)
                _, logit = model(video, gaze)
            else:
                tok, mask, audio, video, labels = batch
                video, labels = video.to(device), labels.to(device)
                _, logit = model(video, torch.zeros(video.size(0), 6, device=device))
            y_pred.extend(logit.argmax(1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    acc = float(accuracy_score(y_true, y_pred))
    _, _, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    _, _, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {"accuracy": round(acc, 6),
            "weighted_f1": round(float(wf1), 6),
            "macro_f1": round(float(mf1), 6)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = parse_args()
    cfg  = _load_config(args.config)

    # CLI overrides
    if args.epochs:        cfg["training"]["epochs"]        = args.epochs
    if args.batch_size:    cfg["training"]["batch_size"]    = args.batch_size
    if args.lr:            cfg["training"]["learning_rate"] = args.lr
    if args.seed:          cfg["training"]["seed"]          = args.seed
    if args.fusion_type:   cfg["gaze"]["fusion_type"]       = args.fusion_type
    if args.fusion_lambda: cfg["gaze"]["fusion_lambda"]     = args.fusion_lambda

    log_cfg = cfg.get("logging", {})
    logger = Logger(
        project=cfg.get("project", {}).get("name", "gaze_telme"),
        run_name=log_cfg.get("run_name", "phase6_video_gaze"),
        log_dir=str(_ROOT / log_cfg.get("log_dir", "logs/phase5").lstrip("${")),
        use_wandb=(not args.no_wandb) and cfg.get("logging", {}).get("use_wandb", False),
        config=cfg,
    )

    train(cfg, logger)
