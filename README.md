<p align="center">
  <h1 align="center">Gaze-TelME</h1>
  <p align="center">
    <a>정진웅<sub>1</sub></a>
    ·
    <a>권희진<sub>2</sub></a>
    ·
    <a>김수현<sub>1</sub></a>
    ·
    <a>문수라<sub>3</sub></a>
    ·
    <a>박희성<sub>1</sub></a>
    ·
    <a>슈멍<sub>1</sub></a>
  </p>
  <p align="center">
    <i>Sungkyunkwan University · AAI<sub>1</sub>, DSC<sub>2</sub> and CNT<sub>3</sub> Departments</i><br>
    <i>2026-1 SKK:AI - Social AI session</i>
  </p>
</p>

## Overview

This project explores **gaze-aware multimodal emotion recognition in conversation (MERC)** on the MELD dataset.

We combine two existing models:

- **[TelME](https://github.com/yuntaeyang/TelME)** — a MERC baseline that fuses text (RoBERTa-large), audio (Data2Vec), and video (TimeSformer) via knowledge distillation and attentive score fusion (ASF).
- **[Sharingan](https://github.com/idiap/sharingan)** — a video gaze-following model used here to derive pseudo-gaze signals from MELD dialogue video clips.

The goal is to extract utterance-level gaze features from MELD clips, project them into the video student's hidden space, and test whether gaze provides useful interaction cues for emotion recognition.

이 프로젝트는 MELD 데이터셋에서 대화 감정 인식 성능을 높이기 위해, MERC 모델인 `TelME`와 gaze estimation 모델인 `Sharingan`을 결합하는 것을 목표로 한다. `Sharingan`으로부터 MELD 영상의 pseudo-gaze 신호를 추출하고, 이를 `TelME`의 텍스트-오디오-비디오 기반 감정 인식 파이프라인에 통합해 gaze가 유의미한 상호작용 단서로 작동하는지 검증한다.

---

## Repository Structure

```text
gaze_telme/
├── TelME/                  # Upstream TelME source (MELD/ subdir patched)
│   └── MELD/               # Patched: dataset.py, utils.py support gaze_vec
├── sharingan/              # Upstream Sharingan source (weights/checkpoints excluded)
│   └── src/                # Gaze model architecture
│
├── pipeline/               # End-to-end preprocessing pipeline
│   ├── extract_clips.py    # Phase 2.1 — symlink/copy clips from MELD.Raw
│   ├── sample_frames.py    # Phase 2.2 — uniform frame sampling (FPS=6, max=32)
│   ├── detect_faces.py     # Phase 2.3 — RetinaFace/MTCNN + DeepSORT tracking
│   ├── gaze_infer.py       # Phase 2.4 — Sharingan batched gaze inference
│   └── build_features.py   # Phase 3   — R^6 gaze feature vector + z-score scaler
│
├── models/
│   ├── gaze_projector.py   # GazeProjector MLP: R^6 → R^768
│   └── video_gaze_student.py  # VideoGazeStudent (TimeSformer + gaze fusion)
│
├── train/
│   ├── train_student_gaze.py  # Phase 6 — KD training with gaze
│   └── train_fusion.py        # Phase 7 — ASF fusion (multi-seed)
│
├── eval/
│   └── eval_all.py         # Phase 8 — 4-condition evaluation + table
│
├── configs/
│   ├── base.yaml           # Shared hyperparameters and paths
│   ├── phase5_video_gaze.yaml  # Gaze student config
│   └── phase7_fusion.yaml  # Fusion config
│
├── utils/
│   ├── seed.py             # Deterministic seed utility
│   ├── metrics.py          # weighted-F1, aggregate_seeds, paired t-test
│   └── logger.py           # W&B + JSONL dual logger
│
├── analysis/               # Notebooks and visualization scripts
│   ├── eda_gaze.ipynb
│   ├── quant.ipynb
│   ├── qual.ipynb
│   ├── viz_gaze_overlay.py
│   ├── viz_feature_dist.py
│   └── viz_compare.py
│
├── tests/
│   ├── test_dataloader.py
│   └── test_video_gaze_student.py
│
├── scripts/                # Shell runners
│   ├── run_phase1.sh       # Phase 1  — TelME baseline training
│   ├── run_phase2.sh       # Phase 2-3 — gaze pipeline
│   ├── run_phase5.sh       # Phase 6-7 — gaze student + fusion training
│   └── run_eval.sh         # Phase 8  — evaluation
│
├── data/                   # Excluded from git (see .gitignore)
├── features/               # Excluded from git
├── results/                # Evaluation outputs
└── logs/                   # Training logs
```

---

## TelME Patches

Two files in `TelME/MELD/` are patched to support gaze:

| File | Change |
|------|--------|
| `TelME/MELD/dataset.py` | Added `gaze_pkl` param; loads `features/gaze/{split}.pkl`; attaches `gaze_vec` per utterance turn; falls back to `zeros(6)` for missing keys |
| `TelME/MELD/utils.py` | `make_batchs` auto-detects gaze from turn length (5 elements); returns 6-tuple `(..., batch_gaze_t)` when active |

All other files in `TelME/MELD/` are unmodified upstream code. `TelME/IEMOCAP/` is excluded entirely.

---

## Setup

### 0. Clone this repository

```bash
git clone https://github.com/JinWoong-Jung/gaze_telme.git
cd gaze_telme
```

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate gaze_telme
```

If `conda env create` fails on your platform, install from `requirements.txt` instead:

```bash
pip install -r requirements.txt
```

`ffmpeg` must also be available in your shell environment.

### 2. Add upstream source directories

`TelME/` and `sharingan/` are included in this repository as plain directories (patched sources only — no model weights). If they are missing locally, clone them:

```bash
git clone https://github.com/yuntaeyang/TelME.git TelME
git clone https://github.com/idiap/sharingan.git sharingan
```

Then re-apply the patches from `TelME/MELD/dataset.py` and `TelME/MELD/utils.py` if needed.

### 3. Download pretrained extractor models

The pipeline uses the following Hugging Face models:

| Role | Model ID |
|------|----------|
| Text teacher | `roberta-large` |
| Audio student | `facebook/data2vec-audio-base-960h` |
| Video student | `facebook/timesformer-base-finetuned-k400` |
| Video processor | `MCG-NJU/videomae-base` |

Pre-download into `pretrained/`:

```bash
python - <<'PY'
from transformers import (
    RobertaTokenizer, RobertaModel,
    AutoProcessor, AutoImageProcessor,
    Data2VecAudioModel, TimesformerModel,
)
RobertaTokenizer.from_pretrained("roberta-large", cache_dir="pretrained/roberta-large")
RobertaModel.from_pretrained("roberta-large", cache_dir="pretrained/roberta-large")
AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h", cache_dir="pretrained/data2vec-audio-base-960h")
Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h", cache_dir="pretrained/data2vec-audio-base-960h")
AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", cache_dir="pretrained/videomae-base")
TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", cache_dir="pretrained/timesformer-base-finetuned-k400")
PY
```

You may also let Hugging Face use its default cache under `~/.cache/huggingface/`.

### 4. Download Sharingan checkpoints and weights

Sharingan requires pretrained files for gaze inference:

```bash
# checkpoints (childplay.pt, gazefollow.pt, videoattentiontarget.pt)
wget -O /tmp/sharingan_checkpoints.tar.gz \
  "https://zenodo.org/records/14066123/files/sharingan_checkpoints.tar.gz?download=1"
tar -xzf /tmp/sharingan_checkpoints.tar.gz -C sharingan/checkpoints/

# weights (gaze360_resnet18.pt, multimae .pth, yolov5m_crowdhuman.pt)
wget -O /tmp/sharingan_weights.tar.gz \
  "https://zenodo.org/records/14066123/files/sharingan_weights.tar.gz?download=1"
tar -xzf /tmp/sharingan_weights.tar.gz -C sharingan/weights/
```

### 5. Download and place the MELD dataset

```bash
mkdir -p data/MELD.Raw
tar -xzf data/MELD.Raw.tar.gz -C data/MELD.Raw
```

Expected layout:

```text
data/MELD.Raw/
├── train_sent_emo.csv
├── dev_sent_emo.csv
├── test_sent_emo.csv
└── {train,dev,test}_splits_complete/   # video clips
```

### 6. Verify the environment

```bash
python scripts/check_env.py
```

All checks should pass (`torch`, `transformers`, `cv2`, `ffmpeg`). On macOS, `decord` is optional.

---

## Running the Pipeline

### Phase 1 — TelME baseline (requires GPU)

```bash
bash scripts/run_phase1.sh
```

Trains the vanilla TelME model on MELD. Saves checkpoint to `models/checkpoints/telme_baseline.pt`.

### Phase 2-3 — Gaze feature extraction

```bash
bash scripts/run_phase2.sh [train|dev|test|all]
```

Runs the full preprocessing pipeline per split:

1. `extract_clips.py` — symlink clips from `data/MELD.Raw/`
2. `sample_frames.py` — uniform sampling at 6 FPS, max 32 frames
3. `detect_faces.py` — MTCNN face detection + DeepSORT tracking → `features/cache/faces/`
4. `gaze_infer.py` — Sharingan batched inference → `features/cache/gaze/`
5. `build_features.py` — R^6 feature vector + z-score normalization → `features/gaze/{split}.pkl`

### Phase 6-7 — Gaze student + ASF fusion (requires GPU)

```bash
# Default: gated fusion, λ=0.3
bash scripts/run_phase5.sh

# Override: additive fusion, λ=0.5
bash scripts/run_phase5.sh add 0.5
```

Trains `VideoGazeStudent` (Phase 6) then the full ASF fusion model across seeds 42/7/2024 (Phase 7).

### Phase 8 — Evaluation

```bash
bash scripts/run_eval.sh
```

Evaluates 4 conditions and prints a markdown table to `results/table_main.md`:

| Condition | Description |
|-----------|-------------|
| A — baseline | TelME without gaze |
| B — ours | TelME + gaze features |
| C — random gaze | gaze replaced with random noise |
| D — zero gaze | gaze zeroed out |

---

## Gaze Feature Vector (R^6)

Each utterance is summarized by a 6-dimensional feature vector, z-score normalized from the training set:

| Dim | Name | Description |
|-----|------|-------------|
| 0 | `p_face` | Fraction of frames with a detected face |
| 1 | `p_out` | Fraction of frames where gaze is out-of-frame |
| 2 | `switch_rate` | Rate of gaze target switches (L2 diff > 0.1 threshold) |
| 3 | `entropy` | Spatial entropy of gaze distribution (4×4 grid) |
| 4 | `target_count` | Estimated number of gaze targets (KMeans, k≤3) |
| 5 | `p_center` | Fraction of gaze points in the central [0.3, 0.7]² region |

---

## Model Checkpoints

Checkpoints are saved to `models/checkpoints/` (excluded from git via `.gitignore`):

| File | Phase | Description |
|------|-------|-------------|
| `telme_baseline.pt` | 1 | Vanilla TelME on MELD |
| `video_gaze_student_{fusion}_{lambda}.pt` | 6 | VideoGazeStudent checkpoint |
| `fusion_seed{seed}.pt` | 7 | ASF fusion model per seed |
