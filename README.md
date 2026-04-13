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
This project explores gaze-aware multimodal emotion recognition in conversation on the MELD dataset. We combine two existing models: `TelME`, a MERC baseline that fuses text, audio, and video, and `Sharingan`, a gaze-following model used here to derive pseudo-gaze signals from dialogue videos.
The goal is to extract utterance-level gaze features from MELD video clips and inject them into the TelME pipeline to test whether gaze can provide useful interaction cues for emotion recognition. In this repository, `TelME/` and `sharingan/` are kept as upstream model sources, while all project-specific integration code is implemented under `erc/`.


이 프로젝트는 MELD 데이터셋에서 대화 감정 인식 성능을 높이기 위해, MERC 모델인 `TelME`와 gaze estimation 모델인 `Sharingan`을 결합하는 것을 목표로 한다. `Sharingan`으로부터 MELD 영상의 pseudo-gaze 신호를 추출하고, 이를 `TelME`의 텍스트-오디오-비디오 기반 감정 인식 파이프라인에 통합해 gaze가 유의미한 상호작용 단서로 작동하는지 검증한다.
이 저장소에서는 `TelME/`와 `sharingan/`을 각각 upstream 모델 저장소로 유지하고, 두 모델을 연결하는 전처리, 특징 추출, 융합, 학습 및 평가 코드는 `erc/` 아래에서 구현한다.


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

If `conda env create` fails on your platform, you can install from `requirements.txt` instead.

```bash
pip install -r requirements.txt
```

`ffmpeg` must also be available in your shell environment.

### 2. Prepare the upstream repositories

This project assumes the following layout:

```text
gaze_telme/
├── TelME/
├── sharingan/
└── erc/
```

After cloning this repository, make sure both `TelME/` and `sharingan/` exist locally.

- `sharingan/` is expected at the repository root.
- `TelME/` may need to be cloned manually if it was not copied into your local checkout as a normal directory.

If needed, run:

```bash
git clone https://github.com/yuntaeyang/TelME.git TelME
git clone https://github.com/idiap/sharingan.git sharingan
```

### 3. Download pretrained extractor models

The current code expects the following pretrained Hugging Face models:

- text: `roberta-large`
- audio: `facebook/data2vec-audio-base-960h`
- video: `facebook/timesformer-base-finetuned-k400`
- video processor: `MCG-NJU/videomae-base`

You can pre-download them into `pretrained/` with:

```bash
mkdir -p pretrained

python - <<'PY'
from transformers import (
    RobertaTokenizer, RobertaModel,
    AutoProcessor, AutoImageProcessor,
    Data2VecAudioModel, TimesformerModel
)

RobertaTokenizer.from_pretrained("roberta-large", cache_dir="pretrained/roberta-large")
RobertaModel.from_pretrained("roberta-large", cache_dir="pretrained/roberta-large")

AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h", cache_dir="pretrained/data2vec-audio-base-960h")
Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h", cache_dir="pretrained/data2vec-audio-base-960h")

AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", cache_dir="pretrained/videomae-base")
TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", cache_dir="pretrained/timesformer-base-finetuned-k400")
PY
```

If you prefer, you can also let Hugging Face cache these models under `~/.cache/huggingface/`.

### 4. Download Sharingan checkpoints and weights

For gaze extraction, Sharingan requires its own pretrained files.

- checkpoints: https://zenodo.org/records/14066123/files/sharingan_checkpoints.tar.gz?download=1
- weights: https://zenodo.org/records/14066123/files/sharingan_weights.tar.gz?download=1

Unpack them under:

```text
sharingan/checkpoints/
sharingan/weights/
```

### 5. Download and place the MELD dataset

Create the following data layout:

```text
data/
└── MELD.Raw/
    ├── train_meld_emo.csv
    ├── dev_meld_emo.csv
    ├── test_meld_emo.csv
    ├── train_sent_emo.csv
    ├── dev_sent_emo.csv
    ├── test_sent_emo.csv
    └── ... video files ...
```

If you have the raw archive, unpack it with:

```bash
mkdir -p data/MELD.Raw
tar -xzf data/MELD.Raw.tar.gz -C data/MELD.Raw
```

### 6. Verify the local environment

Run:

```bash
python scripts/check_env.py
```

This should confirm that `torch`, `transformers`, `cv2`, and `ffmpeg` are available. On macOS, `decord` is optional.

### 7. Notes before training

- `Phase 1` training requires substantial compute. CPU-only environments are suitable for setup and smoke tests, but not for full TelME reproduction.
- If you use offline mode for Transformers, make sure all required pretrained models are already cached locally.
- Keep custom integration code under `erc/`, and treat `TelME/` and `sharingan/` as upstream source directories.

## Training and Testing

## Model Checkpoints
