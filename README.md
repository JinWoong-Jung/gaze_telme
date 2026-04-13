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
The goal is to extract utterance-level gaze features from MELD video clips and inject them into the TelME pipeline to test whether gaze can provide useful interaction cues for emotion recognition. In this repository, `TelME/` and `sharingan/` are kept as upstream model sources, while all project-specific integration code is implemented under `erc'/`.


이 프로젝트는 MELD 데이터셋에서 대화 감정 인식 성능을 높이기 위해, MERC 모델인 `TelME`와 gaze estimation 모델인 `Sharingan`을 결합하는 것을 목표로 한다. `Sharingan`으로부터 MELD 영상의 pseudo-gaze 신호를 추출하고, 이를 `TelME`의 텍스트-오디오-비디오 기반 감정 인식 파이프라인에 통합해 gaze가 유의미한 상호작용 단서로 작동하는지 검증한다.
이 저장소에서는 `TelME/`와 `sharingan/`을 각각 upstream 모델 저장소로 유지하고, 두 모델을 연결하는 전처리, 특징 추출, 융합, 학습 및 평가 코드는 `erc/` 아래에서 구현한다.


## Setup
### 1. Create the Conda environment

### 2. Download data

### 3. Configure the paths in the configuration files

## Training and Testing

## Model Checkpoints