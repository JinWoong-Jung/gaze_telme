# Gaze-aware MERC — 프로젝트 TODO

---

## 1. 프로젝트 개요

### 한 줄 요약
**MELD 감정 인식 대화(MERC) 태스크에서, Sharingan이 생성한 pseudo-gaze 신호를 TelME 멀티모달 증류 구조에 주입하여 성능을 개선한다.**

### 아키텍처
```
텍스트  → RoBERTa-large  (Teacher, 고정)
오디오  → Data2Vec       (Audio Student)
비디오  → TimeSformer    (Video Student)
Gaze   → Sharingan      (pseudo-gaze → R^6 피처)
              │
              ▼
      GazeProjector (R^6 → R^768)
              │
      Video + Gaze Fusion  ← [add | concat_proj | gated] 중 선택
              │
      ASF (Attentive Score Fusion)  ← Teacher-leading
              │
      7-class 감정 분류 (anger / disgust / fear / joy / neutral / sadness / surprise)
```

### 데이터셋
- **MELD** (Multimodal EmotionLines Dataset): Friends TV 드라마 기반 대화 감정 데이터
- 분할: train ≈ 9,989 / dev ≈ 1,109 / test ≈ 2,610 발화

### 핵심 연구 질문
| # | 질문 |
|---|------|
| Q1 | Gaze 신호가 Weighted-F1을 유의하게 향상시키는가? (p < 0.05) |
| Q2 | 어떤 감정 클래스가 Gaze로 가장 이득을 보는가? |
| Q3 | Random/Zero gaze 대비 실제 gaze가 유의미한 차이를 보이는가? |

---

## 2. 현재 구현 완료 항목 (코드 준비 완료, 실행은 GPU 필요)

### 프로젝트 구조
```
gaze_erc/
├── configs/               # 실험 설정 YAML
│   ├── base.yaml          # 공통 하이퍼파라미터
│   ├── phase1_baseline.yaml
│   ├── phase5_video_gaze.yaml
│   └── phase7_fusion.yaml
├── pipeline/              # 데이터 처리 파이프라인
│   ├── extract_clips.py   # 발화 클립 정리
│   ├── sample_frames.py   # 프레임 샘플링 (FPS=6, max=32)
│   ├── detect_faces.py    # RetinaFace + DeepSORT 얼굴 검출·트래킹
│   ├── gaze_infer.py      # Sharingan gaze 추론
│   └── build_features.py  # R^6 gaze_vec 생성 + z-score 정규화
├── models/                # 모델 정의
│   ├── gaze_projector.py  # R^6 → R^768 MLP
│   └── video_gaze_student.py  # 3가지 fusion 방식 포함
├── train/                 # 학습 스크립트
│   ├── train_student_gaze.py  # Phase 6: VideoGazeStudent KD 학습
│   └── train_fusion.py        # Phase 7: ASF 융합 학습 (3-seed)
├── eval/                  # 평가 스크립트
│   └── eval_all.py        # A/B/C/D 조건 전체 평가
├── analysis/              # 분석 & 시각화
│   ├── eda_gaze.ipynb     # 피처 분포 탐색
│   ├── quant.ipynb        # 정량 분석 (confusion matrix, F1 delta)
│   ├── qual.ipynb         # 정성 분석 (오분류 샘플 20개)
│   ├── viz_gaze_overlay.py    # 프레임 위 gaze 오버레이
│   ├── viz_feature_dist.py    # 감정별 피처 분포
│   └── viz_compare.py     # Baseline vs Ours 비교 차트
├── utils/                 # 공용 유틸리티
│   ├── seed.py            # 재현성 시드 고정
│   ├── metrics.py         # Acc / Weighted-F1 / Macro-F1 / t-test
│   └── logger.py          # W&B + JSONL 로깅 래퍼
├── tests/                 # 단위 테스트
│   ├── test_dataloader.py
│   └── test_video_gaze_student.py
└── scripts/               # 셸 실행 진입점
    ├── check_env.py
    ├── run_phase1.sh
    ├── run_phase2.sh
    ├── run_phase5.sh
    └── run_eval.sh
```

### TelME 수정 사항 (완료)
| 파일 | 변경 내용 |
|------|----------|
| `TelME/MELD/preprocessing.py` | 데이터 경로를 `data/MELD.Raw/`로 패치 |
| `TelME/MELD/dataset.py` | `gaze_pkl` 인자 추가 — 각 발화에 gaze_vec 첨부 |
| `TelME/MELD/utils.py` | `make_batchs`가 gaze 포함 시 6-tuple 반환 (하위 호환 유지) |

---

## 3. 앞으로 해야 할 작업 (GPU 서버 기준 순서)

> **전제조건**: GPU 서버에서 `conda activate gaze_telme` 후 프로젝트 루트에서 실행

---

### Step 0 — 환경 확인 (5분)

**무엇을 하는가**
GPU, ffmpeg, 주요 라이브러리가 정상적으로 설치되었는지 점검한다.

**어떻게 하는가**
```bash
python scripts/check_env.py
```

**완료 기준**
```
[OK] torch: torch=2.1.0, cuda=True, mps=False
[OK] ffmpeg: ffmpeg version ...
[OK] transformers: ...
→ Environment sanity check passed.
```

---

### Step 1 — TelME Baseline 학습 (약 6–12 시간)

**무엇을 하는가**
비교 기준(Baseline)이 될 TelME 원본 모델을 학습한다.
Teacher(RoBERTa) → Student(Audio + Video) → ASF Fusion 순으로 3단계 학습.
이 결과가 논문 재현 수치와 일치해야 이후 실험의 신뢰도가 담보된다.

**어떻게 하는가**
```bash
bash scripts/run_phase1.sh
```
학습 완료 후 생성 파일:
```
models/checkpoints/teacher.bin
models/checkpoints/student_audio/total_student.bin
models/checkpoints/student_video/total_student.bin
models/checkpoints/total_fusion.bin
results/baseline.json
```

**완료 기준**
- `results/baseline.json`에 Weighted-F1이 논문 수치(≈ 0.65)와 ±0.5%p 이내
- 불일치 시 seed/epoch 조정 후 재시도

---

### Step 2 — 데이터 전처리 파이프라인 실행 (약 2–4 시간)

**무엇을 하는가**
MELD 비디오에서 Sharingan이 처리할 수 있는 형태의 입력을 생성한다.
4단계 파이프라인: 클립 정리 → 프레임 샘플링 → 얼굴 검출 → gaze 추론.

| 단계 | 스크립트 | 출력 |
|------|----------|------|
| 2.1 클립 정리 | `pipeline/extract_clips.py` | `data/processed/clips/` |
| 2.2 프레임 샘플링 | `pipeline/sample_frames.py` | `data/processed/frames/` (FPS=6, max=32장) |
| 2.3 얼굴 검출 | `pipeline/detect_faces.py` | `features/cache/faces/*.npz` |
| 2.4 Gaze 추론 | `pipeline/gaze_infer.py` | `features/cache/gaze/*.npz` |

**어떻게 하는가**
```bash
bash scripts/run_phase2.sh all    # train / dev / test 모두 처리
# 또는 분할별로:
bash scripts/run_phase2.sh train
bash scripts/run_phase2.sh dev
bash scripts/run_phase2.sh test
```

**완료 기준**
- `features/cache/gaze/train/`, `dev/`, `test/` 아래 `.npz` 파일 생성률 ≥ 98%
- `features/cache/gaze/missing.txt` 존재 시 누락 발화 확인

> **주의**: Sharingan 저장소 클론과 가중치 다운로드가 먼저 필요하다.
> ```bash
> git clone <sharingan-repo-url> sharingan
> # 가중치를 sharingan_ckpt/ 에 배치
> ```

---

### Step 3 — Gaze 피처 엔지니어링 (Step 2에 포함, 별도 실행 가능)

**무엇을 하는가**
프레임 단위 raw gaze 출력을 발화 단위 6차원 벡터(R^6)로 집약한다.
각 피처의 의미:

| 인덱스 | 피처 | 의미 |
|--------|------|------|
| 0 | `p_face` | 얼굴을 응시한 프레임 비율 |
| 1 | `p_out` | 화면 밖을 응시한 프레임 비율 |
| 2 | `switch_rate` | 연속 프레임 간 gaze 목표 전환 빈도 |
| 3 | `entropy` | gaze 분산도 (4×4 grid 히스토그램) |
| 4 | `target_count` | 서로 다른 gaze 클러스터 수 |
| 5 | `p_center` | 화면 중앙([0.3,0.7]²)을 응시한 비율 |

train 기준 z-score 정규화 → `features/gaze/scaler.pkl` 저장 후 dev/test 재사용.

**어떻게 하는가**
`bash scripts/run_phase2.sh`에 포함되어 자동 실행됨. 수동 실행:
```bash
python pipeline/build_features.py --split train   # 스케일러 fit + transform
python pipeline/build_features.py --split dev
python pipeline/build_features.py --split test
```

**완료 기준**
- `features/gaze/train.pkl`, `dev.pkl`, `test.pkl` 생성
- zero-vector 비율 < 5%
- `analysis/eda_gaze.ipynb` 실행하여 분포 이상 여부 확인

---

### Step 4 — 단위 테스트 통과 확인 (10분)

**무엇을 하는가**
Step 2-3이 완료된 후, 모델 코드와 데이터로더가 올바르게 동작하는지 검증한다.
gaze_vec shape, batch 구성, grad flow 등을 자동으로 확인한다.

**어떻게 하는가**
```bash
python -m pytest tests/ -v
```

**완료 기준**
```
tests/test_dataloader.py        PASSED (gaze on/off 모두)
tests/test_video_gaze_student.py PASSED (3 fusion type × 여러 batch size)
```

---

### Step 5 — VideoGazeStudent 학습 (약 4–8 시간)

**무엇을 하는가**
Gaze가 주입된 비디오 학생 모델을 Knowledge Distillation으로 학습한다.
기존 TimeSformer Video Student를 `VideoGazeStudent`로 교체하고,
GazeProjector(R^6→R^768)로 gaze_vec을 비디오 hidden에 융합한다.

3가지 융합 방식 실험:
| 방식 | 수식 |
|------|------|
| `add` | `video_h + λ × gaze_h`, λ ∈ {0.1, 0.3, 0.5, 1.0} |
| `concat_proj` | `Linear([video_h; gaze_h] → 768)` |
| `gated` | `g = σ(W·[video;gaze]); out = g·video + (1-g)·gaze_h` |

**어떻게 하는가**
```bash
bash scripts/run_phase5.sh gated 0.3    # 권장 (기본값)
bash scripts/run_phase5.sh add 0.5      # λ sweep용
bash scripts/run_phase5.sh concat_proj  # 비교용
```

또는 직접:
```bash
python train/train_student_gaze.py \
    --config configs/phase5_video_gaze.yaml \
    --fusion-type gated \
    --fusion-lambda 0.3
```

**완료 기준**
- `models/checkpoints/student_video_gaze/total_student.bin` 생성
- dev Weighted-F1 ≥ 기존 video student baseline (퇴보 없음)

---

### Step 6 — ASF 융합 학습 (약 2–4 시간 × 3 seed)

**무엇을 하는가**
Step 5에서 학습된 VideoGazeStudent를 비디오 브랜치로 사용하여
ASF(Attentive Score Fusion) 헤드를 재학습한다.
텍스트(Teacher) + 오디오 + Gaze 주입 비디오 → 최종 감정 분류.
3개 seed(42, 7, 2024)로 학습하여 평균±표준편차를 보고한다.

**어떻게 하는가**
Step 5 완료 후 `run_phase5.sh`가 자동으로 이어서 실행함. 또는 직접:
```bash
python train/train_fusion.py --config configs/phase7_fusion.yaml
python train/train_fusion.py --config configs/phase7_fusion.yaml --seed 7
python train/train_fusion.py --config configs/phase7_fusion.yaml --seed 2024
```

**완료 기준**
- `models/checkpoints/total_fusion_gaze_seed{42,7,2024}.bin` 생성
- `results/ours.json`에 3-seed 평균 + 표준편차 기록

---

### Step 7 — 전체 평가 실행 (1시간)

**무엇을 하는가**
4가지 조건을 동일한 테스트 셋으로 평가하여 Gaze의 효과를 검증한다.

| 조건 | 설명 |
|------|------|
| A. Baseline TelME | 원본 TelME (gaze 없음) |
| B. + Gaze (ours) | Gaze 주입 모델 |
| C. Random gaze | gaze_vec ~ N(0,1) (sanity check) |
| D. Zero gaze | gaze_vec = 0 (ablation) |

**어떻게 하는가**
```bash
bash scripts/run_eval.sh
```
또는 직접:
```bash
python eval/eval_all.py --out results/
```

**완료 기준**
- `results/eval_all.json` 생성 (A/B/C/D 전체 지표)
- `results/table_main.md` 생성 (논문용 비교 표)
- B vs A의 Weighted-F1 향상 + p-value < 0.05 확인

---

### Step 8 — 분석 및 시각화 (2–4 시간)

**무엇을 하는가**
결과의 원인을 파악하고 논문 제출용 그림을 생성한다.

**어떻게 하는가**

*정량 분석 (Jupyter)*
```bash
jupyter notebook analysis/quant.ipynb
# - Confusion matrix 비교 (A vs B)
# - Class-wise F1 delta 바 차트
# - 감정별 gaze 피처 box plot
```

*정성 분석 (Jupyter)*
```bash
jupyter notebook analysis/qual.ipynb
# - A 틀리고 B 맞춘 샘플 20개 수집
# - results/error_analysis.md 저장
```

*시각화 스크립트*
```bash
python analysis/viz_gaze_overlay.py --dialogue 0 --utterance 3 --split dev
python analysis/viz_feature_dist.py --split train --out results/figures/
python analysis/viz_compare.py --out results/figures/
```

**완료 기준**
- `results/figures/` 에 논문용 그림 일체
- `results/error_analysis.md` 작성

---

## 4. 전체 실행 순서 요약

```bash
# 0. 환경 확인
python scripts/check_env.py

# 1. TelME Baseline 학습
bash scripts/run_phase1.sh

# 2-3. 전처리 + gaze 피처 생성 (train/dev/test 모두)
bash scripts/run_phase2.sh all

# 4. 단위 테스트
python -m pytest tests/ -v

# 5-6. VideoGazeStudent + ASF 융합 학습
bash scripts/run_phase5.sh gated 0.3

# 7. 전체 평가
bash scripts/run_eval.sh

# 8. 분석 및 시각화
jupyter notebook analysis/quant.ipynb
jupyter notebook analysis/qual.ipynb
python analysis/viz_compare.py --out results/figures/
```

---

## 5. 체크리스트

### 실행 전 준비
- [ ] GPU 서버 `gaze_telme` conda 환경 설치 완료
- [ ] `sharingan/` 저장소 클론 완료
- [ ] `sharingan_ckpt/` 가중치 배치 완료
- [ ] `data/MELD.Raw/` 압축 해제 완료

### 실행 단계
- [ ] Step 0: `check_env.py` 전체 OK
- [ ] Step 1: `results/baseline.json` 생성, 논문 수치와 ±0.5%p 이내
- [ ] Step 2: gaze cache `.npz` 생성률 ≥ 98%
- [ ] Step 3: `features/gaze/*.pkl` 생성, zero-vector < 5%
- [ ] Step 4: `pytest tests/` 전체 PASS
- [ ] Step 5: `student_video_gaze/total_student.bin` 생성
- [ ] Step 6: `results/ours.json` (3-seed 평균 기록)
- [ ] Step 7: `results/table_main.md` 생성, B > A 확인
- [ ] Step 8: `results/figures/` 그림 일체, `error_analysis.md` 작성

### 최종 산출물
- [ ] `results/baseline.json` — TelME 원본 수치
- [ ] `results/ours.json` — Gaze 모델 3-seed 평균
- [ ] `results/table_main.md` — 논문 메인 비교 표
- [ ] `results/error_analysis.md` — 오류 분석
- [ ] `results/figures/` — 논문용 그림 일체
- [ ] `models/checkpoints/*.bin` — 모든 체크포인트

---

## 6. 주요 리스크 및 대응

| 리스크 | 증상 | 대응 |
|--------|------|------|
| Sharingan↔MELD 도메인 갭 | gaze_vec 품질 저하 | 100샘플 수동 검토 후 `analysis/viz_gaze_overlay.py`로 확인 |
| 얼굴 미검출 발화 증가 | zero-vector > 5% | `pipeline/detect_faces.py` threshold 낮추기 (0.9 → 0.7) |
| Gaze 주입 후 성능 퇴보 | B < A | fusion λ sweep, gaze dropout 증가, `use_gaze=False`로 baseline 재확인 |
| 3-seed 분산 큼 | std > 0.005 | seed 추가 (2025, 123) 후 평균 재계산 |
| OOM (GPU 메모리 부족) | CUDA OOM | batch_size 2로 축소, `torch.cuda.empty_cache()` 주기 줄이기 |
