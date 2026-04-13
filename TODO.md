# 📘 Gaze-aware MERC with TelME — Project TODO

## 🎯 Objective
Sharingan 기반 **pseudo-gaze 신호**를 TelME 구조에 주입하여 MELD 데이터셋의
Multimodal Emotion Recognition in Conversation (MERC) 성능을 개선한다.

## 🧠 Final Architecture
```
Text  (Teacher: RoBERTa-large)
Audio (Student: Data2Vec)
Video (Student: TimeSformer) ─┐
Gaze  (Sharingan pseudo-feat) ─┴─► Video–Gaze Fusion (student-level)
                                   └─► Teacher-leading ASF Fusion
                                       └─► Emotion Prediction (7-class)
```

## 📂 Repository Layout (목표 구조)
```
gaze_erc/
├── configs/                 # yaml 기반 실험 설정 (hydra or OmegaConf)
│   ├── base.yaml
│   ├── phase1_baseline.yaml
│   ├── phase5_video_gaze.yaml
│   └── phase7_fusion.yaml
├── data/
│   ├── MELD.Raw.tar.gz      # (존재) 원본 배포 아카이브
│   ├── MELD.Raw/            # tar 해제 결과 (train/dev/test .mp4 + .csv)
│   └── processed/
│       ├── clips/{train,dev,test}/dia{D}_utt{U}.mp4
│       └── frames/{train,dev,test}/dia{D}_utt{U}/frame_{i:03d}.jpg
├── preprocess/
│   ├── extract_clips.py     # csv 기반 utterance 단위 슬라이싱
│   ├── sample_frames.py     # FPS 리샘플 + 최대 프레임 제한
│   └── detect_faces.py      # RetinaFace + DeepSORT 트래킹
├── gaze/
│   └── sharingan_infer.py   # frame → gaze_map/gaze_point/inout
├── sharingan/               # Sharingan upstream repo
├── sharingan_ckpt/          # Sharingan 사전학습 가중치
├── features/
│   ├── gaze/{train,dev,test}.pkl   # utterance-level gaze_vec ∈ R^6
│   └── cache/                       # 프레임 단위 raw 출력 (.npz)
├── models/
│   ├── gaze_projector.py
│   ├── video_gaze_student.py
│   └── checkpoints/
│       ├── teacher.bin
│       ├── student_audio.bin
│       ├── student_video.bin
│       ├── student_video_gaze.bin
│       └── total_fusion.bin
├── TelME/                   # clone 대상 (.gitignore에 포함)
├── scripts/
│   ├── run_phase1_baseline.sh
│   ├── run_phase2_gaze.sh
│   ├── run_phase5_train.sh
│   └── run_eval.sh
├── utils/
│   ├── seed.py
│   ├── metrics.py           # Acc / Weighted-F1 / Macro-F1
│   └── logger.py            # wandb or tensorboard wrapper
├── erc/                     # 분석/시각화 노트북 및 스크립트
├── requirements.txt
├── environment.yml
└── TODO.md
```

---

## 📌 Phase 0. Environment Setup - Finished

### TODO
- [O] **conda 환경 생성 & 고정**
  ```bash
  conda create -n gaze_telme python=3.9 -y
  conda activate gaze_telme
  # GPU: macOS에서는 MPS, 학습 서버에서는 CUDA 11.8 가정
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
      --index-url https://download.pytorch.org/whl/cu118
  pip install transformers==4.36.2 torch-geometric==2.4.0 \
      scikit-learn pandas numpy tqdm einops opencv-python \
      decord ffmpeg-python omegaconf wandb
  pip install facenet-pytorch retina-face deep-sort-realtime
  ```
- [O] `requirements.txt`, `environment.yml` 커밋으로 버전 고정
- [O] **TelME clone**
  ```bash
  git clone https://github.com/yuntaeyang/TelME.git TelME
  ```
- [O] **Sharingan 저장소 clone & 가중치 다운로드**
  ```bash
  git clone <sharingan-repo> sharingan
  # 사전학습 ckpt → sharingan_ckpt/
  ```
- [O] **MELD 원본 압축 해제**
  ```bash
  mkdir -p data/MELD.Raw && tar -xzf data/MELD.Raw.tar.gz -C data/MELD.Raw
  ```
- [O] `.gitignore`에 `data/MELD.Raw/`, `data/processed/`, `features/cache/`, `TelME/`, `*.bin`, `*.pkl`, `wandb/` 추가
- [O] `utils/seed.py`로 seed 고정 (torch, numpy, random, `CUBLAS_WORKSPACE_CONFIG`)
- [O] **Sanity check 스크립트**: `python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"`

### 완료 기준
- `conda activate gaze_telme` 후 `python scripts/check_env.py`가 torch/GPU/ffmpeg/decord 모두 OK 출력

---

## 📌 Phase 1. TelME Baseline Reproduction

### TODO
- [O] TelME 원본 MELD 데이터 경로를 `data/MELD.Raw`에 맞게 수정
  - `TelME/MELD/preprocessing.py`의 하드코딩 경로 확인 및 patch
- [ ] 각 단계 실행 및 로그 저장 (`logs/phase1/*.log`)
  ```bash
  python TelME/MELD/teacher.py  2>&1 | tee logs/phase1/teacher.log
  python TelME/MELD/student.py  2>&1 | tee logs/phase1/student.log
  python TelME/MELD/fusion.py   2>&1 | tee logs/phase1/fusion.log
  ```
- [ ] 체크포인트 존재 검증
  - [ ] `teacher.bin`
  - [ ] `student_audio/total_student.bin`
  - [ ] `student_video/total_student.bin`
  - [ ] `total_fusion.bin`
- [ ] baseline metric 기록 (`results/baseline.json`)
  - [ ] Accuracy
  - [ ] Weighted-F1 (보고 기준)
  - [ ] Macro-F1
- [ ] 논문 수치와 ±0.5%p 이내 일치 확인 → 불일치 시 seed/epoch 재시도

### 완료 기준
- `results/baseline.json`에 세 지표가 기록되고 README에 표 형태로 정리

---

## 📌 Phase 2. MELD Video → Gaze Pipeline

### 🎯 Goal
MELD 발화 영상의 프레임 단위 gaze raw 출력을 캐시한다.

### TODO

#### 2.1 Utterance clip 추출 (`preprocess/extract_clips.py`)
- [ ] `train_sent_emo.csv` / `dev_sent_emo.csv` / `test_sent_emo.csv` 파싱
- [ ] `StartTime`, `EndTime` (HH:MM:SS,mmm) → seconds 변환
- [ ] `ffmpeg -ss {start} -to {end} -i {src.mp4} -c copy {dst.mp4}` 로 슬라이싱
- [ ] 실패 케이스(파일 없음/길이 0) → `data/processed/missing_clips.csv`
- [ ] 예상 산출: ≈13,708 clips (train) + dev + test

#### 2.2 Frame sampling (`preprocess/sample_frames.py`)
- [ ] **FPS = 6**, **max_frames = 32** (균등 샘플링)
- [ ] decord 사용 (opencv보다 빠름)
- [ ] 출력: `frames/{split}/dia{D}_utt{U}/frame_{i:03d}.jpg`
- [ ] 멀티프로세싱 (`multiprocessing.Pool`, num_workers=8)

#### 2.3 Face detection & tracking (`preprocess/detect_faces.py`)
- [ ] **Detector**: RetinaFace (facenet-pytorch) — threshold 0.9
- [ ] **Tracker**: deep-sort-realtime, max_age=10
- [ ] 프레임별 `bbox, track_id, conf` 저장 → `features/cache/faces/{split}/dia{D}_utt{U}.npz`
- [ ] **Edge case**: 얼굴 0개 프레임 → `bbox=None`, gaze 단계에서 처리

#### 2.4 Gaze inference (`gaze/sharingan_infer.py`)
- [ ] Sharingan 모델 로드 (`sharingan_ckpt/`)
- [ ] head bbox 필요 → 2.3 결과 재사용
- [ ] 배치 추론 (batch_size=16 권장)
  ```python
  gaze_map, gaze_point, inout = model(frame, head_bbox)
  ```
- [ ] 출력 스키마 (`features/cache/gaze/{split}/dia{D}_utt{U}.npz`):
  ```python
  {
    "dialogue_id": int,
    "utterance_id": int,
    "num_frames": int,
    "frame_points": np.ndarray,   # (N, K, 2)  K=faces per frame
    "frame_inout":  np.ndarray,   # (N, K)
    "frame_heatmap_stats": np.ndarray,  # (N, K, 4)  mean/std/max/entropy
    "frame_face_bbox": np.ndarray,       # (N, K, 4)
    "valid_mask": np.ndarray,            # (N,)
  }
  ```
- [ ] **GPU 메모리 관리**: torch.no_grad + `torch.cuda.empty_cache()` per 500 clips
- [ ] **재실행 안전성**: 이미 존재하는 .npz는 skip (`--resume`)

### 완료 기준
- train/dev/test 3-split 모두 `features/cache/gaze/` 아래 .npz 생성률 ≥ 98%
- 누락 목록 `features/cache/gaze/missing.txt` 기록

---

## 📌 Phase 3. Gaze Feature Engineering

### 🎯 Goal
프레임 단위 raw gaze → **utterance-level gaze_vec ∈ R^6**

### TODO
- [ ] `features/gaze/build_features.py` 작성
- [ ] Core features (수식 고정, 코드 주석에 정의 명시)
  - [ ] `p_face` = (얼굴을 응시한 프레임 수) / (유효 프레임 수)
  - [ ] `p_out` = (inout < 0.5인 프레임 수) / 유효 프레임 수
  - [ ] `switch_rate` = (연속 프레임 간 gaze target 변경 횟수) / (유효 프레임 수 - 1)
  - [ ] `entropy` = gaze point를 4x4 grid에 binning 후 Shannon entropy
  - [ ] `target_count` = 서로 다른 gaze target id 수 (간단 군집화: KMeans or DBSCAN)
  - [ ] `p_center` = gaze point가 중앙 [0.3, 0.7]^2에 위치한 비율
- [ ] 정규화: train set 기준 z-score → `features/gaze/scaler.pkl`로 저장 후 dev/test에 재사용
- [ ] 저장 포맷 (`features/gaze/{train,dev,test}.pkl`):
  ```python
  {(dialogue_id, utterance_id): np.float32 array of shape (6,)}
  ```
- [ ] **Edge cases**
  - [ ] 얼굴 미검출 utterance → zero vector + `is_valid=False` 플래그
  - [ ] gaze inference 실패 → 직전/직후 유효 프레임 linear fallback
  - [ ] 프레임 수 < 3 → zero vector
- [ ] **분포 sanity check**: 각 feature의 histogram을 `erc/eda_gaze.ipynb`로 확인

### 완료 기준
- `features/gaze/train.pkl` 로드 시 dict key 수 ≈ 발화 수, 값 shape = (6,)
- zero-vector 비율 < 5%

---

## 📌 Phase 4. TelME Data Pipeline Modification

### 🎯 Goal
TelME 데이터로더가 `gaze_vec`을 batch에 포함하도록 수정

### TODO
- [ ] **타겟 파일**
  - [ ] `TelME/MELD/preprocessing.py`
  - [ ] `TelME/MELD/utils.py` (collate_fn)
- [ ] Dataset `__getitem__`에 gaze 추가
  ```python
  gaze = self.gaze_cache.get((did, uid), np.zeros(6, dtype=np.float32))
  return text, audio, video, gaze, label
  ```
- [ ] `collate_fn`에 `gaze_batch = torch.stack([...]).to(device)` 추가
- [ ] **하위 호환**: `use_gaze` config flag로 on/off → baseline 재실행 가능하도록
- [ ] 단위 테스트: `tests/test_dataloader.py`
  - batch shape, device, dtype 확인

### 완료 기준
- `use_gaze=False`일 때 Phase 1 결과와 100% 동일 (결정론적)
- `use_gaze=True`일 때 batch tuple 길이 = 5

---

## 📌 Phase 5. Video + Gaze Fusion (Core)

### 🎯 Goal
Video student hidden에 gaze 표현을 주입

### TODO
- [ ] **타겟 파일**: `TelME/MELD/model.py` (또는 `models/video_gaze_student.py`로 분리)
- [ ] **GazeProjector 구현**
  ```python
  class GazeProjector(nn.Module):
      def __init__(self, in_dim=6, hidden=128, out_dim=768, p=0.1):
          super().__init__()
          self.mlp = nn.Sequential(
              nn.LayerNorm(in_dim),
              nn.Linear(in_dim, hidden),
              nn.GELU(),
              nn.Dropout(p),
              nn.Linear(hidden, out_dim),
          )
      def forward(self, gaze):       # (B, 6)
          return self.mlp(gaze)      # (B, 768)
  ```
- [ ] **Fusion 방식 (3가지 실험)**
  - [ ] `add`: `video_hidden + λ * gaze_hidden` (λ ∈ {0.1, 0.3, 0.5, 1.0})
  - [ ] `concat+proj`: `Linear(768+768 → 768)`
  - [ ] `gated`: `g = σ(W[video;gaze]); out = g*video + (1-g)*gaze`
- [ ] hidden_dim **768 고정** (TimeSformer / RoBERTa 호환)
- [ ] `configs/phase5_video_gaze.yaml`에 fusion_type, lambda 노출
- [ ] 단위 테스트: random input → output shape (B, 768)

### 완료 기준
- `python -m pytest tests/test_video_gaze_student.py` 통과
- forward + backward 모두 grad 흐름 확인

---

## 📌 Phase 6. Student Training (Distillation)

### 🎯 Goal
Video+Gaze student를 TelME distillation 체계로 학습

### TODO
- [ ] **타겟 파일**: `TelME/MELD/student.py` 확장 → `scripts/train_student_video_gaze.py`
- [ ] 기존 video student를 `VideoGazeStudent`로 교체
- [ ] Loss 유지:  `L = CE + α·LogitDistill(KL) + β·FeatureDistill(MSE)`
  - α, β는 TelME 원본 값 그대로 사용
- [ ] optimizer: AdamW, lr=1e-5 (teacher feature 손상 방지)
- [ ] scheduler: linear warmup 10% + linear decay
- [ ] epoch = 10, patience = 3 (dev Weighted-F1)
- [ ] mixed precision (`torch.cuda.amp`)
- [ ] **체크포인트**: `models/checkpoints/student_video_gaze.bin`
- [ ] wandb 로깅 (`project=gaze_telme`, run name = phase5_fusion_type + λ)

### 완료 기준
- dev Weighted-F1이 Phase 1 video student 대비 ≥ baseline (퇴보 없음 우선)
- 체크포인트 로드 후 재평가 수치 재현

---

## 📌 Phase 7. Fusion Training (ASF)

### 🎯 Goal
Gaze 주입된 video student로 ASF fusion 재학습

### TODO
- [ ] **타겟 파일**: `TelME/MELD/fusion.py` → `scripts/train_fusion.py`
- [ ] video student ckpt 경로를 `student_video_gaze.bin`으로 교체
- [ ] ASF 구조 **변경 없음** (text-leading)
  ```python
  fused = ASF(text_hidden, audio_hidden, video_hidden_with_gaze)
  ```
- [ ] hyperparam은 TelME 원본 유지
- [ ] 최종 ckpt: `models/checkpoints/total_fusion_gaze.bin`
- [ ] **3-seed 평균** 기록 (seed ∈ {42, 7, 2024})

### 완료 기준
- `results/ours.json`에 3-seed 평균 + std 기록

---

## 📌 Phase 8. Evaluation

### TODO
- [ ] **Metrics** (`utils/metrics.py`)
  - [ ] Accuracy
  - [ ] Weighted-F1 ← primary
  - [ ] Macro-F1
  - [ ] Class-wise F1 (7 emotions)
- [ ] **Experiments**
  - [ ] (A) Baseline TelME
  - [ ] (B) + Gaze (ours)
  - [ ] (C) Random gaze (sanity: gaze_vec ~ N(0,1))
  - [ ] (D) Zero gaze (sanity: gaze_vec = 0)
- [ ] **Ablation** (fix best fusion from Phase 5)
  - [ ] only `p_face`
  - [ ] only `p_out`
  - [ ] only `switch_rate`
  - [ ] only `entropy`
  - [ ] full 6-dim
- [ ] **Significance test**: 3-seed 결과에 대한 paired t-test
- [ ] 결과 표 → `results/table_main.md`

### 완료 기준
- 표가 README에 포함되고, (B) vs (A)의 Weighted-F1 차이와 p-value 명시

---

## 📌 Phase 9. Analysis

### TODO
- [ ] **Quantitative** (`erc/analysis_quant.ipynb`)
  - [ ] confusion matrix (A) vs (B)
  - [ ] class-wise F1 delta bar chart
  - [ ] per-emotion gaze feature 평균 (box plot)
- [ ] **Qualitative** (`erc/analysis_qual.ipynb`)
  - [ ] (A)가 틀리고 (B)가 맞춘 20 samples 수집
  - [ ] 각 샘플의 원본 클립 경로 + gaze_vec + 예측 비교 표
  - [ ] 사례별 3줄 코멘터리
- [ ] 에러 패턴 문서화 → `results/error_analysis.md`

---

## 📌 Phase 10. Visualization

### TODO
- [ ] 프레임 위 gaze point overlay 영상 (`erc/viz_gaze_overlay.py`)
  - cv2.circle + cv2.arrowedLine
- [ ] Emotion별 gaze feature 분포 plot (`erc/viz_feature_dist.py`)
- [ ] Baseline vs Ours 비교 bar chart (`erc/viz_compare.py`)
- [ ] 발표용 그림 → `results/figures/`

---

## 📌 Phase 11. Optional Extension

### TODO
- [ ] Independent **4th modality** gaze branch
  - GazeEncoder (Transformer over frame-level gaze features)
- [ ] ASF를 4-modality로 확장 (text-leading 유지)
- [ ] Gaze-aware **edge weighting** (graph 기반 TelME 변형 시)
- [ ] 비교 실험: (B) 3-mod+gaze-injection vs (E) 4-mod

---

## ⚠️ Critical Risks & Mitigation

| 리스크 | 영향 | 대응 |
|---|---|---|
| 얼굴 검출 실패 | gaze_vec 무효 | zero vector + `is_valid` 마스크 + 학습 시 down-weight |
| Gaze 노이즈 | 학습 불안정 | LayerNorm + feature z-score + λ sweep |
| 화자–얼굴 mismatch | 잘못된 signal | speaker diarization 교차검증 (후속) |
| Sharingan ↔ MELD domain gap | 일반화 저하 | 사전 검증: 100 샘플 수동 품질 평가 |
| MELD 라벨 불균형 | Macro-F1 저하 | class-weighted CE (옵션) |
| 재현성 | 결과 흔들림 | seed 고정 + `torch.use_deterministic_algorithms(True)` + 3-seed 평균 |

---

## 📊 Key Research Questions

- **Q1.** Gaze 신호가 MERC 성능을 향상시키는가? (Weighted-F1 기준, p<0.05)
- **Q2.** 어떤 감정 클래스가 가장 이득을 보는가? (class-wise F1 delta)
- **Q3.** Gaze는 유효한 **interaction signal**인가? (random/zero gaze 대비 유의차)

---

## ✅ Execution Order (Quick Runbook)

```bash
# Phase 0
bash scripts/setup_env.sh

# Phase 1
bash scripts/run_phase1_baseline.sh

# Phase 2-3
python preprocess/extract_clips.py   --split train
python preprocess/sample_frames.py   --split train --fps 6 --max-frames 32
python preprocess/detect_faces.py    --split train
python gaze/sharingan_infer.py       --split train --resume
python features/gaze/build_features.py --split train
# (dev/test 동일)

# Phase 5-7
python scripts/train_student_video_gaze.py --config configs/phase5_video_gaze.yaml
python scripts/train_fusion.py             --config configs/phase7_fusion.yaml

# Phase 8
python scripts/eval_all.py --out results/
```

---

## 🗂️ Deliverables Checklist
- [ ] `results/baseline.json`, `results/ours.json`
- [ ] `results/table_main.md`, `results/error_analysis.md`
- [ ] `models/checkpoints/*.bin` (4종)
- [ ] `features/gaze/{train,dev,test}.pkl` + `scaler.pkl`
- [ ] `configs/*.yaml`, `scripts/*.sh`
- [ ] `README.md` 실험 재현 절차
