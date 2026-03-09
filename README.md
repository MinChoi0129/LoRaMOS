# FarMOS (Temporary Title)

LiDAR 포인트 클라우드 기반 Moving Object Segmentation (MOS).
BEV + RV dual-branch 구조와 Deformable Attention으로 원거리 이동 객체까지 탐지.

---

## 1. Environment

### Python & PyTorch

```bash
conda create -n farmos python=3.9 -y
conda activate farmos

# Blackwell Architecture 지원
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

### Deformable Attention CUDA Extension

```bash
cd deformattn/
python setup.py build_ext --inplace
cd ..
```

빌드 확인:
```bash
python -c "from deformattn.modules import MSDeformAttn; print('OK')"
```

---

## 2. Dataset

SemanticKITTI dataset을 사용한다.

### 디렉토리 구조

```
/path/to/sequences/
├── 00/
│   ├── calib.txt
│   ├── poses.txt
│   ├── velodyne/
│   │   ├── 000000.bin
│   │   └── ...
│   └── labels/
│       ├── 000000.label
│       └── ...
├── 01/
├── ...
└── 21/
```

| Split | Sequences |
|-------|-----------|
| Train | 00-07, 09-10 |
| Valid | 08 |
| Test  | 11-21 |

### 경로 설정

`config/train.yaml`에서 `sequence_dir` 수정:

```yaml
sequence_dir: "/path/to/sequences/"
```

---

## 3. Training

### 새 실험 시작

`scripts/train.sh`에서 `MODE="new"` 확인 후:

```bash
bash scripts/train.sh
```

자동으로 `logs/ExpXX/`에 다음 번호로 실험이 생성된다.

### Resume

`scripts/train.sh`에서:

```bash
MODE="keep"
RESUME_EXP="Exp18"   # 이어서 학습할 실험 번호
```

```bash
bash scripts/train.sh
```

### 주요 하이퍼파라미터 (`config/train.yaml`)

| 항목 | 기본값 |
|------|--------|
| batch_size | 4 |
| optimizer | adamw |
| lr | 0.001 |
| weight_decay | 0.01 |
| scheduler | cosine |
| epochs | 100 |

### 데이터 설정 (`datasets/config.py`)

| 항목 | 값 |
|------|-----|
| MAX_POINTS | 160,000 |
| NUM_TEMPORAL_FRAMES | 5 |
| BEV Grid | 512 x 512 x 30 |
| RV Grid | 64 x 2048 x 30 |

---

## 4. Validation (Prediction 저장)

`scripts/val.sh`에서 `EXP_ID`, `MODE` 설정 후:

```bash
bash scripts/val.sh
```

- `MODE="val"`: sequence 08
- `MODE="test"`: sequences 11-21
- 결과: `logs/ExpXX/predictions/sequences/XX/predictions/*.label`

---

## 5. Evaluation (IoU 계산)

`scripts/eval.sh`에서 `EXP_ID`, `SEQUENCES` 설정 후:

```bash
bash scripts/eval.sh
```

- 전체 IoU + 거리별 (10m 단위) IoU 출력
- 단일: `SEQUENCES="8"` / 복수: `SEQUENCES="8 9 10"`

---

## 6. Speed Benchmark

```bash
bash scripts/speed.sh
```

---

## 7. Visualization

`scripts/visualize.sh`에서 `EXP_ID`, `FRAME_ID` 설정 후:

```bash
bash scripts/visualize.sh
```

---

## 8. Weights Download

```
https://
```

---

## Project Structure

```
FarMOS/
├── FarMOS_train.py              # 학습
├── FarMOS_valid.py              # Prediction 저장
├── FarMOS_eval.py               # IoU 평가
├── FarMOS_speed.py              # 속도 벤치마크
├── FarMOS_visualization.py      # 시각화
├── config/
│   ├── train.yaml               # 학습 하이퍼파라미터
│   └── semantic-kitti-mos.yaml  # 클래스 정의, split, learning map
├── networks/
│   ├── MainNetwork.py           # FarMOS 전체 모델
│   ├── SubNetworks.py           # BEVNet, RVNet
│   ├── backbone_BEV.py          # BEV backbone (DeformAttnBottleneck 포함)
│   └── backbone_RV.py           # RV backbone
├── datasets/
│   ├── config.py                # 데이터 상수 (MAX_POINTS, grid size 등)
│   ├── dataloader.py            # Train/Val/Test Dataset
│   ├── preprocessing.py         # 텐서 변환, 패딩
│   ├── pointcloud.py            # 좌표 변환, quantization
│   └── augmentation.py          # 학습 augmentation
├── utils/
│   ├── builder.py               # optimizer, scheduler, dataloader 생성
│   ├── metrics.py               # iouEval
│   ├── logger.py                # Logger, wandb 연동
│   ├── checkpoint.py            # 체크포인트 저장/로드
│   └── projector_unprojector.py # BEV/RV ↔ 3D 변환
├── deformattn/                  # Deformable Attention CUDA extension
│   ├── setup.py
│   ├── src/
│   ├── functions/
│   └── modules/
├── scripts/
│   ├── train.sh
│   ├── val.sh
│   ├── eval.sh
│   ├── speed.sh
│   └── visualize.sh
└── logs/                        # 실험 결과 (자동 생성)
    └── ExpXX/
        ├── checkpoints/
        ├── predictions/
        ├── code_snapshot/
        └── wandb/
```
