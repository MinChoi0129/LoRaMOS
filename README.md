# FarMOS

LiDAR 포인트 클라우드 기반 Moving Object Segmentation (MOS).
BEV + RV dual-branch 구조와 Deformable Attention으로 원거리 이동 객체까지 탐지.

---

## Environment

```bash
conda create -n farmos python=3.9 -y
conda activate farmos

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### MinkowskiEngine (Sparse Convolution)

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

# thrust 헤더 추가 (CUDA 12 호환)
sed -i '1i #include <thrust/execution_policy.h>' src/convolution_kernel.cuh
sed -i '1i #include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_gpu.cu
sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' src/spmm.cu
sed -i '1i #include <thrust/execution_policy.h>' src/3rdparty/concurrent_unordered_map.cuh

# setup.py의 ext_modules에 NVTX_DISABLE 추가 (nvtx3 헤더 충돌 방지)
# define_macros=[('NVTX_DISABLE', None)],
# extra_compile_args: cxx/nvcc 모두 '-DNVTX_DISABLE' 추가

# 시스템 GCC 11로 빌드 (conda GCC 14는 pybind11 호환 문제)
CUDA_HOME=/usr/local/cuda-12.8 \
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu \
python setup.py install

cd ..
```

빌드 확인:

```bash
python -c "import MinkowskiEngine as ME; print(f'MinkowskiEngine {ME.__version__} OK')"
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

## Dataset

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

## Training

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
RESUME_EXP="Exp18"
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
| NUM_TEMPORAL_FRAMES | 8 |
| BEV Grid (H, W, D) | 512 x 512 x 30 |
| BEV Range | X: [-50, 50], Y: [-50, 50], Z: [-4, 2] |
| RV Grid (H, W, D) | 64 x 2048 x 50 |
| RV Range | Phi: [-180, 180]°, Theta: [-25, 3]°, R: [2, 50]m |

### Checkpointing

학습 중 자동 저장되는 체크포인트:

| 파일 | 기준 |
|------|------|
| `latest.pth` | 매 epoch |
| `best_*.pth` | best overall moving IoU |
| `best_0_10m_*.pth` | best 0-10m moving IoU |
| `best_10_20m_*.pth` | best 10-20m moving IoU |
| `best_20_30m_*.pth` | best 20-30m moving IoU |
| `best_30_40m_*.pth` | best 30-40m moving IoU |
| `best_40_50m_*.pth` | best 40-50m moving IoU |

---

## Validation (Prediction 저장)

`scripts/val.sh`에서 `EXP_ID`, `MODE` 설정 후:

```bash
bash scripts/val.sh
```

- `MODE="val"`: sequence 08
- `MODE="test"`: sequences 11-21
- 결과: `logs/ExpXX/predictions/sequences/XX/predictions/*.label`

---

## Evaluation (IoU 계산)

`scripts/eval.sh`에서 `EXP_ID`, `SEQUENCES` 설정 후:

```bash
bash scripts/eval.sh
```

- 전체 IoU + 거리별 (10m 단위) IoU 출력
- 단일: `SEQUENCES="8"` / 복수: `SEQUENCES="8 9 10"`

---

## Speed Benchmark

```bash
bash scripts/speed.sh
```

---

## Visualization

`scripts/visualize.sh`에서 `EXP_ID`, `FRAME_ID` 설정 후:

```bash
bash scripts/visualize.sh
```

`infer()`가 반환하는 중간 피처들 (`visualization` key)이 자동으로 `images/` 디렉토리에 저장된다.

