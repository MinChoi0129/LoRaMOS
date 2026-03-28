# FarMOS

LiDAR MOS with BEV + RV dual-branch and Deformable Attention for far-range moving object detection.

## Setup

```bash
conda create -n farmos python=3.9 -y && conda activate farmos
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cd deformattn/ && python setup.py build_ext --inplace && cd ..
```

## Dataset

SemanticKITTI format: `sequences/{seq_id}/{calib.txt, poses.txt, velodyne/, labels/}`

- **SemanticKITTI**: Train 00-07,09-10 / Val 08 / Test 11-21
- **Apollo** (cross-dataset): Test 00-04, labels `0`=static `252`=moving

Intensity is auto-normalized to [0,1] at load time (values >1.0 are divided by 255).

Set `sequence_dir` in `config/train.yaml`.

## Training

```bash
bash scripts/train.sh
```

Experiments auto-numbered as `logs/ExpXX/`. Set `MODE="keep"` and `RESUME_EXP` to resume.

## Prediction

```bash
bash scripts/val_kitti.sh    # SemanticKITTI
bash scripts/val_apollo.sh   # Apollo
```

## Evaluation

```bash
bash scripts/eval_kitti.sh   # SemanticKITTI
bash scripts/eval_apollo.sh  # Apollo
```

Outputs overall + range-wise (10m bins) IoU for static/moving.

### Label mapping for other models' predictions

Edit `learning_map` in the dataset config to match the prediction format:

| Model | Static | Moving |
|---|---|---|
| FarMOS / 4DMOS / MF-MOS | 9 | 251 |
| StreamMOS | 1 | 2 |
| Apollo GT | 0 | 252 |

## Speed Benchmark

```bash
bash scripts/speed.sh
```

## Visualization

```bash
bash scripts/visualize.sh
```

Saves intermediate features and predictions as PNGs to `images/`.
