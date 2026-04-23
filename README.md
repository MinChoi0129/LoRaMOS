# LoRa-MOS

**Range-Robust LiDAR Moving Object Segmentation via Logarithmic Spatial Representation and Range-Balanced Learning**

Official implementation of LoRa-MOS. The model targets robust moving-object segmentation at long range by combining a log-radial Cartesian BEV representation with range-balanced training.

---

## Setup

```bash
conda create -n loramos python=3.9 -y && conda activate loramos
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cd deformattn/ && python setup.py build_ext --inplace && cd ..
```

## Dataset

All datasets follow the SemanticKITTI layout: `sequences/{seq_id}/{calib.txt, poses.txt, velodyne/, labels/}`.

| Dataset | Required | Description |
|---|---|---|
| [SemanticKITTI](https://) | Yes | Train 00-07, 09-10 / Val 08 / Test 11-21 |
| [Object Bank](https://) | For training | Copy-paste augmentation objects. Set path in `datasets/config.py` |
| [Apollo](https://) | Optional | Cross-dataset eval (Test 00-04). Inference uses a SemanticKITTI-trained model directly — no fine-tuning |

Intensity is auto-normalized to `[0, 1]` at load time (values > 1.0 are divided by 255).

Set `sequence_dir` in `config/train.yaml`.

## Pretrained Checkpoint Setup

Run once after cloning:

```bash
bash scripts/setup_exp.sh
```

Places the weights at the expected location and cleans up the setup artifacts when done.

## Training

```bash
bash scripts/train.sh
```

New experiments are auto-numbered as `logs/ExpXX/`. To resume, set `MODE="keep"` and `RESUME_EXP` inside `scripts/train.sh`.

## Prediction

```bash
bash scripts/val.sh              # SemanticKITTI
bash scripts/apollo/val.sh       # Apollo
```

## Evaluation

```bash
bash scripts/eval.sh             # SemanticKITTI
bash scripts/apollo/eval.sh      # Apollo
```

Reports overall and range-binned (10 m bins) IoU for the static / moving classes.

## Speed Benchmark

```bash
bash scripts/speed.sh
```

## Visualization

Two modes are available:

### File-based (PNG / NPY dump)

```bash
bash scripts/viz_file.sh              # SemanticKITTI
bash scripts/apollo/viz_file.sh       # Apollo
```

Saves intermediate features and predictions to `visualization/2d/` (PNG) and `visualization/3d/` (NPY). To view the NPY point clouds, see https://github.com/MinChoi0129/MOS_visualizer.

### Live streaming via Rerun

```bash
bash scripts/viz_rerun.sh
```

Streams predictions and the corresponding `image_2` camera frame to a Rerun web viewer over gRPC. The script prints a URL of the form `http://localhost:9090/?url=...`; open it in a browser on the host. Forward both ports (`9090` for the web viewer, `9876` for gRPC) from your container or remote host.

Requires `rerun-sdk` (already listed in `requirements.txt`).
