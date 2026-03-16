#!/bin/bash

# ============================================================
# FarMOS Speed Benchmark
# ============================================================

# Dataset
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"
FRAME_ID=4017
CHECKPOINT="/home/work/FarMOS/logs/Exp26/checkpoints/latest.pth"

# ============================================================

python FarMOS_visualization.py --sequence_dir ${SEQUENCE_DIR} --frame_id ${FRAME_ID} --checkpoint ${CHECKPOINT}
