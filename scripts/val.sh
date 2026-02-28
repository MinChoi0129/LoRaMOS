#!/bin/bash

# ============================================================
# FarMOS Validation / Test
# ============================================================

# Mode: "val" = seq 08 (predict + evaluate)
#        "test" = seq 11-21 (predict only)
MODE="val"

# Dataset
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"
CONFIG="config/semantic-kitti-mos.yaml"

# Inference
BATCH_SIZE=1
NUM_WORKERS=4

# Checkpoint & Output
CHECKPOINT="logs/Exp01/checkpoints/best_68.pth"
PRED_DIR="logs/Exp01/predictions/"

# ============================================================

python FarMOS_valid.py \
    --mode ${MODE} \
    --sequence_dir ${SEQUENCE_DIR} \
    --config ${CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --checkpoint ${CHECKPOINT} \
    --pred_dir ${PRED_DIR}
