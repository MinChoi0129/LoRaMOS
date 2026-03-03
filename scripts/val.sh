#!/bin/bash

# ============================================================
# FarMOS Validation / Test
# ============================================================

EXP_ID="Exp01"

# Mode: "val" = seq 08 (predict + evaluate)
#        "test" = seq 11-21 (predict only)
MODE="val"

# Dataset
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"
CONFIG="config/semantic-kitti-mos.yaml"

# Inference
BATCH_SIZE=1
NUM_WORKERS=4

# ============================================================
CODE_DIR="logs/${EXP_ID}/code"
CHECKPOINT=$(ls logs/${EXP_ID}/checkpoints/best_*.pth)

cd "${CODE_DIR}"
echo "Checkpoint: ${CHECKPOINT}"

python FarMOS_valid.py \
    --mode ${MODE} \
    --sequence_dir ${SEQUENCE_DIR} \
    --config ${CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --checkpoint "../checkpoints/$(basename ${CHECKPOINT})" \
    --pred_dir "../predictions/"
