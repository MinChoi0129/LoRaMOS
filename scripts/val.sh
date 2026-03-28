#!/bin/bash
# FarMOS Prediction - SemanticKITTI

EXP_ID="Exp36"
MODE="val"  # "val" = seq 08, "test" = seq 11-21

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"
CONFIG="config/semantic-kitti-mos.yaml"
CHECKPOINT=$(ls logs/${EXP_ID}/checkpoints/best_*.pth | head -1)

cd "logs/${EXP_ID}/code"
echo "Checkpoint: ${CHECKPOINT}"

python FarMOS_valid.py \
    --mode ${MODE} \
    --sequence_dir ${SEQUENCE_DIR} \
    --config ${CONFIG} \
    --batch_size 1 \
    --num_workers 4 \
    --checkpoint "../checkpoints/$(basename ${CHECKPOINT})" \
    --pred_dir "../predictions_kitti/"
