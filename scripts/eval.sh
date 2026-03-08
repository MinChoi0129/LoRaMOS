#!/bin/bash

# ============================================================
# FarMOS Evaluation (pred vs GT, range-wise & overall)
# ============================================================

EXP_ID="Exp16"

# Dataset
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"
CONFIG="config/semantic-kitti-mos.yaml"

# Evaluation target sequences
SEQUENCES="8" # 8 9 10 11 <-- 이렇게 주면 리스트로 들어감

# Range settings
RANGE_MAX=50
RANGE_STEP=10

# ============================================================
PRED_DIR="logs/${EXP_ID}/predictions/"

python FarMOS_eval.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --pred_dir "${PRED_DIR}" \
    --config ${CONFIG} \
    --sequences ${SEQUENCES} \
    --range_max ${RANGE_MAX} \
    --range_step ${RANGE_STEP}
