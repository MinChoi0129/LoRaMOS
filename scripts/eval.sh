#!/bin/bash
# LoRaMOS Evaluation - SemanticKITTI

EXP_ID="Exp36"
SEQUENCES="8"
PRED_DIR="logs/${EXP_ID}/predictions_kitti/"

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"
CONFIG="config/semantic-kitti-mos.yaml"

python LoRaMOS_eval.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --pred_dir "${PRED_DIR}" \
    --config ${CONFIG} \
    --sequences ${SEQUENCES} \
    --range_max 50 \
    --range_step 10
