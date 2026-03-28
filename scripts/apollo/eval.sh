#!/bin/bash
# FarMOS Evaluation - Apollo

EXP_ID="Exp36"
SEQUENCES="0 1 2 3 4"
PRED_DIR="logs/${EXP_ID}/predictions_apollo/"

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_Apollo/sequences/"
CONFIG="config/apollo-mos.yaml"

python FarMOS_eval.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --pred_dir "${PRED_DIR}" \
    --config ${CONFIG} \
    --sequences ${SEQUENCES} \
    --range_max 50 \
    --range_step 10
