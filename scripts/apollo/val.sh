#!/bin/bash
# FarMOS Prediction - Apollo

EXP_ID="Exp36"

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_Apollo/sequences/"
CONFIG="config/apollo-mos.yaml"
CHECKPOINT=$(ls logs/${EXP_ID}/checkpoints/best_*.pth | head -1)

cd "logs/${EXP_ID}/code"
echo "Checkpoint: ${CHECKPOINT}"

python FarMOS_valid.py \
    --mode test \
    --sequence_dir ${SEQUENCE_DIR} \
    --config ${CONFIG} \
    --batch_size 1 \
    --num_workers 4 \
    --checkpoint "../checkpoints/$(basename ${CHECKPOINT})" \
    --pred_dir "../predictions_apollo/"
