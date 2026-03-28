#!/bin/bash
# FarMOS Training

MODE="new"          # "new" = from scratch, "keep" = resume
RESUME_EXP="Exp37"  # used when MODE="keep"
TRAIN_CONFIG="config/train.yaml"

# ============================================================
if [ "${MODE}" = "new" ]; then
    LAST=$(ls -d logs/Exp[0-9][0-9] 2>/dev/null | sort -V | tail -1 | grep -o '[0-9]\+')
    if [ -z "${LAST}" ]; then NEXT=1; else NEXT=$((10#${LAST} + 1)); fi
    EXP_ID=$(printf "Exp%02d" ${NEXT})
    LOG_DIR="logs/${EXP_ID}/"
    RESUME_FLAG=""
    echo "=== New experiment: ${EXP_ID} ==="
elif [ "${MODE}" = "keep" ]; then
    EXP_ID=${RESUME_EXP}
    LOG_DIR="logs/${EXP_ID}/"
    RESUME_FLAG="--resume"
    echo "=== Resuming experiment: ${EXP_ID} ==="
fi

clear && python FarMOS_train.py \
    --train_config ${TRAIN_CONFIG} \
    --log_dir ${LOG_DIR} \
    --wandb_name ${EXP_ID} \
    ${RESUME_FLAG}
