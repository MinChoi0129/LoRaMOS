#!/bin/bash

# ============================================================
# FarMOS Training
# ============================================================

# Mode: "new"  = from scratch (auto Exp numbering)
#        "keep" = resume training (set RESUME_EXP below)
MODE="new"

# Training config (hyperparameters)
TRAIN_CONFIG="config/train.yaml"

# ============================================================
# [keep mode] Resume from this experiment
RESUME_EXP="Exp01"
# ============================================================

if [ "${MODE}" = "new" ]; then
    # Auto-increment: find next ExpXX number
    LAST=$(ls -d logs/Exp[0-9][0-9] 2>/dev/null | sort -V | tail -1 | grep -o '[0-9]\+')
    if [ -z "${LAST}" ]; then
        NEXT=1
    else
        NEXT=$((10#${LAST} + 1))
    fi
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
