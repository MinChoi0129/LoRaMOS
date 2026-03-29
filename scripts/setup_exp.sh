#!/bin/bash
# Setup Exp36 directory structure for first-time users.
# Creates logs/Exp36/ with code snapshot and reassembles split checkpoint.

set -e

EXP_ID="Exp36"
LOG_DIR="logs/${EXP_ID}"
CKPT_SRC="best_80.pth"
CKPT_DST="${LOG_DIR}/checkpoints/best_80.pth"

if [ -d "${LOG_DIR}" ]; then
    echo "[skip] ${LOG_DIR} already exists."
    exit 0
fi

# Create experiment directory and run code snapshot
echo "[1/3] Creating ${LOG_DIR} and snapshotting code..."
python -c "from core.builder import snapshot_code; snapshot_code('${LOG_DIR}')"
mkdir -p "${LOG_DIR}/checkpoints"

# Reassemble split checkpoint if needed
if [ ! -f "${CKPT_SRC}" ]; then
    if ls ${CKPT_SRC}.part* 1>/dev/null 2>&1; then
        echo "[2/3] Reassembling ${CKPT_SRC} from split parts..."
        cat ${CKPT_SRC}.part* > "${CKPT_SRC}"
    else
        echo "[2/3] ${CKPT_SRC} and split parts not found, skipping."
    fi
else
    echo "[2/3] ${CKPT_SRC} already exists."
fi

# Copy checkpoint
if [ -f "${CKPT_SRC}" ]; then
    echo "[3/3] Copying ${CKPT_SRC} -> ${CKPT_DST}"
    cp "${CKPT_SRC}" "${CKPT_DST}"
else
    echo "[3/3] ${CKPT_SRC} not found, skipping."
fi

echo "Done. ${LOG_DIR} is ready."
