#!/bin/bash
# Setup Exp36 directory structure for first-time users.
# Creates logs/Exp36/ with code snapshot and moves best_80.pth into checkpoints/.

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
echo "[1/2] Creating ${LOG_DIR} and snapshotting code..."
python -c "from core.builder import snapshot_code; snapshot_code('${LOG_DIR}')"
mkdir -p "${LOG_DIR}/checkpoints"

# Move checkpoint
if [ -f "${CKPT_SRC}" ]; then
    echo "[2/2] Copying ${CKPT_SRC} -> ${CKPT_DST}"
    cp "${CKPT_SRC}" "${CKPT_DST}"
else
    echo "[2/2] ${CKPT_SRC} not found in project root, skipping."
fi

echo "Done. ${LOG_DIR} is ready."
