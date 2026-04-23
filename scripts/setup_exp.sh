#!/bin/bash
# One-shot setup for Exp36. Reassembles the split checkpoint, places it under
# logs/Exp36/, cleans up the split parts, then self-destructs.

set -e

EXP_ID="Exp36"
LOG_DIR="logs/${EXP_ID}"
CKPT_SRC="best_80.pth"
CKPT_DST="${LOG_DIR}/checkpoints/${CKPT_SRC}"

if [ -d "${LOG_DIR}" ]; then
    echo "[skip] ${LOG_DIR} already exists."
    exit 0
fi

echo "[1/3] Creating ${LOG_DIR} and snapshotting code..."
python -c "from core.builder import snapshot_code; snapshot_code('${LOG_DIR}')"
mkdir -p "${LOG_DIR}/checkpoints"

echo "[2/3] Preparing ${CKPT_SRC}..."
if [ ! -f "${CKPT_SRC}" ] && ls ${CKPT_SRC}.part* 1>/dev/null 2>&1; then
    cat ${CKPT_SRC}.part* > "${CKPT_SRC}"
fi
rm -f ${CKPT_SRC}.part*

if [ ! -f "${CKPT_SRC}" ]; then
    echo "  Error: ${CKPT_SRC} not available (no file and no split parts)."
    exit 1
fi

echo "[3/3] Moving ${CKPT_SRC} -> ${CKPT_DST}"
mv "${CKPT_SRC}" "${CKPT_DST}"

echo "Done. ${LOG_DIR} is ready. Removing setup script."
rm -f "$0"
