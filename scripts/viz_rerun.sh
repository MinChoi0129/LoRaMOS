#!/bin/bash
# LoRaMOS Rerun Visualization (live streaming from inside Docker)
#
# Requires: pip install rerun-sdk
# Open viewer from outside: http://<host>:${WEB_PORT}
# Forward both WEB_PORT and GRPC_PORT from the container (e.g. -p 9090:9090 -p 9876:9876)

CHECKPOINT="logs/Exp36/checkpoints/best_80.pth"
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"
SEQUENCE=8

# ============================================================
CONFIG="config/semantic-kitti-mos.yaml"
WEB_PORT=9090
GRPC_PORT=9876

python LoRaMOS_viz_rerun.py \
    --sequence_dir "${SEQUENCE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --sequence ${SEQUENCE} \
    --config ${CONFIG} \
    --web_port ${WEB_PORT} \
    --grpc_port ${GRPC_PORT}
