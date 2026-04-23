#!/bin/bash
# LoRaMOS Visualization

FRAME_ID=4017
CHECKPOINT="logs/Exp36/checkpoints/best_80.pth"

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"

python LoRaMOS_visualization.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --config config/semantic-kitti-mos.yaml \
    --frame_id ${FRAME_ID} \
    --checkpoint ${CHECKPOINT}
