#!/bin/bash
# LoRaMOS Visualization - Apollo

SEQ_ID=0
FRAME_ID=1580
CHECKPOINT="logs/Exp36/checkpoints/best_80.pth"

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_Apollo/sequences/"

python LoRaMOS_viz_file.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --config config/apollo-mos.yaml \
    --mode test \
    --seq_id ${SEQ_ID} \
    --frame_id ${FRAME_ID} \
    --checkpoint ${CHECKPOINT}
