#!/bin/bash
# LoRaMOS Speed Benchmark

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"

python LoRaMOS_speed.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --warmup_iters 50 \
    --num_iters 2000
