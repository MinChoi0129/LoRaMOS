#!/bin/bash
# FarMOS Speed Benchmark

# ============================================================
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"

python FarMOS_speed.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --warmup_iters 50 \
    --num_iters 2000
