#!/bin/bash

# ============================================================
# FarMOS Speed Benchmark
# ============================================================

# Dataset
SEQUENCE_DIR="/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences/"

# Benchmark
WARMUP_ITERS=50
NUM_ITERS=2000

# ============================================================

python FarMOS_speed.py \
    --sequence_dir ${SEQUENCE_DIR} \
    --warmup_iters ${WARMUP_ITERS} \
    --num_iters ${NUM_ITERS}
