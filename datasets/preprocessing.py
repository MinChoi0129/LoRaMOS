"""파일리스트 구축, 시퀀스 로드, 패딩, 텐서 변환 파이프라인."""

import os
import numpy as np
import torch

from datasets.config import (
    MAX_POINTS,
    NUM_TEMPORAL_FRAMES,
    BEV_RANGE_X,
    BEV_RANGE_Y,
    BEV_RANGE_Z,
    BEV_GRID_SIZE,
    RV_RANGE_PHI,
    RV_RANGE_THETA,
    RV_RANGE_R,
    RV_GRID_SIZE,
)
from datasets.pointcloud import (
    transform_points,
    relabel,
    quantize_cartesian,
    quantize_spherical,
    make_point_features,
    generate_rv_features,
    generate_rv_label,
)


# ============================================================
# File List
# ============================================================


def build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=True):
    """
    단일 시퀀스에 대해 T=NUM_TEMPORAL_FRAMES 프레임씩 묶은 메타 리스트 구축.
    프레임 순서: [t-(T-1), ..., t-1, t0]  (현재 프레임 t0이 마지막 index)
    """
    seq_path = os.path.join(sequence_dir, seq_id)
    velodyne_path = os.path.join(seq_path, "velodyne")
    labels_path = os.path.join(seq_path, "labels")

    num_frames = len(poses)
    flist = []

    for current_idx in range(num_frames):
        current_pose_inv = np.linalg.inv(poses[current_idx])
        meta_list = []

        for offset in range(NUM_TEMPORAL_FRAMES - 1, -1, -1):
            frame_idx = max(0, current_idx - offset)
            file_id = str(frame_idx).zfill(6)

            fname_pcd = os.path.join(velodyne_path, f"{file_id}.bin")
            pose_diff = current_pose_inv @ poses[frame_idx]

            if include_labels:
                fname_label = os.path.join(labels_path, f"{file_id}.label")
                meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
            else:
                meta_list.append((fname_pcd, pose_diff, seq_id, file_id))

        flist.append(meta_list)

    return flist


# ============================================================
# Sequence Loading
# ============================================================


def load_sequence_with_labels(meta_list, learning_map, movable_learning_map=None):
    """T개 프레임의 포인트 클라우드 + 레이블 로드 (Train, Val 공용)"""
    point_clouds = []
    label_list = []
    movable_label_list = [] if movable_learning_map is not None else None

    for frame_meta in meta_list:
        fname_pcd, fname_label, pose_diff, _, _ = frame_meta

        raw_points = np.fromfile(fname_pcd, dtype=np.float32).reshape(-1, 4)
        transformed_points = transform_points(raw_points, pose_diff)
        point_clouds.append(transformed_points)

        raw_label = np.fromfile(fname_label, dtype=np.uint32).reshape(-1)
        semantic_label = raw_label & 0xFFFF
        remapped_label = relabel(semantic_label, learning_map)
        label_list.append(remapped_label)

        if movable_learning_map is not None:
            movable_label = relabel(semantic_label, movable_learning_map)
            movable_label_list.append(movable_label)

    return point_clouds, label_list, movable_label_list


def load_sequence_without_labels(meta_list):
    """T개 프레임의 포인트 클라우드만 로드 (Test 전용)"""
    point_clouds = []

    for frame_meta in meta_list:
        fname_pcd, pose_diff, _, _ = frame_meta
        raw_points = np.fromfile(fname_pcd, dtype=np.float32).reshape(-1, 4)
        transformed_points = transform_points(raw_points, pose_diff)
        point_clouds.append(transformed_points)

    return point_clouds


# ============================================================
# Padding
# ============================================================


def pad_to_max(point_clouds, label_list=None, movable_label_list=None):
    """
    MAX_POINTS까지 더미 패딩 (필터링 없음 -> num_valid = 원본 포인트 수).
    """
    valid_point_counts = []
    has_labels = label_list is not None
    has_movable = movable_label_list is not None

    for t in range(len(point_clouds)):
        num_valid = point_clouds[t].shape[0]
        assert num_valid <= MAX_POINTS, f"포인트 수({num_valid})가 MAX_POINTS({MAX_POINTS})를 초과합니다."
        valid_point_counts.append(num_valid)

        pad_length = MAX_POINTS - num_valid
        if pad_length > 0:
            point_clouds[t] = np.pad(
                point_clouds[t],
                ((0, pad_length), (0, 0)),
                mode="constant",
                constant_values=-1000,
            )
            point_clouds[t][-pad_length:, 2] = -4000

            if has_labels:
                label_list[t] = np.pad(label_list[t], (0, pad_length), mode="constant", constant_values=0)
            if has_movable:
                movable_label_list[t] = np.pad(movable_label_list[t], (0, pad_length), mode="constant", constant_values=0)

    if has_labels:
        return point_clouds, label_list, movable_label_list, valid_point_counts
    return point_clouds, valid_point_counts


# ============================================================
# Tensor Building
# ============================================================


def build_tensors(all_points, augmentor=None):
    """
    T개 프레임 -> 양자화 + 7채널 피쳐 -> 텐서 변환.
    Returns: xyzi [T,7,N,1], des_coord [T,N,3,1], sph_coord [T,N,3,1],
             spherical_coords_raw [T*N,3], rv_input [5,H,W]
    """
    all_points_concat = np.concatenate(all_points, axis=0)

    if augmentor is not None:
        all_points_concat = augmentor(all_points_concat)

    points_xyzi = all_points_concat[:, :4]

    cartesian_coords = quantize_cartesian(points_xyzi, BEV_RANGE_X, BEV_RANGE_Y, BEV_RANGE_Z, BEV_GRID_SIZE)
    spherical_coords = quantize_spherical(points_xyzi, RV_RANGE_PHI, RV_RANGE_THETA, RV_RANGE_R, RV_GRID_SIZE)
    point_features = make_point_features(points_xyzi, cartesian_coords)

    xyzi = torch.FloatTensor(point_features.astype(np.float32))
    xyzi = xyzi.view(NUM_TEMPORAL_FRAMES, MAX_POINTS, 7, 1)
    xyzi = xyzi.permute(0, 2, 1, 3).contiguous()

    des_coord = torch.FloatTensor(cartesian_coords.astype(np.float32))
    des_coord = des_coord.view(NUM_TEMPORAL_FRAMES, MAX_POINTS, 3, 1)

    sph_coord = torch.FloatTensor(spherical_coords.astype(np.float32))
    sph_coord = sph_coord.view(NUM_TEMPORAL_FRAMES, MAX_POINTS, 3, 1)

    points_xyzi_t0 = points_xyzi[-MAX_POINTS:]
    spherical_coords_t0 = spherical_coords[-MAX_POINTS:]
    rv_input = generate_rv_features(points_xyzi_t0, spherical_coords_t0, RV_GRID_SIZE[0], RV_GRID_SIZE[1])
    rv_input = torch.FloatTensor(rv_input)

    return xyzi, des_coord, sph_coord, spherical_coords, rv_input


def build_label_tensors(label_list, spherical_coords, movable_label_list):
    """3D 포인트 레이블 + 2D RV 레이블 텐서 생성."""
    current_moving_label_3d = torch.LongTensor(label_list[-1].astype(np.int64))

    spherical_coords_t0 = spherical_coords[-MAX_POINTS:]
    current_movable_label_2d = generate_rv_label(
        spherical_coords_t0,
        movable_label_list[-1],
        RV_GRID_SIZE[0],
        RV_GRID_SIZE[1],
    )
    current_movable_label_2d = torch.LongTensor(current_movable_label_2d)

    return current_moving_label_3d, current_movable_label_2d
