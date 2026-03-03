import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import random


# ============================================================
# Configuration Constants
# ============================================================

DATASET_CONFIG_PATH = "config/semantic-kitti-mos.yaml"
STATIC_FRAMES_TXT_PATH = "config/train_split_dynamic_pointnumber.txt"

MAX_POINTS = 160000
NUM_TEMPORAL_FRAMES = 3

# BEV (Cartesian) quantization parameters
BEV_RANGE_X = (-50.0, 50.0)
BEV_RANGE_Y = (-50.0, 50.0)
BEV_RANGE_Z = (-4.0, 2.0)
BEV_GRID_SIZE = (512, 512, 30)

# RV (Spherical) quantization parameters
RV_RANGE_PHI = (-180.0, 180.0)  # degrees
RV_RANGE_THETA = (-25.0, 3.0)  # degrees
RV_RANGE_R = (2.0, 50.0)
RV_GRID_SIZE = (64, 2048, 30)


# ============================================================
# Calibration & Pose Utilities
# ============================================================


def parse_calibration(filename):
    """KITTI calib.txt 파싱 → 4x4 변환 행렬 dict 반환"""
    calibration = {}
    with open(filename, "r") as f:
        for line in f:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            matrix = np.zeros((4, 4))
            matrix[0, 0:4] = values[0:4]
            matrix[1, 0:4] = values[4:8]
            matrix[2, 0:4] = values[8:12]
            matrix[3, 3] = 1.0
            calibration[key] = matrix
    return calibration


def parse_poses(filename, calibration):
    """KITTI poses.txt → LiDAR 좌표계 기준 4x4 pose 행렬 리스트 반환"""
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    with open(filename, "r") as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(Tr_inv @ pose @ Tr)
    return poses


# ============================================================
# Point Cloud Utilities
# ============================================================


def transform_points(points, transform_matrix):
    """4x4 pose 변환 행렬을 포인트 클라우드에 적용 (intensity 보존)"""
    result = points.copy()
    homogeneous = result[:, :4].T
    homogeneous[-1] = 1.0
    transformed = transform_matrix @ homogeneous
    result[:, :3] = transformed[:3].T
    return result


def filter_points_mask(points, range_x, range_y, range_z):
    """공간 범위 내 유효 포인트의 boolean mask 반환"""
    valid_x = (points[:, 0] >= range_x[0]) & (points[:, 0] < range_x[1])
    valid_y = (points[:, 1] >= range_y[0]) & (points[:, 1] < range_y[1])
    valid_z = (points[:, 2] >= range_z[0]) & (points[:, 2] < range_z[1])
    return valid_x & valid_y & valid_z


def relabel(raw_labels, label_map):
    """원본 semantic label → learning_map에 따라 재매핑 (0: unlabeled, 1: static, 2: moving)"""
    remapped = np.zeros(raw_labels.shape[0], dtype=raw_labels.dtype)
    for source_label, target_label in label_map.items():
        remapped[raw_labels == source_label] = target_label
    return remapped


# ============================================================
# Quantization
# ============================================================


def quantize_cartesian(points_xyzi, range_x, range_y, range_z, grid_size):
    """
    3D 좌표 → BEV 격자 인덱스로 양자화
    Returns: [N, 3] = (x_quan, y_quan, z_quan)
    """
    x = points_xyzi[:, 0].copy()
    y = points_xyzi[:, 1].copy()
    z = points_xyzi[:, 2].copy()

    dx = (range_x[1] - range_x[0]) / grid_size[0]
    dy = (range_y[1] - range_y[0]) / grid_size[1]
    dz = (range_z[1] - range_z[0]) / grid_size[2]

    x_quan = (x - range_x[0]) / dx
    y_quan = (y - range_y[0]) / dy
    z_quan = (z - range_z[0]) / dz

    return np.stack((x_quan, y_quan, z_quan), axis=-1)


def quantize_spherical(points_xyzi, phi_range, theta_range, r_range, grid_size):
    """
    3D 좌표 → 구면 격자 인덱스로 양자화
    Returns: [N, 3] = (theta_quan, phi_quan, r_quan)
    """
    height, width, depth = grid_size

    phi_rad_min = np.radians(phi_range[0])
    phi_rad_max = np.radians(phi_range[1])
    theta_rad_min = np.radians(theta_range[0])
    theta_rad_max = np.radians(theta_range[1])

    dphi = (phi_rad_max - phi_rad_min) / width
    dtheta = (theta_rad_max - theta_rad_min) / height
    dr = (r_range[1] - r_range[0]) / depth

    x, y, z = points_xyzi[:, 0], points_xyzi[:, 1], points_xyzi[:, 2]
    dist = np.sqrt(x**2 + y**2 + z**2) + 1e-12

    phi = phi_rad_max - np.arctan2(x, y)
    phi_quan = phi / dphi

    theta = theta_rad_max - np.arcsin(z / dist)
    theta_quan = theta / dtheta

    r_quan = (dist - r_range[0]) / dr

    return np.stack((theta_quan, phi_quan, r_quan), axis=-1)


# ============================================================
# Feature & Label Generation
# ============================================================


def make_point_features(points_xyzi, cartesian_coords):
    """
    7채널 포인트 피쳐 생성: [x, y, z, intensity, distance, diff_x, diff_y]
    diff_x/diff_y: BEV 격자 내 서브픽셀 오프셋
    """
    x = points_xyzi[:, 0].copy()
    y = points_xyzi[:, 1].copy()
    z = points_xyzi[:, 2].copy()
    intensity = points_xyzi[:, 3].copy()

    distance = np.sqrt(x**2 + y**2 + z**2) + 1e-12

    diff_x = cartesian_coords[:, 0] - np.floor(cartesian_coords[:, 0])
    diff_y = cartesian_coords[:, 1] - np.floor(cartesian_coords[:, 1])

    return np.stack((x, y, z, intensity, distance, diff_x, diff_y), axis=-1)


def generate_rv_label(spherical_coord_t0, label_t0, rv_height=64, rv_width=2048):
    """
    현재 프레임의 3D 레이블 → Range View 2D 레이블로 scatter
    중복 셀은 최소 거리(nearest point) 기준으로 결정 (Painter's algorithm)
    """
    label_2d = np.zeros((rv_height, rv_width), dtype=np.int64)

    theta_idx = np.floor(spherical_coord_t0[:, 0]).astype(np.int64)
    phi_idx = np.floor(spherical_coord_t0[:, 1]).astype(np.int64)
    depth = spherical_coord_t0[:, 2]  # r_quan (range)

    valid = (theta_idx >= 0) & (theta_idx < rv_height) & (phi_idx >= 0) & (phi_idx < rv_width)
    valid_idx = np.where(valid)[0]

    # 먼 포인트 먼저, 가까운 포인트가 덮어씀 → 최종 레이블은 가장 가까운 포인트의 것
    order = np.argsort(-depth[valid_idx])
    sorted_idx = valid_idx[order]

    label_2d[theta_idx[sorted_idx], phi_idx[sorted_idx]] = label_t0[sorted_idx].astype(np.int64)

    return label_2d


def generate_rv_features(points_xyzi_t0, spherical_coord_t0, rv_height=64, rv_width=2048):
    """
    현재 프레임의 3D 포인트 → Range View 2D 피처맵 (Painter's algorithm)
    각 픽셀은 가장 가까운 포인트의 [x, y, z, intensity, range] 값 사용
    (generate_rv_label과 동일한 nearest-point 기준)
    Returns: [5, rv_height, rv_width]
    """
    rv_features = np.zeros((5, rv_height, rv_width), dtype=np.float32)

    theta_idx = np.floor(spherical_coord_t0[:, 0]).astype(np.int64)
    phi_idx = np.floor(spherical_coord_t0[:, 1]).astype(np.int64)
    depth = spherical_coord_t0[:, 2]

    valid = (theta_idx >= 0) & (theta_idx < rv_height) & (phi_idx >= 0) & (phi_idx < rv_width)
    valid_idx = np.where(valid)[0]

    # Painter's algorithm: 먼 포인트 먼저, 가까운 포인트가 덮어씀
    order = np.argsort(-depth[valid_idx])
    sorted_idx = valid_idx[order]

    th = theta_idx[sorted_idx]
    ph = phi_idx[sorted_idx]

    x, y, z = points_xyzi_t0[:, 0], points_xyzi_t0[:, 1], points_xyzi_t0[:, 2]
    rv_features[0, th, ph] = x[sorted_idx]
    rv_features[1, th, ph] = y[sorted_idx]
    rv_features[2, th, ph] = z[sorted_idx]
    rv_features[3, th, ph] = points_xyzi_t0[sorted_idx, 3]
    rv_features[4, th, ph] = np.sqrt(x[sorted_idx] ** 2 + y[sorted_idx] ** 2 + z[sorted_idx] ** 2) + 1e-12

    return rv_features


# ============================================================
# Data Augmentation
# ============================================================


class DataAugment:
    def __init__(
        self,
        noise_mean=0,
        noise_std=0.01,
        theta_range=(-180, 180),
        shift_range=((0, 0), (0, 0), (0, 0)),
        size_range=(0.95, 1.05),
        flip_x=True,
        flip_y=True,
    ):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.theta_range = theta_range
        self.shift_range = shift_range
        self.size_range = size_range
        self.flip_x = flip_x
        self.flip_y = flip_y

    def __call__(self, points):
        """
        포인트 클라우드 증강: noise → shift → scale → flip → rotation
        Input/Output: [N, C] (x, y, z, intensity, ...)
        """
        # Random Gaussian noise on XYZ
        noise = np.random.normal(self.noise_mean, self.noise_std, size=(points.shape[0], 3))
        points[:, :3] += noise

        # Random shift per axis
        for axis in range(3):
            shift = random.uniform(self.shift_range[axis][0], self.shift_range[axis][1])
            points[:, axis] += shift

        # Random scale
        scale = random.uniform(self.size_range[0], self.size_range[1])
        points[:, :3] *= scale

        # Random flip on XY plane (RV에서는 비물리적 occlusion 패턴 생성하므로 기본 OFF)
        if self.flip_x and random.random() < 0.5:
            points[:, 0] *= -1
        if self.flip_y and random.random() < 0.5:
            points[:, 1] *= -1

        # Random rotation on XY plane
        theta_deg = random.uniform(self.theta_range[0], self.theta_range[1])
        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        points[:, :2] = points[:, :2] @ rotation_matrix

        return points


# ============================================================
# Common helpers for building file lists and processing
# ============================================================


def build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=True):
    """
    단일 시퀀스에 대해 T=NUM_TEMPORAL_FRAMES 프레임씩 묶은 메타 리스트 구축.
    프레임 순서: [t-(T-1), ..., t-1, t0]  (현재 프레임 t0이 마지막 index)
    경계 처리: max(0, idx) 로 clamp (시퀀스 시작부에서 과거 프레임 부족 시 반복 사용)
    """
    seq_path = os.path.join(sequence_dir, seq_id)
    velodyne_path = os.path.join(seq_path, "velodyne")
    labels_path = os.path.join(seq_path, "labels")

    num_frames = len(poses)
    flist = []

    for current_idx in range(num_frames):
        current_pose_inv = np.linalg.inv(poses[current_idx])
        meta_list = []

        # offset: (T-1) → 0 순서 = 과거 → 현재
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


def filter_and_pad(point_clouds, label_list=None, movable_label_list=None):
    """
    범위 필터링 → MAX_POINTS까지 더미 패딩.
    Returns:
        point_clouds: 각 프레임 [MAX_POINTS, 4]
        label_list:   각 프레임 [MAX_POINTS] (입력 시에만)
        movable_label_list: 각 프레임 [MAX_POINTS] (입력 시에만)
        valid_point_counts: 각 프레임의 유효 포인트 수
    """
    valid_point_counts = []
    has_labels = label_list is not None
    has_movable = movable_label_list is not None

    for t in range(len(point_clouds)):
        # 공간 필터링
        valid_mask = filter_points_mask(
            point_clouds[t],
            range_x=BEV_RANGE_X,
            range_y=BEV_RANGE_Y,
            range_z=BEV_RANGE_Z,
        )
        point_clouds[t] = point_clouds[t][valid_mask]
        if has_labels:
            label_list[t] = label_list[t][valid_mask]
        if has_movable:
            movable_label_list[t] = movable_label_list[t][valid_mask]

        num_valid = point_clouds[t].shape[0]
        assert num_valid <= MAX_POINTS, f"필터링 후 포인트 수({num_valid})가 MAX_POINTS({MAX_POINTS})를 초과합니다."
        valid_point_counts.append(num_valid)

        # 더미 패딩
        pad_length = MAX_POINTS - num_valid
        if pad_length > 0:
            point_clouds[t] = np.pad(
                point_clouds[t],
                ((0, pad_length), (0, 0)),
                mode="constant",
                constant_values=-1000,
            )
            point_clouds[t][-pad_length:, 2] = -4000  # z축 극단값으로 BEV/RV 투영 시 자동 제외

            if has_labels:
                label_list[t] = np.pad(
                    label_list[t],
                    (0, pad_length),
                    mode="constant",
                    constant_values=0,  # unlabeled
                )
            if has_movable:
                movable_label_list[t] = np.pad(
                    movable_label_list[t],
                    (0, pad_length),
                    mode="constant",
                    constant_values=0,  # unlabeled
                )

    if has_labels:
        return point_clouds, label_list, movable_label_list, valid_point_counts
    return point_clouds, valid_point_counts


def build_tensors(all_points, augmentor=None):
    """
    T개 프레임 → 양자화 + 7채널 피쳐 생성 → 텐서 변환.
    Args:
        all_points: list of [MAX_POINTS, 4] per frame
        augmentor:  DataAugment instance (Train만 사용)
    Returns:
        xyzi:       [T, 7, N, 1]
        des_coord:  [T, N, 3, 1]
        sph_coord:  [T, N, 3, 1]
    """
    # T개 프레임을 하나로 연결 (증강을 동일하게 적용하기 위함)
    all_points_concat = np.concatenate(all_points, axis=0)  # [T*N, 4]

    # Data augmentation (Train only)
    if augmentor is not None:
        all_points_concat = augmentor(all_points_concat)

    points_xyzi = all_points_concat[:, :4]

    # Cartesian quantization → BEV 좌표
    cartesian_coords = quantize_cartesian(
        points_xyzi,
        range_x=BEV_RANGE_X,
        range_y=BEV_RANGE_Y,
        range_z=BEV_RANGE_Z,
        grid_size=BEV_GRID_SIZE,
    )

    # Spherical quantization → RV 좌표
    spherical_coords = quantize_spherical(
        points_xyzi,
        phi_range=RV_RANGE_PHI,
        theta_range=RV_RANGE_THETA,
        r_range=RV_RANGE_R,
        grid_size=RV_GRID_SIZE,
    )

    # 7채널 포인트 피쳐
    point_features = make_point_features(points_xyzi, cartesian_coords)

    # Numpy → Torch + reshape
    # [T*N, 7] → [T, N, 7, 1] → permute → [T, 7, N, 1]
    xyzi = torch.FloatTensor(point_features.astype(np.float32))
    xyzi = xyzi.view(NUM_TEMPORAL_FRAMES, MAX_POINTS, 7, 1)
    xyzi = xyzi.permute(0, 2, 1, 3).contiguous()

    # [T*N, 3] → [T, N, 3, 1]
    des_coord = torch.FloatTensor(cartesian_coords.astype(np.float32))
    des_coord = des_coord.view(NUM_TEMPORAL_FRAMES, MAX_POINTS, 3, 1)

    sph_coord = torch.FloatTensor(spherical_coords.astype(np.float32))
    sph_coord = sph_coord.view(NUM_TEMPORAL_FRAMES, MAX_POINTS, 3, 1)

    # RV features: t0 프레임의 nearest-point range image [5, 64, 2048]
    points_xyzi_t0 = points_xyzi[-MAX_POINTS:]
    spherical_coords_t0 = spherical_coords[-MAX_POINTS:]
    rv_input = generate_rv_features(
        points_xyzi_t0,
        spherical_coords_t0,
        rv_height=RV_GRID_SIZE[0],
        rv_width=RV_GRID_SIZE[1],
    )
    rv_input = torch.FloatTensor(rv_input)

    return xyzi, des_coord, sph_coord, spherical_coords, rv_input


def build_label_tensors(label_list, spherical_coords, movable_label_list):
    """
    3D 포인트 레이블 + 2D RV 레이블 텐서 생성.
    Args:
        label_list:          list of [MAX_POINTS] per frame (moving labels)
        spherical_coords:    [T*N, 3] 전체 프레임 구면 좌표 (build_tensors에서 반환)
        movable_label_list:  list of [MAX_POINTS] per frame (movable labels)
    Returns:
        current_moving_label_3d:  [N]
        current_movable_label_2d: [64, 2048]
    """
    # 3D 레이블: 현재 프레임(t0 = 마지막 프레임) — moving labels
    current_moving_label_3d = torch.LongTensor(label_list[-1].astype(np.int64))

    # 2D RV 레이블: movable labels 사용
    spherical_coords_t0 = spherical_coords[-MAX_POINTS:]
    current_movable_label_2d = generate_rv_label(
        spherical_coords_t0,
        movable_label_list[-1],
        rv_height=RV_GRID_SIZE[0],
        rv_width=RV_GRID_SIZE[1],
    )
    current_movable_label_2d = torch.LongTensor(current_movable_label_2d)

    return current_moving_label_3d, current_movable_label_2d


# ============================================================
# Dataset: Train
# ============================================================


class DataloadTrain(Dataset):
    def __init__(self, sequence_dir):
        """
        Args:
            sequence_dir: SemanticKITTI sequences 디렉토리 경로
                          (예: "/data/semantic-kitti/dataset/sequences")
        """
        self.sequence_dir = sequence_dir
        self.flist = []

        with open(DATASET_CONFIG_PATH, "r") as f:
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.movable_learning_map = self.task_cfg["movable_learning_map"]
        self.augmentor = DataAugment()

        # 학습 시퀀스 파일 리스트 구축
        train_sequences = [str(s).zfill(2) for s in self.task_cfg["split"]["train"]]
        for seq_id in train_sequences:
            seq_path = os.path.join(sequence_dir, seq_id)
            calib = parse_calibration(os.path.join(seq_path, "calib.txt"))
            poses = parse_poses(os.path.join(seq_path, "poses.txt"), calib)
            self.flist.extend(build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=True))

        print(f"[Train] Before static removal: {len(self.flist)} samples")
        self._remove_static_frames()
        print(f"[Train] After static removal:  {len(self.flist)} samples")

    def _remove_static_frames(self):
        """dynamic point가 존재하는 프레임만 유지 (txt 파일 기반 필터링)"""
        if not os.path.exists(STATIC_FRAMES_TXT_PATH):
            print(f"  Warning: {STATIC_FRAMES_TXT_PATH} not found, skipping static removal.")
            return

        with open(STATIC_FRAMES_TXT_PATH, "r") as f:
            lines = f.readlines()

        keep_set = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            seq_id, file_id, _ = parts
            if seq_id not in keep_set:
                keep_set[seq_id] = set()
            keep_set[seq_id].add(file_id)

        new_flist = []
        for meta_list in self.flist:
            # 현재 프레임(t0)은 리스트의 마지막 항목
            current_seq_id = meta_list[-1][3]
            current_file_id = meta_list[-1][4]
            if current_seq_id in keep_set and current_file_id in keep_set[current_seq_id]:
                new_flist.append(meta_list)

        self.flist = new_flist

    def __getitem__(self, index):
        meta_list = self.flist[index]

        # 1. 포인트 클라우드 & 레이블 로드 (pose 변환 포함)
        point_clouds, label_list, movable_label_list = load_sequence_with_labels(
            meta_list, self.task_cfg["learning_map"], self.movable_learning_map
        )

        # 2. 필터링 & 패딩
        point_clouds, label_list, movable_label_list, _ = filter_and_pad(point_clouds, label_list, movable_label_list)

        # 3. 양자화 + 7채널 피쳐 (augmentation 적용)
        xyzi, des_coord, sph_coord, spherical_coords_raw, rv_input = build_tensors(
            point_clouds, augmentor=self.augmentor
        )

        # 4. 레이블 텐서
        label_3d, label_2d = build_label_tensors(label_list, spherical_coords_raw, movable_label_list)

        return (
            xyzi,  # [T, 7, N, 1]
            des_coord,  # [T, N, 3, 1]
            sph_coord,  # [T, N, 3, 1]
            rv_input,  # [5, 64, 2048]
            label_3d,  # [N]
            label_2d,  # [64, 2048]
        )

    def __len__(self):
        return len(self.flist)


# ============================================================
# Dataset: Validation
# ============================================================


class DataloadVal(Dataset):
    def __init__(self, sequence_dir):
        self.sequence_dir = sequence_dir
        self.flist = []

        with open(DATASET_CONFIG_PATH, "r") as f:
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.movable_learning_map = self.task_cfg["movable_learning_map"]

        val_sequences = [str(s).zfill(2) for s in self.task_cfg["split"]["valid"]]
        for seq_id in val_sequences:
            seq_path = os.path.join(sequence_dir, seq_id)
            calib = parse_calibration(os.path.join(seq_path, "calib.txt"))
            poses = parse_poses(os.path.join(seq_path, "poses.txt"), calib)
            self.flist.extend(build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=True))

        print(f"[Val] Total samples: {len(self.flist)}")

    def __getitem__(self, index):
        meta_list = self.flist[index]

        point_clouds, label_list, movable_label_list = load_sequence_with_labels(
            meta_list, self.task_cfg["learning_map"], self.movable_learning_map
        )
        point_clouds, label_list, movable_label_list, valid_point_counts = filter_and_pad(
            point_clouds, label_list, movable_label_list
        )

        # Val은 augmentation 없음
        xyzi, des_coord, sph_coord, spherical_coords_raw, rv_input = build_tensors(point_clouds)
        moving_label_3d, movable_label_2d = build_label_tensors(label_list, spherical_coords_raw, movable_label_list)

        # 평가용 메타 정보: 현재 프레임의 유효 포인트 수, 시퀀스 ID, 프레임 ID
        num_valid_t0 = valid_point_counts[-1]
        current_seq_id = meta_list[-1][3]
        current_file_id = meta_list[-1][4]

        return (
            xyzi,  # [T, 7, N, 1]
            des_coord,  # [T, N, 3, 1]
            sph_coord,  # [T, N, 3, 1]
            rv_input,  # [5, 64, 2048]
            moving_label_3d,  # [N]
            movable_label_2d,  # [64, 2048]
            num_valid_t0,  # int
            current_seq_id,  # str
            current_file_id,  # str
        )

    def __len__(self):
        return len(self.flist)


# ============================================================
# Dataset: Test
# ============================================================


class DataloadTest(Dataset):
    def __init__(self, sequence_dir, target_sequence):
        """
        Args:
            sequence_dir:    SemanticKITTI sequences 디렉토리 경로
            target_sequence: 테스트할 시퀀스 번호 (예: 11 또는 "11")
        """
        self.sequence_dir = sequence_dir
        self.flist = []

        seq_id = str(target_sequence).zfill(2)
        seq_path = os.path.join(sequence_dir, seq_id)
        calib = parse_calibration(os.path.join(seq_path, "calib.txt"))
        poses = parse_poses(os.path.join(seq_path, "poses.txt"), calib)
        self.flist = build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=False)

        print(f"[Test] Sequence {seq_id}: {len(self.flist)} samples")

    def __getitem__(self, index):
        meta_list = self.flist[index]

        point_clouds = load_sequence_without_labels(meta_list)
        point_clouds, valid_point_counts = filter_and_pad(point_clouds)

        xyzi, des_coord, sph_coord, _, rv_input = build_tensors(point_clouds)

        num_valid_t0 = valid_point_counts[-1]
        current_seq_id = meta_list[-1][2]
        current_file_id = meta_list[-1][3]

        return (
            xyzi,  # [T, 7, N, 1]
            des_coord,  # [T, N, 3, 1]
            sph_coord,  # [T, N, 3, 1]
            rv_input,  # [5, 64, 2048]
            num_valid_t0,  # int
            current_seq_id,  # str
            current_file_id,  # str
        )

    def __len__(self):
        return len(self.flist)
