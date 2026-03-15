"""포인트 클라우드 I/O, 변환, 양자화, 피쳐/레이블 생성 유틸리티."""

import numpy as np


# ============================================================
# Calibration & Pose
# ============================================================


def parse_calibration(filename):
    """KITTI calib.txt 파싱 -> 4x4 변환 행렬 dict 반환"""
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
    """KITTI poses.txt -> LiDAR 좌표계 기준 4x4 pose 행렬 리스트 반환"""
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
# Point Transform & Label Remap
# ============================================================


def transform_points(points, transform_matrix):
    """4x4 pose 변환 행렬을 포인트 클라우드에 적용 (intensity 보존)"""
    result = points.copy()
    homogeneous = result[:, :4].T
    homogeneous[-1] = 1.0
    transformed = transform_matrix @ homogeneous
    result[:, :3] = transformed[:3].T
    return result


def relabel(raw_labels, label_map):
    """원본 semantic label -> learning_map에 따라 재매핑"""
    remapped = np.zeros(raw_labels.shape[0], dtype=raw_labels.dtype)
    for source_label, target_label in label_map.items():
        remapped[raw_labels == source_label] = target_label
    return remapped


# ============================================================
# Quantization
# ============================================================


def quantize_cartesian(points_xyzi, range_x, range_y, range_z, grid_size):
    """3D 좌표 -> BEV 격자 인덱스로 양자화. Returns: [N, 3] — [col(x), row(y), depth(z)]"""
    height, width, depth = grid_size

    x = points_xyzi[:, 0].copy()
    y = points_xyzi[:, 1].copy()
    z = points_xyzi[:, 2].copy()

    dx = (range_x[1] - range_x[0]) / width
    dy = (range_y[1] - range_y[0]) / height
    dz = (range_z[1] - range_z[0]) / depth

    x_quan = (x - range_x[0]) / dx
    y_quan = (y - range_y[0]) / dy
    z_quan = (z - range_z[0]) / dz

    return np.stack((x_quan, y_quan, z_quan), axis=-1)


def quantize_spherical(points_xyzi, phi_range, theta_range, r_range, grid_size):
    """3D 좌표 -> 구면 격자 인덱스로 양자화. Returns: [N, 3] — [col(phi), row(theta), depth(r)]"""
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

    return np.stack((phi_quan, theta_quan, r_quan), axis=-1)


# ============================================================
# Feature & Label Generation
# ============================================================


def make_point_features(points_xyzi, cartesian_coords):
    """7채널 포인트 피쳐: [x, y, z, intensity, distance, diff_x, diff_y]"""
    x = points_xyzi[:, 0].copy()
    y = points_xyzi[:, 1].copy()
    z = points_xyzi[:, 2].copy()
    intensity = points_xyzi[:, 3].copy()
    distance = np.sqrt(x**2 + y**2 + z**2) + 1e-12
    diff_x = cartesian_coords[:, 0] - np.floor(cartesian_coords[:, 0])
    diff_y = cartesian_coords[:, 1] - np.floor(cartesian_coords[:, 1])
    return np.stack((x, y, z, intensity, distance, diff_x, diff_y), axis=-1)


def generate_rv_label(spherical_coord_t0, label_t0, rv_height, rv_width):
    """3D 레이블 -> Range View 2D 레이블 (Painter's algorithm: nearest point 우선)"""
    label_2d = np.zeros((rv_height, rv_width), dtype=np.int64)

    phi_idx = np.floor(spherical_coord_t0[:, 0]).astype(np.int64)
    theta_idx = np.floor(spherical_coord_t0[:, 1]).astype(np.int64)
    depth = spherical_coord_t0[:, 2]

    valid = (theta_idx >= 0) & (theta_idx < rv_height) & (phi_idx >= 0) & (phi_idx < rv_width)
    valid_idx = np.where(valid)[0]

    order = np.argsort(-depth[valid_idx])
    sorted_idx = valid_idx[order]

    label_2d[theta_idx[sorted_idx], phi_idx[sorted_idx]] = label_t0[sorted_idx].astype(np.int64)
    return label_2d


def generate_rv_features(points_xyzi_t0, spherical_coord_t0, rv_height, rv_width):
    """3D 포인트 -> Range View 2D 피처맵 [5, H, W] (Painter's algorithm)"""
    rv_features = np.zeros((5, rv_height, rv_width), dtype=np.float32)

    phi_idx = np.floor(spherical_coord_t0[:, 0]).astype(np.int64)
    theta_idx = np.floor(spherical_coord_t0[:, 1]).astype(np.int64)
    depth = spherical_coord_t0[:, 2]

    valid = (theta_idx >= 0) & (theta_idx < rv_height) & (phi_idx >= 0) & (phi_idx < rv_width)
    valid_idx = np.where(valid)[0]

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
