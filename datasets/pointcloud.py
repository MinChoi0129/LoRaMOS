import numpy as np

from datasets.config import BEV_WARP_ALPHA


def parse_calibration(filename):
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


def transform_points(points, transform_matrix):
    result = points.copy()
    homogeneous = result[:, :4].T
    homogeneous[-1] = 1.0
    transformed = transform_matrix @ homogeneous
    result[:, :3] = transformed[:3].T
    return result


def relabel(raw_labels, label_map):
    remapped = np.zeros(raw_labels.shape[0], dtype=raw_labels.dtype)
    for source_label, target_label in label_map.items():
        remapped[raw_labels == source_label] = target_label
    return remapped


def quantize_cartesian(points_xyzi, range_x, range_y, range_z, grid_size, alpha=None):
    # Log-radial warped Cartesian BEV quantization -> [N, 3] (col, row, depth)
    height, width, depth = grid_size
    alpha = alpha if alpha is not None else BEV_WARP_ALPHA
    r_max = max(abs(range_x[0]), abs(range_x[1]))  # 50m

    x = points_xyzi[:, 0].copy()
    y = points_xyzi[:, 1].copy()
    z = points_xyzi[:, 2].copy()

    if alpha != 0:
        r = np.sqrt(x**2 + y**2) + 1e-12
        warp_factor = alpha * np.log(1.0 + r / alpha) / r  # -> 1.0 as r -> 0
        x_warped = x * warp_factor
        y_warped = y * warp_factor
        range_warped = alpha * np.log(1.0 + r_max / alpha)
    else:
        x_warped = x
        y_warped = y
        range_warped = r_max

    x_quan = (x_warped + range_warped) / (2 * range_warped) * width
    y_quan = (y_warped + range_warped) / (2 * range_warped) * height

    dz = (range_z[1] - range_z[0]) / depth
    z_quan = (z - range_z[0]) / dz

    return np.stack((x_quan, y_quan, z_quan), axis=-1)


def quantize_spherical(points_xyzi, phi_range, theta_range, r_range, grid_size):
    # Spherical grid quantization -> [N, 3] (col=phi, row=theta, depth=r)
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


def make_point_features(points_xyzi, cartesian_coords):
    # 7-channel feature: [x, y, z, intensity, distance, diff_x, diff_y]
    x = points_xyzi[:, 0].copy()
    y = points_xyzi[:, 1].copy()
    z = points_xyzi[:, 2].copy()
    intensity = points_xyzi[:, 3].copy()
    distance = np.sqrt(x**2 + y**2 + z**2) + 1e-12
    diff_x = cartesian_coords[:, 0] - np.floor(cartesian_coords[:, 0])
    diff_y = cartesian_coords[:, 1] - np.floor(cartesian_coords[:, 1])
    return np.stack((x, y, z, intensity, distance, diff_x, diff_y), axis=-1)


def generate_rv_label(spherical_coord_t0, label_t0, rv_height, rv_width):
    # 3D label -> RV 2D label (painter's algorithm)
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
    # 3D points -> RV 2D feature map [5, H, W] (painter's algorithm)
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
