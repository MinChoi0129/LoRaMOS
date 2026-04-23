import os
import random
import numpy as np
import cv2
from scipy.spatial import Delaunay

from datasets.config import MAX_POINTS


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
        noise = np.random.normal(self.noise_mean, self.noise_std, size=(points.shape[0], 3))
        points[:, :3] += noise

        for axis in range(3):
            shift = random.uniform(self.shift_range[axis][0], self.shift_range[axis][1])
            points[:, axis] += shift

        scale = random.uniform(self.size_range[0], self.size_range[1])
        points[:, :3] *= scale

        if self.flip_x and random.random() < 0.5:
            points[:, 0] *= -1
        if self.flip_y and random.random() < 0.5:
            points[:, 1] *= -1

        theta_deg = random.uniform(self.theta_range[0], self.theta_range[1])
        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        points[:, :2] = points[:, :2] @ rotation_matrix

        return points


ROAD_LABEL = 40


def _in_range(v, r):
    return (v >= r[0]) & (v < r[1])


def _in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def _compute_box_3d(center, size, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    l, w, h = size
    x = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners = np.dot(R, np.vstack([x, y, z]))
    corners[0] += center[0]
    corners[1] += center[1]
    corners[2] += center[2]
    return corners.T


def _rotate_along_z(pcds, theta_deg):
    rot = cv2.getRotationMatrix2D((0, 0), theta_deg, 1.0)[:, :2].T
    pcds[:, :2] = pcds[:, :2].dot(rot)
    return pcds


class SequenceCopyPaste:
    def __init__(self, object_dir, paste_max_obj_num=3):
        self.object_dir = object_dir
        self.sub_dirs = ('car', 'truck', 'other-vehicle', 'person',
                         'bicyclist', 'motorcyclist', 'bicycle', 'motorcycle')
        self.velo_range = {
            'car': (-15, 15), 'truck': (-15, 15), 'other-vehicle': (-15, 15),
            'person': (-3, 3), 'bicyclist': (-8, 8), 'motorcyclist': (-8, 8),
            'bicycle': (-8, 8), 'motorcycle': (-8, 8),
        }
        self.paste_max_obj_num = paste_max_obj_num

        self.obj_files = {}
        for cat in self.sub_dirs:
            cat_dir = os.path.join(object_dir, cat)
            files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)
                     if f.endswith('.npz') and not f.startswith('08_')]
            self.obj_files[cat] = files
            print(f'[CopyPaste] {cat}: {len(files)}')

    def _make_sequential_obj(self, fname, seq_num):
        npz = np.load(fname)
        pcds = npz['pcds']
        cat = str(npz['cate'])
        bbox_center, bbox_size, bbox_yaw = npz['center'], npz['size'] * 1.05, float(npz['yaw'])
        bbox_corners = _compute_box_3d(bbox_center, bbox_size, bbox_yaw)

        velo = random.uniform(*self.velo_range[cat])
        vx = -velo * np.sin(bbox_yaw)
        vy = velo * np.cos(bbox_yaw)

        obj_list = []
        for t in range(seq_num):
            p = pcds.copy()
            p[:, 0] -= vx * t * 0.1
            p[:, 1] -= vy * t * 0.1
            p[:, :3] += np.random.normal(0, 0.001, size=(p.shape[0], 3))
            c = bbox_corners.copy()
            c[:, 0] -= vx * t * 0.1
            c[:, 1] -= vy * t * 0.1
            obj_list.append((p, c))

        return obj_list, np.abs(velo)

    def _get_fov(self, pcds_obj):
        x, y, z = pcds_obj[:, 0], pcds_obj[:, 1], pcds_obj[:, 2]
        d = np.sqrt(x**2 + y**2 + z**2) + 1e-12
        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)
        return (phi.min(), phi.max()), (theta.min(), theta.max())

    def _check_occlusion(self, pcds, raw_labels, phi_fov, theta_fov):
        x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
        d = np.sqrt(x**2 + y**2 + z**2) + 1e-12
        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)
        fov_mask = _in_range(phi, phi_fov) & _in_range(theta, theta_fov)
        in_fov_labels = raw_labels[fov_mask]
        obj_mask = _in_range(in_fov_labels, (10, 33)) | _in_range(in_fov_labels, (252, 260))
        return obj_mask.sum() < 3, fov_mask

    def _valid_position(self, pcds, raw_labels, pcds_obj):
        phi_fov, theta_fov = self._get_fov(pcds_obj)
        if abs(phi_fov[1] - phi_fov[0]) < 1 and abs(theta_fov[1] - theta_fov[0]) < 1:
            return self._check_occlusion(pcds, raw_labels, phi_fov, theta_fov)
        return False, None

    def _paste_single(self, point_clouds, label_list, movable_label_list, raw_labels):
        T = len(point_clouds)
        cat = random.choice(self.sub_dirs)
        fname = random.choice(self.obj_files[cat])
        obj_list, velo = self._make_sequential_obj(fname, T)

        if velo >= 1:
            motion_label = 2
        elif velo < 0.3:
            motion_label = 1
        else:
            motion_label = 0

        if len(obj_list[0][0]) < 10:
            return point_clouds, label_list, movable_label_list

        # Prevent exceeding MAX_POINTS
        max_current = max(pc.shape[0] for pc in point_clouds)
        if max_current + len(obj_list[0][0]) > MAX_POINTS:
            return point_clouds, label_list, movable_label_list

        t0 = T - 1
        road_mask = (raw_labels[t0] == ROAD_LABEL)
        road_pts = point_clouds[t0][road_mask]

        theta_list = np.arange(0, 360, 18).tolist()
        random.shuffle(theta_list)

        for theta in theta_list:
            obj_aug = [(_rotate_along_z(p.copy(), theta), _rotate_along_z(c.copy(), theta))
                       for p, c in obj_list]

            try:
                road_in_box = _in_hull(road_pts[:, :2], obj_aug[0][1][:4, :2])
            except Exception:
                continue
            local_road = road_pts[road_in_box]
            if local_road.shape[0] <= 5:
                continue
            road_height = float(local_road[:, 2].mean())
            for t in range(T):
                obj_aug[t][0][:, 2] += road_height - obj_aug[t][0][:, 2].min()

            valid_results = [self._valid_position(point_clouds[t], raw_labels[t], obj_aug[t][0])
                             for t in range(T)]
            if not all(v[0] for v in valid_results):
                continue

            for t in range(T):
                _, fov_mask = valid_results[t]
                keep = ~fov_mask
                obj_pts = obj_aug[t][0]
                n_obj = obj_pts.shape[0]

                point_clouds[t] = np.concatenate([point_clouds[t][keep], obj_pts], axis=0)
                label_list[t] = np.concatenate([
                    label_list[t][keep],
                    np.full(n_obj, motion_label, dtype=label_list[t].dtype)
                ])
                movable_label_list[t] = np.concatenate([
                    movable_label_list[t][keep],
                    np.full(n_obj, 2, dtype=movable_label_list[t].dtype)
                ])
                raw_labels[t] = np.concatenate([
                    raw_labels[t][keep],
                    np.full(n_obj, 30, dtype=raw_labels[t].dtype)
                ])
            break

        return point_clouds, label_list, movable_label_list

    def __call__(self, point_clouds, label_list, movable_label_list, raw_labels):
        paste_num = random.randint(0, self.paste_max_obj_num)
        for _ in range(paste_num):
            point_clouds, label_list, movable_label_list = self._paste_single(
                point_clouds, label_list, movable_label_list, raw_labels
            )
        return point_clouds, label_list, movable_label_list
