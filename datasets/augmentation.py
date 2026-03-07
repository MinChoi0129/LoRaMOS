import numpy as np
import random


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
        포인트 클라우드 증강: noise -> shift -> scale -> flip -> rotation
        Input/Output: [N, C] (x, y, z, intensity, ...)
        """
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
