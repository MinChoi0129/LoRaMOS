import torch
from networks import backbone_moving


# Projectors (좌표 flip 걱정 없이)
def project_with_time_stack(input_3d, stacked_coord, proj_H, proj_W, mode="max"):
    """
    Args:
        pcd_feat: [B*T, C, N, 1] - PointNet Feature
        stacked_coord: [B, T, N, 3, 1] - 데이터로더에서 Quantize된 좌표
        proj_H, proj_W: Quantize 당시 기준 해상도여야만 함. assert!
    """
    B, T, N, _, _ = stacked_coord.shape
    C = input_3d.shape[1]

    feat = input_3d.squeeze(-1).view(B, T, C, N)

    # 데이터로더 구조: [x_quan, y_quan, z_quan] 순서
    proj_x = stacked_coord[:, :, :, 0, 0].floor().long()  # [B, T, N]
    proj_y = stacked_coord[:, :, :, 1, 0].floor().long()  # [B, T, N]

    # 마스킹 (격자 범위를 벗어나는 노이즈 차단)
    valid_mask = (proj_x >= 0) & (proj_x < proj_W) & (proj_y >= 0) & (proj_y < proj_H)

    proj_x = torch.clamp(proj_x, 0, proj_W - 1)
    proj_y = torch.clamp(proj_y, 0, proj_H - 1)

    spatial_idx = proj_y * proj_W + proj_x

    trash_idx = proj_H * proj_W
    spatial_idx[~valid_mask] = trash_idx
    spatial_idx = spatial_idx.unsqueeze(2).expand(-1, -1, C, -1)

    target_shape = (B, T, C, proj_H * proj_W + 1)
    projected = torch.zeros(target_shape, device=feat.device, dtype=feat.dtype)

    reduce_op = {"max": "amax", "min": "amin", "mean": "mean"}[mode]
    projected.scatter_reduce_(dim=3, index=spatial_idx, src=feat, reduce=reduce_op, include_self=False)

    projected = projected[:, :, :, :-1]
    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    projected = projected.view(B, T * C, proj_H, proj_W)

    return projected


# Projectors (좌표 flip 걱정 없이)
def project_without_time_stack(input_3d, t0_coord, proj_H, proj_W, mode="max"):
    """
    Args:
        input_3d: [B, C, N, 1] - 현재 프레임 PointNet Feature
        t0_coord: [B, N, 3, 1] - 데이터로더에서 Quantize된 구면 좌표
        proj_H, proj_W: Quantize 당시 기준 해상도여야만 함. assert!
    """
    B, N, _, _ = t0_coord.shape
    C = input_3d.shape[1]

    feat = input_3d.squeeze(-1)

    # 데이터로더 구조: [theta_quan(H), phi_quan(W), r_quan] 순서
    proj_y = t0_coord[:, :, 0, 0].floor().long()  # theta -> H
    proj_x = t0_coord[:, :, 1, 0].floor().long()  # phi -> W

    valid_mask = (proj_x >= 0) & (proj_x < proj_W) & (proj_y >= 0) & (proj_y < proj_H)

    proj_x = torch.clamp(proj_x, 0, proj_W - 1)
    proj_y = torch.clamp(proj_y, 0, proj_H - 1)

    spatial_idx = proj_y * proj_W + proj_x
    trash_idx = proj_H * proj_W
    spatial_idx[~valid_mask] = trash_idx
    spatial_idx = spatial_idx.unsqueeze(1).expand(-1, C, -1)

    target_shape = (B, C, proj_H * proj_W + 1)
    projected = torch.zeros(target_shape, device=feat.device, dtype=feat.dtype)

    reduce_op = {"max": "amax", "min": "amin", "mean": "mean"}[mode]
    projected.scatter_reduce_(dim=2, index=spatial_idx, src=feat, reduce=reduce_op, include_self=False)

    projected = projected[:, :, :-1]
    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    projected = projected.view(B, C, proj_H, proj_W)

    return projected


# Unprojectos by resolution.
# Note: BEV는 그냥 좌표 받아도 되는데 RV unproject시에는 flip된 좌표 받아야함
# Note: __call__ 시 T축 없어야함. --> # [B, N, 2, 1]
unprojectors = {
    "full": backbone_moving.BilinearSample(scale_rate=(1.0, 1.0)),
    "half": backbone_moving.BilinearSample(scale_rate=(0.5, 0.5)),
    "quarter": backbone_moving.BilinearSample(scale_rate=(0.25, 0.25)),
    "eighth": backbone_moving.BilinearSample(scale_rate=(0.125, 0.125)),
}
