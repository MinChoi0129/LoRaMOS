import torch
from networks import backbone_BEV


def project_to_bev(pcd_feat, des_coord_stacked, proj_H=512, proj_W=512, mode="max"):
    """
    Args:
        pcd_feat: [B*T, C, N, 1] - PointNet Feature
        des_coord_stacked: [B, T, N, 3, 1] - 데이터로더에서 Quantize된 데카르트 좌표
    """
    B, T, N, _, _ = des_coord_stacked.shape
    C = pcd_feat.shape[1]

    feat = pcd_feat.squeeze(-1).view(B, T, C, N)

    # 데이터로더 구조: [x_quan, y_quan, z_quan] 순서
    proj_x = des_coord_stacked[:, :, :, 0, 0].floor().long()  # [B, T, N]
    proj_y = des_coord_stacked[:, :, :, 1, 0].floor().long()  # [B, T, N]

    # 마스킹 (격자 범위를 벗어나는 노이즈 차단)
    valid_mask = (proj_x >= 0) & (proj_x < proj_W) & (proj_y >= 0) & (proj_y < proj_H)

    proj_x = torch.clamp(proj_x, 0, proj_W - 1)
    proj_y = torch.clamp(proj_y, 0, proj_H - 1)

    spatial_idx = proj_y * proj_W + proj_x

    trash_idx = proj_H * proj_W
    spatial_idx[~valid_mask] = trash_idx
    spatial_idx = spatial_idx.unsqueeze(2).expand(-1, -1, C, -1)

    target_shape = (B, T, C, proj_H * proj_W + 1)
    target_bev = torch.zeros(target_shape, device=feat.device, dtype=feat.dtype)

    reduce_op = {"max": "amax", "min": "amin", "mean": "mean"}[mode]
    target_bev.scatter_reduce_(dim=3, index=spatial_idx, src=feat, reduce=reduce_op, include_self=False)

    target_bev = target_bev[:, :, :, :-1]
    target_bev = torch.nan_to_num(target_bev, nan=0.0, posinf=0.0, neginf=0.0)
    target_bev = target_bev.view(B, T * C, proj_H, proj_W)

    return target_bev


def project_to_rv(pcd_feat, sph_coord_t0, proj_H=64, proj_W=2048, mode="max"):
    """
    Args:
        pcd_feat: [B, C, N, 1] - 현재 프레임 PointNet Feature
        sph_coord_t0: [B, N, 3, 1] - 데이터로더에서 Quantize된 구면 좌표
    """
    B, N, _, _ = sph_coord_t0.shape
    C = pcd_feat.shape[1]

    feat = pcd_feat.squeeze(-1)

    # 데이터로더 구조: [theta_quan(H), phi_quan(W), r_quan] 순서
    proj_y = sph_coord_t0[:, :, 0, 0].floor().long()  # theta -> H
    proj_x = sph_coord_t0[:, :, 1, 0].floor().long()  # phi -> W

    valid_mask = (proj_x >= 0) & (proj_x < proj_W) & (proj_y >= 0) & (proj_y < proj_H)

    proj_x = torch.clamp(proj_x, 0, proj_W - 1)
    proj_y = torch.clamp(proj_y, 0, proj_H - 1)

    spatial_idx = proj_y * proj_W + proj_x
    trash_idx = proj_H * proj_W
    spatial_idx[~valid_mask] = trash_idx
    spatial_idx = spatial_idx.unsqueeze(1).expand(-1, C, -1)

    target_shape = (B, C, proj_H * proj_W + 1)
    target_rv = torch.zeros(target_shape, device=feat.device, dtype=feat.dtype)

    reduce_op = {"max": "amax", "min": "amin", "mean": "mean"}[mode]
    target_rv.scatter_reduce_(dim=2, index=spatial_idx, src=feat, reduce=reduce_op, include_self=False)

    target_rv = target_rv[:, :, :-1]
    target_rv = torch.nan_to_num(target_rv, nan=0.0, posinf=0.0, neginf=0.0)
    target_rv = target_rv.view(B, C, proj_H, proj_W)

    return target_rv


unprojectors = {
    "full": backbone_BEV.BilinearSample(scale_rate=(1.0, 1.0)),
    "half": backbone_BEV.BilinearSample(scale_rate=(0.5, 0.5)),
    "quarter": backbone_BEV.BilinearSample(scale_rate=(0.25, 0.25)),
    "eighth": backbone_BEV.BilinearSample(scale_rate=(0.125, 0.125)),
}
