import torch
import torch.nn.functional as F
from datasets.config import BEV_GRID_SIZE, RV_GRID_SIZE

_VIEW_SIZES = {
    "bev": (BEV_GRID_SIZE[0], BEV_GRID_SIZE[1]),
    "rv": (RV_GRID_SIZE[0], RV_GRID_SIZE[1]),
}


def project(feat, coord, view="bev", mode="max"):
    # 3D point features -> 2D grid; coord is [col, row]
    # feat [B*T, C, N, 1], coord [B, T, N, 2] or [B, N, 2] -> [B, T*C, H, W]
    H, W = _VIEW_SIZES[view]

    if coord.dim() == 3:
        coord = coord.unsqueeze(1)

    B, T, N, _ = coord.shape
    C = feat.shape[1]
    feat = feat.squeeze(-1).view(B, T, C, N)

    proj_col = coord[:, :, :, 0].floor().long()
    proj_row = coord[:, :, :, 1].floor().long()
    valid_mask = (proj_col >= 0) & (proj_col < W) & (proj_row >= 0) & (proj_row < H)

    proj_col = torch.clamp(proj_col, 0, W - 1)
    proj_row = torch.clamp(proj_row, 0, H - 1)

    spatial_idx = proj_row * W + proj_col
    trash_idx = H * W
    spatial_idx[~valid_mask] = trash_idx
    spatial_idx = spatial_idx.unsqueeze(2).expand(-1, -1, C, -1)

    target_shape = (B, T, C, H * W + 1)
    projected = torch.zeros(target_shape, device=feat.device, dtype=feat.dtype)

    reduce_op = {"max": "amax", "min": "amin", "mean": "mean"}[mode]
    projected.scatter_reduce_(dim=3, index=spatial_idx, src=feat, reduce=reduce_op, include_self=False)

    projected = projected[:, :, :, :-1]
    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    return projected.view(B, T * C, H, W)


def unproject(grid_feat, coord, scale=1.0):
    # 2D grid features -> 3D points via bilinear sampling
    # grid_feat [B, C, H, W], coord [B, N, 2] -> [B, C, N, 1]
    H, W = grid_feat.shape[2], grid_feat.shape[3]
    grid_x = (2 * coord[:, :, 0] * scale / (W - 1)) - 1
    grid_y = (2 * coord[:, :, 1] * scale / (H - 1)) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
    return F.grid_sample(grid_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
