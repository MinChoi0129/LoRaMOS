import torch
import torch.nn.functional as F
from datasets.config import BEV_GRID_SIZE, RV_GRID_SIZE

# 좌표 규약: 항상 [col(W), row(H)] 순서
# BEV: col=x_quan, row=y_quan | RV: col=phi_quan, row=theta_quan
_VIEW_SIZES = {
    "bev": (BEV_GRID_SIZE[0], BEV_GRID_SIZE[1]),  # (H=512, W=512)
    "rv": (RV_GRID_SIZE[0], RV_GRID_SIZE[1]),      # (H=64, W=2048)
}


def project(feat, coord, view="bev", mode="max"):
    """
    3D 포인트 피처를 2D 격자로 투영.

    Args:
        feat:  [B*T, C, N, 1] — 포인트 피처
        coord: [B, T, N, 2] or [B, N, 2] — 양자화 좌표, [col, row] 순서
               T축이 없으면 단일 프레임으로 처리
        view:  "bev" or "rv" — config에서 해상도 자동 결정
        mode:  "max", "min", "mean"

    Returns:
        [B, T*C, H, W] (T축 있을 때) or [B, C, H, W] (단일 프레임)
    """
    H, W = _VIEW_SIZES[view]

    if coord.dim() == 3:  # [B, N, 2] → 단일 프레임, T=1로 확장
        coord = coord.unsqueeze(1)

    B, T, N, _ = coord.shape
    C = feat.shape[1]

    feat = feat.squeeze(-1).view(B, T, C, N)

    proj_col = coord[:, :, :, 0].floor().long()  # [B, T, N] — col → W
    proj_row = coord[:, :, :, 1].floor().long()  # [B, T, N] — row → H

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
    projected = projected.view(B, T * C, H, W)

    return projected


def unproject(grid_feat, coord, scale=1.0):
    """
    2D 격자 피처를 3D 포인트로 역투영 (bilinear sampling).

    Args:
        grid_feat: [B, C, H, W] — 2D 피처맵
        coord:     [B, N, 2] — 양자화 좌표, [col, row] 순서
        scale:     float — 좌표 스케일 (0.5면 피처맵이 원본의 절반 해상도)

    Returns:
        [B, C, N, 1]
    """
    H = grid_feat.shape[2]
    W = grid_feat.shape[3]

    grid_x = (2 * coord[:, :, 0] * scale / (W - 1)) - 1  # col → x
    grid_y = (2 * coord[:, :, 1] * scale / (H - 1)) - 1  # row → y

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)  # [B, N, 1, 2]
    return F.grid_sample(grid_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
