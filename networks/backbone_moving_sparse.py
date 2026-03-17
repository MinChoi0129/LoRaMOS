import torch
import torch.nn as nn
import MinkowskiEngine as ME


class NonUniformVoxelizer:
    """
    거리 적응형 복셀화: log-radial 좌표 워핑으로
    근거리는 고해상도, 원거리는 저해상도 복셀을 생성.

    alpha=5, voxel_size_xy=0.07 기준 실효 복셀 크기:
      r=2m  → ~0.08m  |  r=10m → ~0.13m
      r=30m → ~0.22m  |  r=50m → ~0.29m
    """

    def __init__(self, voxel_size_xy=0.07, voxel_size_z=0.1, alpha=5.0):
        self.voxel_size_xy = voxel_size_xy
        self.voxel_size_z = voxel_size_z
        self.alpha = alpha

    def warp(self, xyz):
        """Log-radial 워핑: 수평 거리(r)에 따라 xy 좌표를 비선형 압축."""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = torch.sqrt(x * x + y * y).clamp(min=1e-6)
        w = self.alpha * torch.log1p(r / self.alpha) / r
        return torch.stack([x * w, y * w, z], dim=1)

    def quantize(self, xyz):
        """워핑된 좌표를 정수 복셀 인덱스로 변환."""
        warped = self.warp(xyz)
        qxy = torch.floor(warped[:, :2] / self.voxel_size_xy).int()
        qz = torch.floor(warped[:, 2:3] / self.voxel_size_z).int()
        return torch.cat([qxy, qz], dim=1)


class SparsePointEncoder(nn.Module):
    """
    PointNetStacker 대체: ME 기반 경량 3D encoder.
    프레임별 독립 처리. 3D 이웃 관계를 학습하여 원거리 sparse 포인트도 맥락 확보.

    입력: 4ch (intensity, distance, diff_x, diff_y)
    출력: 64ch per point (BEV 투영 전 피처)
    """

    def __init__(self, in_ch=4, out_ch=64, voxel_size_xy=0.07, voxel_size_z=0.1, alpha=5.0):
        super().__init__()
        self.out_ch = out_ch
        self.voxelizer = NonUniformVoxelizer(voxel_size_xy, voxel_size_z, alpha)

        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_ch, 32, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(32, 64, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, out_ch, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(out_ch),
            ME.MinkowskiReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, xyzi, valid_mask):
        """
        Args:
            xyzi:       [B*T, 7, N, 1]  (ch: x,y,z, intensity, distance, diff_x, diff_y)
            valid_mask: [B*T, 1, N, 1]

        Returns:
            pcd_feat:   [B*T, 64, N, 1]
        """
        BT, _, N, _ = xyzi.shape
        device = xyzi.device

        xyz = xyzi[:, :3, :, 0].permute(0, 2, 1)           # [BT, N, 3]
        feat = xyzi[:, 3:, :, 0].permute(0, 2, 1)           # [BT, N, 4] (intensity, dist, diff_x, diff_y)
        vmask = valid_mask[:, 0, :, 0]                        # [BT, N]

        # 프레임별 독립 처리: BT를 배치 차원으로 사용
        xyz_flat = xyz.reshape(-1, 3)                         # [BT*N, 3]
        feat_flat = feat.reshape(-1, 4)                       # [BT*N, 4]
        vmask_flat = vmask.reshape(-1)                        # [BT*N]
        batch_idx = torch.arange(BT, device=device).view(BT, 1).expand(BT, N).reshape(-1)

        valid = vmask_flat > 0.5
        v_xyz = xyz_flat[valid]
        v_feat = feat_flat[valid]
        v_batch = batch_idx[valid]

        # 비균일 복셀화
        v_voxel = self.voxelizer.quantize(v_xyz)
        v_coords = torch.cat([v_batch.int().unsqueeze(1), v_voxel], dim=1)

        # TensorField → sparse → encoder → slice
        tensor_field = ME.TensorField(
            features=v_feat,
            coordinates=v_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=device,
        )
        sparse_input = tensor_field.sparse()
        sparse_output = self.net(sparse_input)
        out_field = sparse_output.slice(tensor_field)

        # 유효 포인트 피처를 원래 [BT, 64, N, 1] 형태로 복원
        out_flat = torch.zeros(BT * N, self.out_ch, device=device)
        out_flat[valid] = out_field.F
        output = out_flat.view(BT, N, self.out_ch).permute(0, 2, 1).unsqueeze(-1)

        return output
