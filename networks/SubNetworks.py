import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import backbone
from utils import multi_view_projector


class BEVNet(nn.Module):
    def __init__(self):
        super(BEVNet, self).__init__()
        block = backbone.BasicBlock

        # ---- Encoder ----
        self.enc1 = self._make_layer(block, 192, 64, num_blocks=3, stride=2)  # → [B, 64, 256, 256]
        self.enc2 = self._make_layer(block, 64, 128, num_blocks=3, stride=2)  # → [B, 128, 128, 128]
        self.enc3 = self._make_layer(block, 128, 256, num_blocks=4, stride=2)  # → [B, 256, 64, 64]

        self.l1_sga = backbone.SemanticGuidedAttention(bev_ch=64, rv_ch=128)
        self.l2_sga = backbone.SemanticGuidedAttention(bev_ch=128, rv_ch=256)

        # ---- Decoder (Feature Pyramid) ----
        self.dec1 = backbone.BasicConv2d(64 + 128 + 256, 128, kernel_size=3, padding=1)
        self.dec2 = backbone.BasicConv2d(128, 32, kernel_size=3, padding=1)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=2, dilation=1):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, bev_feat, rv_feat_shallow_as_3d, rv_feat_deep_as_3d, des_coord_t0):
        """
        bev_feat:               [B, 192, 512, 512]  누적 BEV feature (T*pointnet_ch)
        rv_feat_shallow_as_3d:  [B, 128, N, 1]      RV Encoder L1 → 3D 복원
        rv_feat_deep_as_3d:     [B, 256, N, 1]      RV Encoder L2 → 3D 복원
        des_coord_t0:           [B, 1, N, 3, 1]     t0 프레임 데카르트 좌표 (BEV 투영용)
        """
        # ---- RV 3D features → BEV 2D 투영 ----
        rv_shallow_bev = multi_view_projector.project_to_bev(rv_feat_shallow_as_3d, des_coord_t0)  # [B, 128, 512, 512]
        rv_deep_bev = multi_view_projector.project_to_bev(rv_feat_deep_as_3d, des_coord_t0)  # [B, 256, 512, 512]

        # ---- Encoder L1 + Cross-view Fusion ----
        e1 = self.enc1(bev_feat)  # [B, 64, 256, 256]
        rv_s = F.max_pool2d(rv_shallow_bev, kernel_size=2, stride=2)  # [B, 128, 256, 256]
        e1 = self.l1_sga(e1, rv_s)  # 가이드 어텐션 적용 완료 -> [B, 64, 256, 256]

        # ---- Encoder L2 + Cross-view Fusion ----
        e2 = self.enc2(e1)  # [B, 128, 128, 128]
        rv_d = F.max_pool2d(rv_deep_bev, kernel_size=4, stride=4)  # [B, 256, 128, 128]
        e2 = self.l2_sga(e2, rv_d)  # 가이드 어텐션 적용 완료 -> [B, 128, 128, 128]

        # ---- Encoder L3 (BEV only) ----
        e3 = self.enc3(e2)  # [B, 256, 64, 64]

        # ---- Decoder: Feature Pyramid ----
        target_size = e1.shape[2:]  # (256, 256)
        e2_up = F.interpolate(e2, size=target_size, mode="bilinear", align_corners=True)
        e3_up = F.interpolate(e3, size=target_size, mode="bilinear", align_corners=True)

        dec = torch.cat([e1, e2_up, e3_up], dim=1)  # [B, 448, 256, 256]
        dec = F.dropout2d(dec, p=0.3, training=self.training)
        dec = self.dec1(dec)  # [B, 128, 256, 256]
        dec = self.dec2(dec)  # [B, 32, 256, 256]

        return dec


class RVNet(nn.Module):
    def __init__(self):
        super(RVNet, self).__init__()
        block = backbone.BasicBlock

        # ---- Encoder ----
        self.enc1 = self._make_layer(block, 64, 128, num_blocks=2, stride=2)  # → [B, 128, 32, 1024]
        self.enc2 = self._make_layer(block, 128, 256, num_blocks=3, stride=2)  # → [B, 256, 16, 512]

        # ---- Decoder (Feature Pyramid) ----
        self.dec1 = backbone.BasicConv2d(64 + 128 + 256, 128, kernel_size=3, padding=1)
        self.dec2 = backbone.BasicConv2d(128, 64, kernel_size=3, padding=1)

        # ---- Movable Segmentation Head ----
        self.movable_head = backbone.PredBranch(64, 3)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=2, dilation=1):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, rv_feat_t0):
        """
        input:  rv_feat_t0       [B, 64, 64, 2048]
        output: rv_feat_shallow  [B, 128, 32, 1024]   Encoder L1 중간 feature
                rv_feat_deep     [B, 256, 16, 512]     Encoder L2 중간 feature
                movable_logit_2d [B, 3, 64, 2048]      2D Movable Segmentation 결과
        """
        _, _, H, W = rv_feat_t0.shape

        # ---- Encoder ----
        enc1 = self.enc1(rv_feat_t0)  # [B, 128, 32, 1024]
        enc2 = self.enc2(enc1)  # [B, 256, 16, 512]

        # ---- Decoder: Feature Pyramid ----
        enc1_up = F.interpolate(enc1, size=(H, W), mode="bilinear", align_corners=True)
        enc2_up = F.interpolate(enc2, size=(H, W), mode="bilinear", align_corners=True)

        dec = torch.cat([rv_feat_t0, enc1_up, enc2_up], dim=1)  # [B, 448, 64, 2048]
        dec = F.dropout2d(dec, p=0.3, training=self.training)
        dec = self.dec1(dec)  # [B, 128, 64, 2048]
        dec = self.dec2(dec)  # [B, 64, 64, 2048]

        # ---- Movable Prediction ----
        movable_logit_2d = self.movable_head(dec)  # [B, 3, 64, 2048]

        return enc1, enc2, movable_logit_2d
