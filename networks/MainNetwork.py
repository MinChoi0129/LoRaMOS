import os
import torch
import torch.nn as nn
import yaml
from networks import backbone, loss, SubNetworks
from utils.pretty_print_and_pretty_image import shprint, save_feature_as_img
from utils import multi_view_projector


class FarMOS(nn.Module):
    def __init__(self):
        super(FarMOS, self).__init__()

        self._build_network()
        self._build_loss()
        self._build_utils()

    def _build_network(self):
        self.pointnet_ch = 64
        self.pointnet = backbone.PointNetStacker(7, self.pointnet_ch, pre_bn=True, stack_num=2)
        self.moving_bev_net = SubNetworks.BEVNet()
        self.movable_rv_net = SubNetworks.RVNet()
        self.point_fuse = backbone.CatFusion([self.pointnet_ch, 32], 3)

    def _build_loss(self):
        with open("config/semantic-kitti-mos.yaml", "r") as f:
            task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # Moving loss weights (learning_map 기반)
        moving_content = torch.zeros(3, dtype=torch.float32)
        for cl, freq in task_cfg["content"].items():
            x_cl = task_cfg["learning_map"][cl]
            moving_content[x_cl] += freq
        moving_loss_w = 1 / (moving_content + 0.001)
        moving_loss_w[0] = 0
        self.moving_wce_loss = nn.CrossEntropyLoss(weight=moving_loss_w)

        # Movable loss weights (movable_learning_map 기반)
        movable_content = torch.zeros(3, dtype=torch.float32)
        for cl, freq in task_cfg["content"].items():
            x_cl = task_cfg["movable_learning_map"][cl]
            movable_content[x_cl] += freq
        movable_loss_w = 1 / (movable_content + 0.001)
        movable_loss_w[0] = 0
        self.movable_wce_loss = nn.CrossEntropyLoss(weight=movable_loss_w)

        self.lovasz_loss = loss.lovasz_softmax

    def _build_utils(self):
        self.project_to_bev = multi_view_projector.project_to_bev  # 위에서 재정의한 함수 매핑
        self.project_to_rv = multi_view_projector.project_to_rv
        self.unproject_from_bev = backbone.BilinearSample(scale_rate=(0.5, 0.5))
        self.unproject_from_rv_shallow = backbone.BilinearSample(scale_rate=(0.5, 0.5))
        self.unproject_from_rv_deep = backbone.BilinearSample(scale_rate=(0.25, 0.25))

    def get_loss(self, pred, label, wce_fn):
        """
        어떤 형태의 입력이 들어오든, 강제로 [B, C, N, 1]과 [B, N, 1] 형태의
        유사 이미지(Pseudo-Image) 텐서로 만들어 버립니다.
        """
        B, C = pred.shape[0], pred.shape[1]

        # 1. 4D (pred) 구조로 변환: [B, C, H*W, 1]
        pred = pred.view(B, C, -1).unsqueeze(-1)

        # 2. 3D (label) 구조로 변환: [B, H*W, 1]
        label = label.view(B, -1).unsqueeze(-1)

        # PyTorch의 CrossEntropy는 [B, C, d1, d2] vs [B, d1, d2] 매칭을 완벽히 지원하고,
        # Lovasz Loss는 드디어 자기가 좋아하는 4D 텐서를 보고 안심하며 평탄화를 진행합니다.
        l_wce = wce_fn(pred, label)
        l_lovasz = self.lovasz_loss(pred, label, ignore=0)

        return l_wce + l_lovasz

    def infer(self, xyzi, des_coord, sph_coord):
        """
        xyzi: [B, T, 7, N, 1]
        des_coord: [B, T, N, 3, 1]
        sph_coord: [B, T, N, 3, 1]
        """
        B, T, C_feat, N, _ = xyzi.shape

        # **** [ Step 0: Unprojector(BilinearSample)용 2D 픽셀 좌표 추출 ] ****
        # BEV 좌표: [B, T, N, 3, 1]에서 t0 프레임의 x_quan, y_quan 추출 -> [B, N, 2, 1]
        bev_coords_t0 = des_coord[:, -1, :, :2, :]

        # RV 좌표: sph_coord는 [theta, phi, r] 순서. F.grid_sample은 (X,Y)=(W,H)를 요구하므로 교차 추출
        rv_coords_t0 = torch.stack(
            [sph_coord[:, -1, :, 1, :], sph_coord[:, -1, :, 0, :]],  # W (phi)  # H (theta)
            dim=-2,
        )  # 결과 형태: [B, N, 2, 1]

        # **** [ Step 1: Point Feature 추출 및 시간 분리 ] ****
        # 패딩 포인트 마스킹 (BatchNorm 오염 방지)
        # Channel 4 = distance. 패딩 포인트: ~4243, 유효 포인트: < ~75m
        valid_mask = (xyzi[:, :, 4:5, :, :] < 100.0).float()  # [B, T, 1, N, 1]
        xyzi = xyzi * valid_mask

        # [B, T, 7, N, 1] -> [B*T, 7, N, 1]
        xyzi_flatten = xyzi.view(B * T, C_feat, N, 1)
        pcd_feat = self.pointnet(xyzi_flatten)  # [B*T, 64, N, 1]

        # PointNet 출력에도 마스킹 (패딩 포인트의 BN bias 잔여값 제거)
        valid_mask_flat = valid_mask.view(B * T, 1, N, 1)
        pcd_feat = pcd_feat * valid_mask_flat

        pcd_feat_t0 = pcd_feat.view(B, T, 64, N, 1)[:, -1]  # [B, 64, N, 1]

        # **** [ Step 2: 누적형 Feature->BEV / 현재형 Feature->RV 투영 ] ****
        bev_feat = self.project_to_bev(pcd_feat, des_coord, proj_H=512, proj_W=512, mode="max")
        rv_feat_t0 = self.project_to_rv(pcd_feat_t0, sph_coord, proj_H=64, proj_W=2048, mode="max")

        # **** [ Step 3: Semantic 예측 및 중간-2D-Feature를 3D로 복원 ] ****
        rv_feat_shallow, rv_feat_deep, movable_logit_2d = self.movable_rv_net(rv_feat_t0)

        # 물리 좌표 대신 앞서 슬라이싱한 픽셀 좌표를 삽입
        rv_feat_shallow_as_3d = self.unproject_from_rv_shallow(rv_feat_shallow, rv_coords_t0)
        rv_feat_deep_as_3d = self.unproject_from_rv_deep(rv_feat_deep, rv_coords_t0)

        # **** [ Step 4: BEV Feature와 Semantic 3D Features를 융합 ] ****
        des_coord_t0 = des_coord[:, -1:, :, :, :]  # [B, 1, N, 3, 1]
        moving_feat_2d = self.moving_bev_net(bev_feat, rv_feat_shallow_as_3d, rv_feat_deep_as_3d, des_coord_t0)

        # 물리 좌표 대신 bev 픽셀 좌표를 삽입
        moving_feat_3d = self.unproject_from_bev(moving_feat_2d, bev_coords_t0)
        moving_logit_3d = self.point_fuse(pcd_feat_t0, moving_feat_3d)

        RUN_SAVE_FEATURE = False
        if RUN_SAVE_FEATURE:
            # Cross-view samplers (BEV↔RV)
            sample_bev_full = backbone.BilinearSample(scale_rate=(1.0, 1.0))
            sample_bev_half = backbone.BilinearSample(scale_rate=(0.5, 0.5))
            sample_rv_full = backbone.BilinearSample(scale_rate=(1.0, 1.0))

            # BEV-native → RV pairs
            bev_feat_as_rv = self.project_to_rv(sample_bev_full(bev_feat, bev_coords_t0), sph_coord)
            moving_feat_2d_as_rv = self.project_to_rv(sample_bev_half(moving_feat_2d, bev_coords_t0), sph_coord)

            # RV-native → BEV pairs
            rv_feat_t0_as_bev = self.project_to_bev(sample_rv_full(rv_feat_t0, rv_coords_t0), des_coord_t0)
            rv_feat_shallow_as_bev = self.project_to_bev(rv_feat_shallow_as_3d, des_coord_t0)
            rv_feat_deep_as_bev = self.project_to_bev(rv_feat_deep_as_3d, des_coord_t0)
            movable_logit_2d_as_bev = self.project_to_bev(sample_rv_full(movable_logit_2d, rv_coords_t0), des_coord_t0)

            # Predictions (BEV + RV pairs)
            moving_pred_bev_before = self.project_to_bev(moving_feat_3d, des_coord_t0)
            moving_pred_rv_before = self.project_to_rv(moving_feat_3d, sph_coord)

            moving_pred_bev_after = torch.softmax(self.project_to_bev(moving_logit_3d, des_coord_t0), dim=1).argmax(
                dim=1
            )
            moving_pred_rv_after = torch.softmax(self.project_to_rv(moving_logit_3d, sph_coord), dim=1).argmax(dim=1)

            movable_pred_rv = torch.softmax(movable_logit_2d, dim=1).argmax(dim=1)
            movable_pred_bev = torch.softmax(movable_logit_2d_as_bev, dim=1).argmax(dim=1)

            rv_feat_shallow_sig = torch.sigmoid(rv_feat_shallow)
            rv_feat_deep_sig = torch.sigmoid(rv_feat_deep)
            movable_logit_2d_sig = torch.sigmoid(movable_logit_2d)

            save_feature_as_img(
                [
                    bev_feat,
                    bev_feat_as_rv,
                    rv_feat_t0_as_bev,
                    rv_feat_t0,
                    rv_feat_shallow_as_bev,
                    rv_feat_shallow,
                    rv_feat_deep_as_bev,
                    rv_feat_deep,
                    moving_feat_2d,
                    moving_feat_2d_as_rv,
                    movable_logit_2d_as_bev,
                    movable_logit_2d,
                    moving_pred_bev_before,
                    moving_pred_rv_before,
                    moving_pred_bev_after,
                    moving_pred_rv_after,
                    movable_pred_bev,
                    movable_pred_rv,
                    rv_feat_shallow_sig,
                    rv_feat_deep_sig,
                    movable_logit_2d_sig,
                ],
                [
                    "bev_feat_bev",
                    "bev_feat_rv",
                    "rv_feat_t0_bev",
                    "rv_feat_t0_rv",
                    "rv_feat_shallow_bev",
                    "rv_feat_shallow_rv",
                    "rv_feat_deep_bev",
                    "rv_feat_deep_rv",
                    "moving_feat_2d_bev",
                    "moving_feat_2d_rv",
                    "movable_logit_2d_bev",
                    "movable_logit_2d_rv",
                    "moving_pred_before_bev",
                    "moving_pred_before_rv",
                    "moving_pred_after_bev",
                    "moving_pred_after_rv",
                    "movable_pred_bev",
                    "movable_pred_rv",
                    "rv_feat_shallow_sig_rv",
                    "rv_feat_deep_sig_rv",
                    "movable_logit_2d_sig_rv",
                ],
                "max",
            )

        return moving_logit_3d, movable_logit_2d

    def forward(self, xyzi, des_coord, sph_coord, current_moving_label_3d, current_movable_label_2d):
        moving_logit_3d, movable_logit_2d = self.infer(xyzi, des_coord, sph_coord)

        loss_moving = self.get_loss(moving_logit_3d, current_moving_label_3d, self.moving_wce_loss)
        loss_movable = self.get_loss(movable_logit_2d, current_movable_label_2d, self.movable_wce_loss)
        loss = loss_moving + loss_movable

        return {
            "loss": loss,
            "loss_moving": loss_moving,
            "loss_movable": loss_movable,
        }


if __name__ == "__main__":
    model = FarMOS()

    # 데이터로더 출력 shape에 맞춘 더미 데이터 생성
    B, T, N = 1, 3, 160000
    xyzi = torch.randn(B, T, 7, N, 1)  # 7 channels: x, y, z, i, dist, diff_x, diff_y
    des_coord = torch.rand(B, T, N, 3, 1) * 512  # x_quan, y_quan, z_quan
    sph_coord = torch.rand(B, T, N, 3, 1) * 2048  # theta_quan, phi_quan, r_quan

    # PyTorch CrossEntropyLoss를 위해 [B, N] 형태로 맞춥니다.
    current_moving_label_3d = torch.randint(0, 3, (B, N))

    # 2D RV 보조 네트워크 레이블은 (B, H, W)가 맞습니다.
    current_movable_label_2d = torch.randint(0, 3, (B, 64, 2048))

    # 주의: 모델의 출력(moving_logit_3d)이 [B, C, N, 1] 형태라면,
    # Loss 계산 직전에 pred.squeeze(-1)을 해주어야 [B, C, N] vs [B, N] 매칭이 되어 에러가 나지 않습니다.
    output = model(xyzi, des_coord, sph_coord, current_moving_label_3d, current_movable_label_2d)
    output["loss"].backward()
    print("Forward Pass Successful! Loss:", output["loss"].item())
