import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from networks import backbone_BEV, loss, SubNetworks
from utils.pretty_printer_and_saver import save_feature_as_img, shprint
from utils import projector_unprojector
from datasets.config import NUM_TEMPORAL_FRAMES


class FarMOS(nn.Module):
    def __init__(self):
        super(FarMOS, self).__init__()

        self._build_network()
        self._build_loss()
        self._build_utils()

    def _build_network(self):
        def __print_num_params(model, model_name):
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: [{model_name}] > {num_params:,}")

        self.pointnet_ch = 64
        self.pointnet = backbone_BEV.PointNetStacker(7, self.pointnet_ch, pre_bn=True, stack_num=2)
        self.moving_bev_net = SubNetworks.BEVNet(in_channels=NUM_TEMPORAL_FRAMES * self.pointnet_ch)
        self.movable_rv_net = SubNetworks.RVNet()
        self.point_fuse = backbone_BEV.CatFusion([self.pointnet_ch, 32, 3], 3)

        __print_num_params(self.pointnet, "pointnet")
        __print_num_params(self.moving_bev_net, "moving_bev_net")
        __print_num_params(self.movable_rv_net, "movable_rv_net")
        __print_num_params(self.point_fuse, "point_fuse")

    def _build_loss(self):
        with open("config/semantic-kitti-mos.yaml", "r") as f:
            task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        def get_weights(mapping_key):
            content = torch.zeros(3, dtype=torch.float32)
            for cl, freq in task_cfg["content"].items():
                x_cl = task_cfg[mapping_key][cl]
                content[x_cl] += freq

            # Log-scale weighting
            weights = 1.0 / torch.log(content + 1.02)
            weights[0] = 0  # Unlabeled 무시
            return weights

        moving_loss_w = get_weights("learning_map")
        movable_loss_w = get_weights("movable_learning_map")

        self.moving_nll_loss = nn.NLLLoss(weight=moving_loss_w.double(), ignore_index=0, reduction="none")
        self.movable_nll_loss = nn.NLLLoss(weight=movable_loss_w.double(), ignore_index=0)
        self.lovasz_loss = loss.lovasz_softmax

    def _build_utils(self):
        # Projectors (좌표 flip 걱정 없이)
        self.project_to_bev = projector_unprojector.project_to_bev  # T축 필요 (단일 시간이면 T축에 1이라도 있어야함)
        self.project_to_rv = projector_unprojector.project_to_rv  # T축 없어야함

        # Unprojectos by resolution. (Note. BEV는 그냥 좌표 받아도 되는데 RV unproject시에는 flip된 좌표 받아야함)
        self.unproject_from_full = projector_unprojector.unprojectors["full"]  # T축 없어야함
        self.unproject_from_half = projector_unprojector.unprojectors["half"]  # T축 없어야함

    def get_loss(self, pred, label, mode, dist=None):
        if mode == "moving":
            B, C = pred.shape[0], pred.shape[1]

            pred = pred.view(B, C, -1).unsqueeze(-1)
            label = label.view(B, -1).unsqueeze(-1)

            # Per-point NLL (reduction='none') → [B, N, 1]
            per_point_nll = self.moving_nll_loss(F.log_softmax(pred, dim=1).double(), label)

            # 거리 제곱 가중치: 멀리 있는 점일수록 더 큰 페널티
            dist_w = dist.view(B, -1, 1) ** 2  # [B, N, 1]
            dist_w = dist_w / (dist_w.mean() + 1e-6)  # 정규화
            per_point_nll = per_point_nll * dist_w.double()

            # ignore_index=0인 점 제외하고 평균
            valid = (label != 0).float()
            l_nll = (per_point_nll * valid.double()).sum() / (valid.sum() + 1e-6)
            l_nll = l_nll.float()

            l_lovasz = self.lovasz_loss(pred, label.long(), ignore=0)

            return l_nll + l_lovasz

        elif mode == "movable":
            l_nll = self.movable_nll_loss(F.log_softmax(pred, dim=1).double(), label).float()
            l_lovasz = self.lovasz_loss(pred, label.long(), ignore=0)

            return l_nll + l_lovasz

    def infer(self, xyzi, des_coord, sph_coord, rv_input):
        # """
        # xyzi: [B, T, 7, N, 1]
        # des_coord: [B, T, N, 3, 1]
        # sph_coord: [B, T, N, 3, 1]
        # rv_input: [B, 5, 64, 2048]
        # """
        B, T, C, N, _ = xyzi.shape

        # **************** [ Step 1: Point Feature 추출 ] ****************
        # 패딩 포인트 마스킹 (BatchNorm 오염 방지)
        # Channel 4 = distance. 패딩 포인트: ~4243
        valid_mask = (xyzi[:, :, 4:5, :, :] < 100.0).float()  # [B, T=3, 1, N, 1]
        valid_mask_flat = valid_mask.view(B * T, 1, N, 1)  # [B*T, 1, N, 1]

        xyzi = xyzi * valid_mask  # [B, T, 7, N, 1]
        xyzi_flatten = xyzi.view(B * T, C, N, 1)  # [B*T, 7, N, 1]

        pcd_feat = self.pointnet(xyzi_flatten)  # [B*T, 64, N, 1]
        pcd_feat = pcd_feat * valid_mask_flat  # [B*T, 64, N, 1]

        # **************** [ Step 2: 시간 분리 ] ****************
        pcd_feat_t0 = pcd_feat.view(B, T, 64, N, 1)[:, -1]  # [B, 64, N, 1]

        # 2채널
        bev_coord_t0 = des_coord[:, -1, :, :2, :]  # [B, N, 2, 1]
        rv_coords_t0 = sph_coord[:, -1, :, :2, :].flip(-2)  # [B, N, 2, 1]

        # 3채널
        des_coord_t0 = des_coord[:, -1:, :, :, :]  #  [B, 1, N, 3, 1]

        # **************** [ Step 3: 누적형 Point Feature -> BEV Feature ] ****************
        bev_input = self.project_to_bev(pcd_feat, des_coord)  # [B, 192, H=512, W=512]

        # # **************** [ Step 4: Semantic 예측 및 중간-2D-Feature를 3D로 복원 ] ****************
        movable_logit_2d = self.movable_rv_net(rv_input)  # [B, K=3, H=64, W=2048]
        movable_logit_as_3d = self.unproject_from_full(movable_logit_2d, rv_coords_t0)  # [B, K=3, N, 1]

        # **************** [ Step 5: BEV Semantic Supervision 생성 및 BEV 피쳐 생성 ] ****************
        movable_logit_as_bev = self.project_to_bev(movable_logit_as_3d, des_coord_t0)  # [B, K=3, 512, 512]
        movable_probability_as_bev = torch.softmax(movable_logit_as_bev, dim=1)  # [B, K=3, 512, 512]
        movable_probability_mask_bev = movable_probability_as_bev[:, 2:3, :, :].detach()  # [B, p=1, 512, 512] => 객체 존재 확률
        moving_feat_2d = self.moving_bev_net(
            bev_input,
            movable_probability_mask_bev,
        )  # [B, C=32, H=512, W=512]

        # **************** [ Step 6: 3차원 피처 모두 융합 ] ****************
        moving_feat_3d = self.unproject_from_full(moving_feat_2d, bev_coord_t0)  # [B, C=32, N, 1]
        moving_logit_3d = self.point_fuse(
            pcd_feat_t0,
            moving_feat_3d,
            movable_logit_as_3d.detach(),
        )  # [B, K=3, N, 1]

        RUN_SAVE_FEATURE = False
        if RUN_SAVE_FEATURE:
            save_feature_as_img(
                [
                    bev_input,
                    movable_logit_2d,
                    movable_logit_2d.argmax(dim=1),
                    self.project_to_bev(moving_feat_3d, des_coord_t0),
                    self.project_to_bev(moving_logit_3d, des_coord_t0).argmax(dim=1),
                    movable_probability_mask_bev,
                    moving_feat_2d,
                ],
                [
                    "bev_input",
                    "movable_logit_2d",
                    "movable_logit_2d_argmax",
                    "moving_feat_3d_into_bev",
                    "moving_logit_3d_argmax_into_bev",
                    "movable_probability_mask_bev",
                    "moving_feat_2d",
                ],
                "max",
            )

        return moving_logit_3d, movable_logit_2d

    def forward(self, xyzi, des_coord, sph_coord, rv_input, current_moving_label_3d, current_movable_label_2d):
        moving_logit_3d, movable_logit_2d = self.infer(xyzi, des_coord, sph_coord, rv_input)

        dist_t0 = xyzi[:, -1, 4, :, 0]  # [B, N] — channel 4 = distance
        loss_moving = self.get_loss(moving_logit_3d, current_moving_label_3d, mode="moving", dist=dist_t0)
        loss_movable = self.get_loss(movable_logit_2d, current_movable_label_2d, mode="movable")
        loss = loss_moving + loss_movable

        return {
            "loss": loss,
            "loss_moving": loss_moving,
            "loss_movable": loss_movable,
            "moving_logit_3d": moving_logit_3d,
            "movable_logit_2d": movable_logit_2d,
        }


if __name__ == "__main__":
    model = FarMOS()

    # 데이터로더 출력 shape에 맞춘 더미 데이터 생성
    B, T, N = 2, 3, 160000
    xyzi = torch.randn(B, T, 7, N, 1)  # 7 channels: x, y, z, i, dist, diff_x, diff_y
    des_coord = torch.rand(B, T, N, 3, 1) * 512  # x_quan, y_quan, z_quan
    sph_coord = torch.rand(B, T, N, 3, 1) * 2048  # theta_quan, phi_quan, r_quan

    # PyTorch CrossEntropyLoss를 위해 [B, N] 형태로 맞춥니다.
    current_moving_label_3d = torch.randint(0, 3, (B, N))

    # 2D RV 보조 네트워크 레이블은 (B, H, W)가 맞습니다.
    current_movable_label_2d = torch.randint(0, 3, (B, 64, 2048))

    rv_input = torch.randn(B, 5, 64, 2048)

    # 주의: 모델의 출력(moving_logit_3d)이 [B, C, N, 1] 형태라면,
    # Loss 계산 직전에 pred.squeeze(-1)을 해주어야 [B, C, N] vs [B, N] 매칭이 되어 에러가 나지 않습니다.
    output = model(xyzi, des_coord, sph_coord, rv_input, current_moving_label_3d, current_movable_label_2d)
    print("Forward Pass Successful! Loss:", output["loss"].item())
