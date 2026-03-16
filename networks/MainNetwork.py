import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from networks import backbone_moving, loss, SubNetworks
from utils.projector_unprojector import project, unproject
from datasets.config import NUM_TEMPORAL_FRAMES


class FarMOS(nn.Module):
    def __init__(self):
        super(FarMOS, self).__init__()

        self._build_network()
        self._build_loss()

    def _build_network(self):
        def __print_num_params(model, model_name):
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: [{model_name}] > {num_params:,}")

        self.pointnet_ch = 64
        self.pointnet = backbone_moving.PointNetStacker(7, self.pointnet_ch, pre_bn=True, stack_num=2)
        self.moving_net = SubNetworks.MovingNet(in_channels=NUM_TEMPORAL_FRAMES * self.pointnet_ch)
        self.movable_net = SubNetworks.MovableNet()
        self.point_fuse = backbone_moving.CatFusion([self.pointnet_ch, 32, 3], 3)

        __print_num_params(self.pointnet, "pointnet")
        __print_num_params(self.moving_net, "moving_net")
        __print_num_params(self.movable_net, "movable_net")
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

    def get_loss(self, pred, label, mode, num_valid=None):
        if mode == "moving":
            B, C = pred.shape[0], pred.shape[1]
            N = pred.shape[2]

            pred = pred.view(B, C, -1).unsqueeze(-1)
            label = label.view(B, -1).unsqueeze(-1)

            per_point_nll = self.moving_nll_loss(F.log_softmax(pred, dim=1).double(), label)

            # num_valid 마스킹: 패딩 영역 제외
            target_valid = (label != 0).float()
            if num_valid is not None:
                seq_range = torch.arange(N, device=pred.device).view(1, N, 1)
                valid_mask = seq_range < num_valid.view(B, 1, 1)
                target_valid = target_valid * valid_mask.float()

            l_nll = (per_point_nll * target_valid.double()).sum() / (target_valid.sum() + 1e-6)
            l_nll = l_nll.float()

            l_lovasz = self.lovasz_loss(pred, label.long(), ignore=0)

            return l_nll + l_lovasz

        elif mode == "movable":
            l_nll = self.movable_nll_loss(F.log_softmax(pred, dim=1).double(), label).float()
            l_lovasz = self.lovasz_loss(pred, label.long(), ignore=0)

            return l_nll + l_lovasz

    def infer(self, xyzi, bev_coord, rv_coord, rv_input):
        # **************** [ Step 0: shape, time, 좌표 추출 ] ****************
        B, T, C, N, _ = xyzi.shape

        # **************** [ Step 1: Point Feature 추출 ] ****************
        # 패딩 포인트 마스킹 (BatchNorm 오염 방지)
        valid_mask = (xyzi[:, :, 4:5, :, :] < 100.0).float()  # [B, T, 1, N, 1]
        valid_mask_flat = valid_mask.view(B * T, 1, N, 1)  # [B*T, 1, N, 1]
        xyzi = xyzi * valid_mask  # [B, T, 7, N, 1]
        xyzi_flatten = xyzi.view(B * T, C, N, 1)  # [B*T, 7, N, 1]

        pcd_feat = self.pointnet(xyzi_flatten)  # [B*T, 64, N, 1]
        pcd_feat = pcd_feat * valid_mask_flat  # [B*T, 64, N, 1]
        pcd_feat_t0 = pcd_feat.view(B, T, 64, N, 1)[:, -1]  # [B, 64, N, 1]

        # **************** [ Step 2: Semantic 힌트 생성 ] ***************
        movable_logit_rv = self.movable_net(rv_input)  # [B, K=3, 64, 2048]
        movable_logit_as_3d = unproject(movable_logit_rv, rv_coord[:, -1], scale=1.0)  # [B, K=3, N, 1]

        # Movable logit → BEV (softmax → 객체 존재 확률 마스크)
        movable_logit_as_bev = project(movable_logit_as_3d, bev_coord[:, -1], view="bev")  # [B, K=3, H, W]
        movable_prob_bev = torch.softmax(movable_logit_as_bev, dim=1)
        movable_mask_bev = movable_prob_bev[:, 2:3, :, :].detach()  # [B, 1, H, W]

        # **************** [ Step 3: BEV Moving 예측 생성 ] ****************
        bev_input = project(pcd_feat, bev_coord, view="bev")  # [B, T*64, H, W]

        moving_feat_bev = self.moving_net(bev_input, movable_mask_bev)  # [B, 32, H, W]
        moving_feat_3d = unproject(moving_feat_bev, bev_coord[:, -1], scale=1.0)  # [B, 32, N, 1]

        # **************** [ Step 4: 3차원 피처 모두 융합 ] ****************
        moving_logit_3d = self.point_fuse(pcd_feat_t0, moving_feat_3d, movable_logit_as_3d.detach())  # [B, K=3, N, 1]

        return {
            "moving_logit_3d": moving_logit_3d,
            "movable_logit_2d": movable_logit_rv,
            "visualization": [
                (movable_logit_rv.argmax(dim=1, keepdim=True).float(), "pred_movable_rv"),
                (movable_logit_as_bev.argmax(dim=1, keepdim=True).float(), "pred_movable_bev"),
                (torch.softmax(movable_logit_rv, dim=1)[:, 2:3, :, :], "heatmap_movable_rv"),
                (movable_mask_bev, "heatmap_movable_bev"),
                (bev_input, "feat_bev_input"),
                (moving_feat_bev, "feat_moving_bev"),
            ],
        }

    def forward(self, xyzi, bev_coord, rv_coord, rv_input, current_moving_label_3d, current_movable_label_2d, num_valid_t0):
        output = self.infer(xyzi, bev_coord, rv_coord, rv_input)

        moving_logit_3d = output["moving_logit_3d"]
        movable_logit_2d = output["movable_logit_2d"]

        loss_moving = self.get_loss(moving_logit_3d, current_moving_label_3d, mode="moving", num_valid=num_valid_t0)
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
    from datasets.dataloader import DataloadTrain
    from torch.utils.data import DataLoader

    loader = DataLoader(
        DataloadTrain("/home/ssd_data/ROOT_KITTI/KITTI/dataset/sequences"),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device("cuda")
    model = FarMOS().to(device)
    model.train()

    # Load real data once from dataloader
    print("Loading sample from dataloader...")
    for xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid_t0 in loader:
        xyzi = xyzi.to(device)
        bev_coord = bev_coord.to(device)
        rv_coord = rv_coord.to(device)
        rv_input = rv_input.to(device)
        label_3d = label_3d.to(device)
        label_2d = label_2d.to(device)
        num_valid_t0 = num_valid_t0.to(device)

        output = model(xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid_t0)
        print("Forward Pass Successful! Loss:", output["loss"].item())
        break
