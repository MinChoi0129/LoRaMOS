import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from networks import backbone_moving, loss, SubNetworks
from core.projector_unprojector import project, unproject
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
        bev_in_ch = NUM_TEMPORAL_FRAMES * self.pointnet_ch
        self.movable_fuse_conv = nn.Sequential(
            nn.Conv2d(bev_in_ch + 3, bev_in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(bev_in_ch),
            nn.ReLU(inplace=True),
        )
        self.moving_net = SubNetworks.MovingNet(in_channels=bev_in_ch)
        self.movable_net = SubNetworks.MovableNet()
        self.point_fuse = backbone_moving.CatFusion([self.pointnet_ch, 32, 3], 3)
        self.moving_head_2d = nn.Conv2d(32, 3, kernel_size=1, bias=True)

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

        self.moving_nll_loss_3d = nn.NLLLoss(weight=moving_loss_w.double(), ignore_index=0)
        self.moving_nll_loss_2d = nn.NLLLoss(weight=moving_loss_w.double(), ignore_index=0)
        self.movable_nll_loss = nn.NLLLoss(weight=movable_loss_w.double(), ignore_index=0)
        self.lovasz_loss = loss.LovaszSoftmax(ignore=0)

    def get_loss(self, pred, label, mode):
        if mode == "moving":
            pred = pred.view(pred.shape[0], pred.shape[1], -1).unsqueeze(-1)
            label = label.view(pred.shape[0], -1).unsqueeze(-1)

        l_nll_fn = {"moving": self.moving_nll_loss_3d, "moving_2d": self.moving_nll_loss_2d, "movable": self.movable_nll_loss}[
            mode
        ]
        l_nll = l_nll_fn(F.log_softmax(pred, dim=1).double(), label).float()
        l_lovasz = self.lovasz_loss(pred, label.long())

        return l_nll + l_lovasz

    def infer(self, pcd_input, rv_input, bev_coord, rv_coord):
        # **************** [ Step 0: shape, time, 좌표 추출 ] ****************
        B, T, C, N, _ = pcd_input.shape

        # **************** [ Step 1: Point Feature 추출 ] ****************
        valid_mask = (pcd_input[:, :, 4:5, :, :] < 100.0).float()  # [B, T, 1, N, 1]
        valid_mask_flat = valid_mask.view(B * T, 1, N, 1)  # [B*T, 1, N, 1]
        pcd_input = pcd_input * valid_mask  # [B, T, 7, N, 1]
        pcd_flat = pcd_input.view(B * T, C, N, 1)  # [B*T, 7, N, 1]

        pcd_feat = self.pointnet(pcd_flat)  # [B*T, 64, N, 1]
        pcd_feat = pcd_feat * valid_mask_flat  # [B*T, 64, N, 1]
        pcd_feat_t0 = pcd_feat.view(B, T, 64, N, 1)[:, -1]  # [B, 64, N, 1]

        # **************** [ Step 2: Semantic 힌트 생성 (Range View) ] ***************
        movable_logit_rv = self.movable_net(rv_input)  # [B, K=3, 64, 2048]
        movable_logit_as_3d = unproject(movable_logit_rv, rv_coord[:, -1], scale=1.0)  # [B, K=3, N, 1]

        # Movable logit → BEV (raw logit concat)
        movable_logit_as_bev = project(movable_logit_as_3d, bev_coord[:, -1], view="bev")  # [B, 3, H, W]

        # **************** [ Step 3: BEV Moving 예측 생성 ] ****************
        bev_input = project(pcd_feat, bev_coord, view="bev")  # [B, T*64, H, W]
        bev_input = self.movable_fuse_conv(torch.cat([bev_input, movable_logit_as_bev.detach()], dim=1))  # [B, T*64, H, W]

        moving_feat_bev = self.moving_net(bev_input)  # [B, 32, 256, 256]
        moving_logit_2d_bev = self.moving_head_2d(moving_feat_bev)  # [B, 3, 256, 256]

        moving_feat_3d = unproject(moving_feat_bev, bev_coord[:, -1], scale=0.5)  # [B, 32, N, 1]

        # **************** [ Step 4: 3차원 피처 모두 융합 ] ****************
        moving_logit_3d = self.point_fuse(pcd_feat_t0, moving_feat_3d, movable_logit_as_3d.detach())  # [B, K=3, N, 1]

        return {
            "moving_logit_3d": moving_logit_3d,
            "moving_logit_2d_bev": moving_logit_2d_bev,
            "movable_logit_2d": movable_logit_rv,
            "visualization": [
                (moving_logit_2d_bev.argmax(dim=1, keepdim=True).float(), "pred_moving_bev"),
                (movable_logit_rv.argmax(dim=1, keepdim=True).float(), "pred_movable_rv"),
                (bev_input, "feat_bev_input"),
                (moving_feat_bev, "feat_moving_bev"),
            ],
        }

    def forward(
        self,
        pcd_input,
        rv_input,
        bev_coord,
        rv_coord,
        label_moving_3d,
        label_movable_rv,
        label_moving_bev,
    ):
        output = self.infer(pcd_input, rv_input, bev_coord, rv_coord)

        moving_logit_3d = output["moving_logit_3d"]
        moving_logit_2d_bev = output["moving_logit_2d_bev"]
        movable_logit_2d = output["movable_logit_2d"]

        loss_moving = self.get_loss(moving_logit_3d, label_moving_3d, mode="moving")
        loss_moving_2d = self.get_loss(moving_logit_2d_bev, label_moving_bev, mode="moving_2d")
        loss_movable = self.get_loss(movable_logit_2d, label_movable_rv, mode="movable")
        total_loss = loss_moving + loss_moving_2d + loss_movable

        return {
            "loss": total_loss,
            "loss_moving": loss_moving,
            "loss_moving_2d": loss_moving_2d,
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

    print("Loading sample from dataloader...")
    for batch in loader:
        pcd_input, rv_input, bev_coord, rv_coord, label_moving_3d, label_movable_rv, label_moving_bev, num_valid_t0 = [
            x.to(device) for x in batch
        ]

        output = model(pcd_input, rv_input, bev_coord, rv_coord, label_moving_3d, label_movable_rv, label_moving_bev)
        print("Forward Pass Successful! Loss:", output["loss"].item())
        print(
            f"  loss_moving={output['loss_moving'].item():.4f}, loss_moving_2d={output['loss_moving_2d'].item():.4f}, loss_movable={output['loss_movable'].item():.4f}"
        )
        output["loss"].backward()
        print("Backward Pass Successful!")
        break
