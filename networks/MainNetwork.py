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

        self.moving_nll_loss_3d = nn.NLLLoss(weight=moving_loss_w.double(), ignore_index=0)
        self.movable_nll_loss = nn.NLLLoss(weight=movable_loss_w.double(), ignore_index=0)
        self.lovasz_loss = loss.LovaszSoftmax(ignore=0)

    def get_loss(self, pred, label, mode, dist=None):
        if mode == "moving":
            pred = pred.view(pred.shape[0], pred.shape[1], -1).unsqueeze(-1)
            label = label.view(pred.shape[0], -1).unsqueeze(-1)
            if dist is not None and self.training:
                pred, label = self._range_balanced_sample(pred, label, dist)

        l_nll_fn = {
            "moving": self.moving_nll_loss_3d,
            "movable": self.movable_nll_loss,
        }[mode]
        l_nll = l_nll_fn(F.log_softmax(pred, dim=1).double(), label).float()
        l_lovasz = self.lovasz_loss(pred, label.long())

        return l_nll + l_lovasz

    @staticmethod
    def _range_balanced_sample(pred, label, dist, samples_per_bin=4000):
        """각 거리 bin에서 동일 수의 포인트를 샘플링. 배치를 concat하여 반환."""
        from datasets.config import RANGE_BINS

        B = pred.shape[0]
        all_preds, all_labels = [], []

        for b in range(B):
            for rmin, rmax in RANGE_BINS:
                mask = (dist[b] >= rmin) & (dist[b] < rmax) & (label[b, :, 0] != 0)
                valid_idx = mask.nonzero(as_tuple=False).squeeze(-1)
                if len(valid_idx) == 0:
                    continue
                if len(valid_idx) > samples_per_bin:
                    perm = torch.randperm(len(valid_idx), device=valid_idx.device)[:samples_per_bin]
                    valid_idx = valid_idx[perm]
                all_preds.append(pred[b, :, valid_idx, 0])
                all_labels.append(label[b, valid_idx, 0])

        sampled_pred = torch.cat(all_preds, dim=1).unsqueeze(0).unsqueeze(-1)
        sampled_label = torch.cat(all_labels, dim=0).unsqueeze(0).unsqueeze(-1)
        return sampled_pred, sampled_label

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

        # Movable logit → BEV (softmax → movable prob mask)
        movable_logit_as_bev = project(movable_logit_as_3d, bev_coord[:, -1], view="bev")  # [B, 3, H, W]
        movable_mask_bev = torch.softmax(movable_logit_as_bev, dim=1)[:, 2:3, :, :].detach()  # [B, 1, H, W]

        # **************** [ Step 3: BEV Moving 예측 생성 ] ****************
        bev_input = project(pcd_feat, bev_coord, view="bev")  # [B, T*64, H, W]
        moving_feat_bev = self.moving_net(bev_input, movable_mask_bev)  # [B, 32, 512, 512]
        moving_feat_3d = unproject(moving_feat_bev, bev_coord[:, -1], scale=1.0)  # [B, 32, N, 1]

        # **************** [ Step 4: 3차원 피처 모두 융합 ] ****************
        moving_logit_3d = self.point_fuse(pcd_feat_t0, moving_feat_3d, movable_logit_as_3d.detach())  # [B, K=3, N, 1]

        return {
            "moving_logit_3d": moving_logit_3d,
            "movable_logit_2d": movable_logit_rv,
            "visualization": (
                None
                if self.training
                else [
                    # feat
                    (bev_input, "feat_pcd_bev"),
                    (project(pcd_feat_t0, bev_coord[:, -1], view="bev"), "feat_pcd_bev_t0"),
                    (moving_feat_bev, "feat_moving_bev"),
                    # mask
                    (movable_mask_bev, "mask_movable_bev"),
                    # pred
                    (
                        project(moving_logit_3d.argmax(dim=1, keepdim=True).float(), bev_coord[:, -1], view="bev"),
                        "pred_moving_bev",
                    ),
                    (movable_logit_rv.argmax(dim=1, keepdim=True).float(), "pred_movable_rv"),
                    (movable_logit_as_bev.argmax(dim=1, keepdim=True).float(), "pred_movable_bev"),
                ]
            ),
        }

    def forward(
        self,
        pcd_input,
        rv_input,
        bev_coord,
        rv_coord,
        label_moving_3d,
        label_movable_rv,
    ):
        output = self.infer(pcd_input, rv_input, bev_coord, rv_coord)

        moving_logit_3d = output["moving_logit_3d"]
        movable_logit_2d = output["movable_logit_2d"]

        dist_t0 = pcd_input[:, -1, 4, :, 0]  # [B, N]
        loss_moving = self.get_loss(moving_logit_3d, label_moving_3d, mode="moving", dist=dist_t0)
        loss_movable = self.get_loss(movable_logit_2d, label_movable_rv, mode="movable")
        total_loss = loss_moving + loss_movable

        return {
            "loss": total_loss,
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

    print("Loading sample from dataloader...")
    for batch in loader:
        pcd_input, rv_input, bev_coord, rv_coord, label_moving_3d, label_movable_rv, label_moving_bev, num_valid_t0 = [
            x.to(device) for x in batch
        ]

        output = model(pcd_input, rv_input, bev_coord, rv_coord, label_moving_3d, label_movable_rv)
        print("Forward Pass Successful! Loss:", output["loss"].item())
        print(
            f"  loss_moving={output['loss_moving'].item():.4f}, loss_movable={output['loss_movable'].item():.4f}"
        )
        output["loss"].backward()
        print("Backward Pass Successful!")
        break
