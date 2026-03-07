import argparse
import os
import warnings
import yaml
import torch

warnings.filterwarnings("ignore", message=".*Online softmax.*")
import numpy as np
from tqdm import tqdm

from networks.MainNetwork import FarMOS
from utils.metrics import iouEval
from utils.logger import Logger, init_wandb, log_wandb
from utils.checkpoint import save_checkpoint, load_checkpoint, save_best_checkpoint
from utils.builder import (
    build_optimizer,
    build_scheduler,
    build_train_loader,
    build_val_loader,
    snapshot_code,
)


def get_args():
    parser = argparse.ArgumentParser("FarMOS Training")
    parser.add_argument("--train_config", type=str, default="config/train.yaml")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate(model, val_loader):
    model.eval()
    total_loss, total_mov, total_mbl, n = 0.0, 0.0, 0.0, 0
    moving_evaluator = iouEval(n_classes=3, ignore=[0])
    movable_evaluator = iouEval(n_classes=3, ignore=[0])
    range_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    range_evaluators = {r: iouEval(n_classes=3, ignore=[0]) for r in range_bins}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Val", dynamic_ncols=True):
            xyzi, des_coord, sph_coord, rv_input, label_3d, label_2d, num_valid, _, _ = batch
            xyzi, des_coord, sph_coord, rv_input, label_3d, label_2d = (
                xyzi.cuda(),
                des_coord.cuda(),
                sph_coord.cuda(),
                rv_input.cuda(),
                label_3d.cuda(),
                label_2d.cuda(),
            )

            out = model(xyzi, des_coord, sph_coord, rv_input, label_3d, label_2d)
            total_loss += out["loss"].item()
            total_mov += out["loss_moving"].item()
            total_mbl += out["loss_movable"].item()
            n += 1

            # Moving evaluation (3D point-wise)
            moving_pred = out["moving_logit_3d"].squeeze(-1).argmax(dim=1).cpu().numpy()
            moving_gt = label_3d.cpu().numpy()
            # xyzi channel 4 = distance, 현재 프레임(t=-1)
            depth = xyzi[:, -1, 4, :, 0].cpu().numpy()  # [B, N]

            for b in range(moving_pred.shape[0]):
                nv = num_valid[b].item()
                pred_v = moving_pred[b][:nv]
                gt_v = moving_gt[b][:nv].astype(np.int32)
                moving_evaluator.addBatch(pred_v, gt_v)

                # Range-wise moving evaluation
                depth_v = depth[b][:nv]
                for (rmin, rmax), reval in range_evaluators.items():
                    rmask = (depth_v >= rmin) & (depth_v < rmax)
                    if rmask.any():
                        reval.addBatch(pred_v[rmask], gt_v[rmask])

            # Movable evaluation (2D RV pixel-wise)
            movable_pred = out["movable_logit_2d"].argmax(dim=1).cpu().numpy()
            movable_gt = label_2d.cpu().numpy()
            for b in range(movable_pred.shape[0]):
                movable_evaluator.addBatch(movable_pred[b].flatten(), movable_gt[b].flatten().astype(np.int32))

    _, moving_iou = moving_evaluator.getIoU()
    _, movable_iou = movable_evaluator.getIoU()
    result = {
        "loss": total_loss / n,
        "loss_moving": total_mov / n,
        "loss_movable": total_mbl / n,
        "iou_static": moving_iou[1].item(),
        "iou_moving": moving_iou[2].item(),
        "iou_immovable": movable_iou[1].item(),
        "iou_movable": movable_iou[2].item(),
    }
    for (rmin, rmax), reval in range_evaluators.items():
        _, r_iou = reval.getIoU()
        result[f"iou_moving_{rmin}_{rmax}m"] = r_iou[2].item()
    return result


if __name__ == "__main__":
    args = get_args()
    with open(args.train_config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not args.resume:
        snapshot_code(args.log_dir)

    logger = Logger(args.log_dir)
    logger.log(f"Train config: {cfg}")

    wandb_name = args.wandb_name or os.path.basename(args.log_dir.rstrip("/"))
    init_wandb(argparse.Namespace(**cfg, wandb_name=wandb_name), log_dir=args.log_dir, resume=args.resume)

    train_loader = build_train_loader(cfg)
    val_loader = build_val_loader(cfg["sequence_dir"], cfg["num_workers"])

    model = torch.compile(FarMOS().cuda())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total Trainable parameters: {num_params:,}")
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    best_moving_iou, start_epoch = 0.0, 1
    if args.resume:
        ckpt = load_checkpoint(model, os.path.join(ckpt_dir, "latest.pth"))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_moving_iou = ckpt.get("best_moving_iou", 0.0)
        logger.log(f"Resumed from epoch {ckpt['epoch']}, best_iou_moving: {best_moving_iou:.6f}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        ep_loss, ep_mov, ep_mbl, n = 0.0, 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}", dynamic_ncols=True)
        for xyzi, des_coord, sph_coord, rv_input, label_3d, label_2d in pbar:
            xyzi, des_coord, sph_coord = xyzi.cuda(), des_coord.cuda(), sph_coord.cuda()
            rv_input = rv_input.cuda()
            label_3d, label_2d = label_3d.cuda(), label_2d.cuda()

            optimizer.zero_grad()
            out = model(xyzi, des_coord, sph_coord, rv_input, label_3d, label_2d)
            out["loss"].backward()
            optimizer.step()

            ep_loss += out["loss"].item()
            ep_mov += out["loss_moving"].item()
            ep_mbl += out["loss_movable"].item()
            n += 1
            pbar.set_postfix(loss=f"{ep_loss/n:.4f}", mov=f"{ep_mov/n:.4f}", mbl=f"{ep_mbl/n:.4f}")

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        val = validate(model, val_loader)

        range_str = " | ".join(f"{k}: {val[k]:.4f}" for k in val if k.startswith("iou_moving_"))
        logger.log(
            f"[Epoch {epoch:03d}] Train: {ep_loss/n:.4f} | "
            f"Val: {val['loss']:.4f} (mov: {val['loss_moving']:.4f}, mbl: {val['loss_movable']:.4f}) | "
            f"Moving IoU static: {val['iou_static']:.6f}, moving: {val['iou_moving']:.6f} | "
            f"Movable IoU immovable: {val['iou_immovable']:.6f}, movable: {val['iou_movable']:.6f} | LR: {lr_now:.6f}\n"
            f"  Range-wise Moving IoU: {range_str}"
        )

        wandb_dict = {
            "train/loss": ep_loss / n,
            "train/loss_moving": ep_mov / n,
            "train/loss_movable": ep_mbl / n,
            "val/loss": val["loss"],
            "val/loss_moving": val["loss_moving"],
            "val/loss_movable": val["loss_movable"],
            "val/iou_static": val["iou_static"],
            "val/iou_moving": val["iou_moving"],
            "val/iou_immovable": val["iou_immovable"],
            "val/iou_movable": val["iou_movable"],
            "lr": lr_now,
        }
        for k in val:
            if k.startswith("iou_moving_"):
                wandb_dict[f"val/{k}"] = val[k]
        log_wandb(wandb_dict, step=epoch)

        save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(ckpt_dir, "latest.pth"), best_moving_iou)
        best_moving_iou = save_best_checkpoint(
            model, optimizer, scheduler, epoch, ckpt_dir, val["iou_moving"], best_moving_iou, logger
        )

    logger.log("Training finished.")
    logger.close()
