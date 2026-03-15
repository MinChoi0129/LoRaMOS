import argparse
import os
import warnings
import yaml
import torch

warnings.filterwarnings("ignore", message=".*Online softmax.*")
from tqdm import tqdm

from networks.MainNetwork import FarMOS
from datasets.config import RANGE_BINS
from utils.metrics import validate
from utils.logger import Logger, init_wandb, log_epoch
from utils.checkpoint import load_checkpoint, save_all_best_checkpoints
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

    best_ious = {"moving": 0.0, **{f"{r[0]}_{r[1]}m": 0.0 for r in RANGE_BINS}}
    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(model, os.path.join(ckpt_dir, "latest.pth"))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ious = ckpt.get("best_ious", best_ious)
        logger.log(f"Resumed from epoch {ckpt['epoch']}, best_ious: {best_ious}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        ep_loss, ep_mov, ep_mbl, n = 0.0, 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}", dynamic_ncols=True)
        for xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid_t0 in pbar:
            xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid_t0 = (
                xyzi.cuda(),
                bev_coord.cuda(),
                rv_coord.cuda(),
                rv_input.cuda(),
                label_3d.cuda(),
                label_2d.cuda(),
                num_valid_t0.cuda(),
            )

            optimizer.zero_grad()
            out = model(xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid_t0)
            out["loss"].backward()
            optimizer.step()

            ep_loss += out["loss"].item()
            ep_mov += out["loss_moving"].item()
            ep_mbl += out["loss_movable"].item()
            n += 1
            pbar.set_postfix(loss=f"{ep_loss/n:.4f}", mov=f"{ep_mov/n:.4f}", mbl=f"{ep_mbl/n:.4f}")

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        val = validate(model, val_loader, RANGE_BINS)

        log_epoch(logger, epoch, ep_loss, ep_mov, ep_mbl, n, val, lr_now)
        save_all_best_checkpoints(model, optimizer, scheduler, epoch, ckpt_dir, val, best_ious, RANGE_BINS, logger)

    logger.log("Training finished.")
    logger.close()
