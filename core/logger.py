import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "log.txt")
        self.f = open(self.log_path, "a")

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(msg)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def init_wandb(args, log_dir="logs/", resume=False):
    import wandb

    os.makedirs(log_dir, exist_ok=True)

    resume_id = None
    if resume:
        wandb_dir = os.path.join(log_dir, "wandb")
        if os.path.isdir(wandb_dir):
            runs = sorted(d for d in os.listdir(wandb_dir) if d.startswith("run-"))
            if runs:
                resume_id = runs[-1].split("-")[-1]

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        dir=log_dir,
        id=resume_id or wandb.util.generate_id(),
        resume="must" if resume_id else "never",
    )


def log_wandb(metrics, step):
    import wandb

    metrics["epoch"] = step
    wandb.log(metrics)


def log_epoch(logger, epoch, train_loss, train_mov, train_mov2d, train_mbl, n, val, lr):
    range_str = " | ".join(f"{k}: {val[k]:.4f}" for k in val if k.startswith("iou_moving_"))
    logger.log(
        f"[Epoch {epoch:03d}] Train: {train_loss/n:.4f} | "
        f"Val: {val['loss']:.4f} (mov: {val['loss_moving']:.4f}, mbl: {val['loss_movable']:.4f}) | "
        f"Moving IoU static: {val['iou_static']:.6f}, moving: {val['iou_moving']:.6f} | "
        f"Movable IoU immovable: {val['iou_immovable']:.6f}, movable: {val['iou_movable']:.6f} | LR: {lr:.6f}\n"
        f"  Range-wise Moving IoU: {range_str}"
    )

    wandb_dict = {
        "train/loss": train_loss / n,
        "train/loss_moving": train_mov / n,
        "train/loss_moving_2d": train_mov2d / n,
        "train/loss_movable": train_mbl / n,
        "val/loss": val["loss"],
        "val/loss_moving": val["loss_moving"],
        "val/loss_movable": val["loss_movable"],
        "val/iou_static": val["iou_static"],
        "val/iou_moving": val["iou_moving"],
        "val/iou_immovable": val["iou_immovable"],
        "val/iou_movable": val["iou_movable"],
        "lr": lr,
    }
    for k in val:
        if k.startswith("iou_moving_"):
            wandb_dict[f"val/{k}"] = val[k]
    log_wandb(wandb_dict, step=epoch)
