from functools import partial
import os
import shutil
import torch
from torch.utils.data import DataLoader

from datasets.dataloader import DataloadTrain, DataloadVal, DataloadTest


# ============================================================
# Optimizer / Scheduler
# ============================================================


def build_optimizer(cfg, model):
    name = cfg["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
            nesterov=cfg["nesterov"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg, optimizer):
    name = cfg["scheduler"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    elif name == "step":
        import math

        num_epoch = cfg["epochs"] - cfg["begin_epoch"]

        def schedule_with_warmup(epoch, num_epoch, pct_start, step, decay_factor):
            warmup_epochs = int(num_epoch * pct_start)
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                step_idx = epoch // step
                return math.pow(decay_factor, step_idx)

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                schedule_with_warmup,
                num_epoch=num_epoch,
                pct_start=cfg["pct_start"],
                step=cfg["step"],
                decay_factor=cfg["decay_factor"],
            ),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# ============================================================
# DataLoader
# ============================================================


def build_train_loader(cfg):
    dataset = DataloadTrain(cfg["sequence_dir"])
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )


def build_val_loader(sequence_dir, num_workers=4):
    dataset = DataloadVal(sequence_dir)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_test_loader(sequence_dir, seq_num, batch_size=1, num_workers=4):
    dataset = DataloadTest(sequence_dir, seq_num)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# ============================================================
# Code Snapshot
# ============================================================


def snapshot_code(log_dir):
    """Copy all source files (excluding logs/) to logs/ExpXX/code/."""
    code_dir = os.path.join(log_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    root = os.getcwd()
    skip_dirs = {"logs", "__pycache__", ".git"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        rel_dir = os.path.relpath(dirpath, root)
        for fname in filenames:
            dst_dir = os.path.join(code_dir, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(os.path.join(dirpath, fname), os.path.join(dst_dir, fname))
