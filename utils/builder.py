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
            model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg, optimizer):
    name = cfg["scheduler"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["step_gamma"])
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
# Inference
# ============================================================


def predict(model, xyzi, des_coord, sph_coord):
    """model.infer -> argmax -> numpy [B, N]"""
    moving_logit_3d, _ = model.infer(xyzi, des_coord, sph_coord)
    return moving_logit_3d.squeeze(-1).argmax(dim=1).cpu().numpy()


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
