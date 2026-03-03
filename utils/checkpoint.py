import os
import glob
import torch


def _strip_compile_prefix(state_dict):
    """Remove '_orig_mod.' prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def save_checkpoint(model, optimizer, scheduler, epoch, path, best_moving_iou=0.0):
    """Save checkpoint with clean (non-compiled) state_dict keys."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _strip_compile_prefix(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_moving_iou": best_moving_iou,
        },
        path,
    )


def load_checkpoint(model, path, device="cuda"):
    """Load checkpoint, handling torch.compile prefix mismatch."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_sd = _strip_compile_prefix(ckpt["model_state_dict"])

    # If model is compiled, add prefix back
    model_keys = list(model.state_dict().keys())
    if model_keys and model_keys[0].startswith("_orig_mod."):
        saved_sd = {"_orig_mod." + k: v for k, v in saved_sd.items()}

    model.load_state_dict(saved_sd)
    return ckpt


def save_best_checkpoint(model, optimizer, scheduler, epoch, ckpt_dir, current_iou, best_iou, logger=None):
    """Save best model if current_iou > best_iou. Returns updated best_iou."""
    if current_iou <= best_iou:
        return best_iou

    best_iou = current_iou

    # Remove previous best_*.pth
    for old in glob.glob(os.path.join(ckpt_dir, "best_*.pth")):
        os.remove(old)

    best_path = os.path.join(ckpt_dir, f"best_{epoch}.pth")
    save_checkpoint(model, optimizer, scheduler, epoch, best_path, best_iou)

    if logger:
        logger.log(f"  -> New best model saved: {best_path} (iou_moving: {best_iou:.6f})")

    return best_iou
