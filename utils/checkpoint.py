import os
import glob
import torch


def _strip_compile_prefix(state_dict):
    """Remove '_orig_mod.' prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def save_checkpoint(model, optimizer, scheduler, epoch, path, best_ious=None):
    """Save checkpoint with clean (non-compiled) state_dict keys."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _strip_compile_prefix(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_ious": best_ious or {},
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


def save_best_checkpoint(model, optimizer, scheduler, epoch, ckpt_dir, current_iou, best_iou, logger=None, tag="best"):
    """Save best model if current_iou > best_iou. Returns updated best_iou."""
    if current_iou <= best_iou:
        return best_iou

    best_iou = current_iou

    # Remove previous {tag}_*.pth
    for old in glob.glob(os.path.join(ckpt_dir, f"{tag}_*.pth")):
        os.remove(old)

    best_path = os.path.join(ckpt_dir, f"{tag}_{epoch}.pth")
    save_checkpoint(model, optimizer, scheduler, epoch, best_path)

    if logger:
        logger.log(f"  -> New {tag} model saved: {best_path} (iou: {best_iou:.6f})")

    return best_iou


def save_all_best_checkpoints(model, optimizer, scheduler, epoch, ckpt_dir, val, best_ious, range_bins, logger):
    """Save latest + best overall + best per range bin."""
    save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(ckpt_dir, "latest.pth"), best_ious)
    best_ious["moving"] = save_best_checkpoint(
        model, optimizer, scheduler, epoch, ckpt_dir, val["iou_moving"], best_ious["moving"], logger, tag="best"
    )
    for rmin, rmax in range_bins:
        key = f"{rmin}_{rmax}m"
        best_ious[key] = save_best_checkpoint(
            model, optimizer, scheduler, epoch, ckpt_dir,
            val[f"iou_moving_{rmin}_{rmax}m"], best_ious[key], logger, tag=f"best_{key}"
        )
