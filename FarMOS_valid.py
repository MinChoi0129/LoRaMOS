import argparse
import os
import numpy as np
import yaml
import torch
from tqdm import tqdm

from networks.MainNetwork import FarMOS
from utils.metrics import iouEval
from utils.checkpoint import load_checkpoint
from utils.builder import build_val_loader, build_test_loader, predict


def get_args():
    parser = argparse.ArgumentParser("FarMOS Validation / Test")
    parser.add_argument("--mode", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/semantic-kitti-mos.yaml")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    return parser.parse_args()


def save_predictions(pred_labels, num_valid, seq_id, file_id, pred_dir, inv_map):
    """Save prediction as .label file (SemanticKITTI format)."""
    seq_pred_dir = os.path.join(pred_dir, "sequences", seq_id, "predictions")
    os.makedirs(seq_pred_dir, exist_ok=True)

    pred_valid = pred_labels[:num_valid]
    original_labels = np.zeros_like(pred_valid, dtype=np.uint32)
    for mapped_cls, original_cls in inv_map.items():
        original_labels[pred_valid == mapped_cls] = original_cls

    original_labels.tofile(os.path.join(seq_pred_dir, f"{file_id}.label"))


def run_val(args, model, task_cfg):
    inv_map = task_cfg["learning_map_inv"]
    ignore = [cl for cl, ign in task_cfg["learning_ignore"].items() if ign]
    n_classes = len(inv_map)

    val_loader = build_val_loader(args.sequence_dir, args.num_workers)
    evaluator = iouEval(n_classes, ignore)
    range_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    range_evaluators = {r: iouEval(n_classes, ignore) for r in range_bins}

    print(f"Predicting + Evaluating ({len(val_loader.dataset)} samples)...")

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", dynamic_ncols=True):
            xyzi, des_coord, sph_coord, label_3d, _, num_valid, seq_ids, file_ids = batch
            xyzi, des_coord, sph_coord = xyzi.cuda(), des_coord.cuda(), sph_coord.cuda()

            pred_cls = predict(model, xyzi, des_coord, sph_coord)

            for b in range(pred_cls.shape[0]):
                nv = num_valid[b].item()
                sid, fid = seq_ids[b], file_ids[b]

                save_predictions(pred_cls[b], nv, sid, fid, args.pred_dir, inv_map)

                pred_v = pred_cls[b][:nv]
                gt_v = label_3d[b][:nv].numpy().astype(np.int32)
                evaluator.addBatch(pred_v, gt_v)

                # Range-wise evaluation
                pc_xyz = np.fromfile(
                    os.path.join(args.sequence_dir, sid, "velodyne", f"{fid}.bin"), dtype=np.float32
                ).reshape(-1, 4)[:nv, :3]
                depth = np.linalg.norm(pc_xyz, axis=1)

                for (rmin, rmax), reval in range_evaluators.items():
                    rmask = (depth >= rmin) & (depth < rmax)
                    if rmask.any():
                        reval.addBatch(pred_v[rmask], gt_v[rmask])

    # Print results
    iou = evaluator.getIoU()
    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    for i in range(n_classes):
        if i not in ignore:
            name = {0: "unlabeled", 1: "static", 2: "moving"}
            print(f"  IoU {name.get(i, i):>10s}: {iou[i]:.6f}")

    print("-" * 60)
    print("  Range-wise IoU:")
    print(f"  {'Range':>10s} | {'iou_static':>12s} | {'iou_moving':>12s}")
    print("-" * 44)
    for (rmin, rmax), reval in range_evaluators.items():
        r = reval.getIoU()
        print(f"  {rmin:>2d}-{rmax:>2d}m    | {r[1]:>12.6f} | {r[2]:>12.6f}")
    print("=" * 60)


def run_test(args, model, task_cfg):
    inv_map = task_cfg["learning_map_inv"]

    model.eval()
    with torch.no_grad():
        for seq_num in task_cfg["split"]["test"]:
            test_loader = build_test_loader(args.sequence_dir, seq_num, args.batch_size, args.num_workers)
            seq_id = str(seq_num).zfill(2)

            for batch in tqdm(test_loader, desc=f"Test seq {seq_id}", dynamic_ncols=True):
                xyzi, des_coord, sph_coord, num_valid, seq_ids, file_ids = batch
                xyzi, des_coord, sph_coord = xyzi.cuda(), des_coord.cuda(), sph_coord.cuda()

                pred_cls = predict(model, xyzi, des_coord, sph_coord)

                for b in range(pred_cls.shape[0]):
                    save_predictions(pred_cls[b], num_valid[b].item(), seq_ids[b], file_ids[b], args.pred_dir, inv_map)

    print(f"\nTest predictions saved to: {args.pred_dir}")


if __name__ == "__main__":
    args = get_args()

    with open(args.config) as f:
        task_cfg = yaml.load(f, Loader=yaml.FullLoader)

    model = FarMOS().cuda()
    ckpt = load_checkpoint(model, args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    if args.mode == "val":
        run_val(args, model, task_cfg)
    elif args.mode == "test":
        run_test(args, model, task_cfg)
