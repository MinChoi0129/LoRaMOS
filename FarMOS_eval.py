import argparse
import os
import glob
import numpy as np
import yaml
from tqdm import tqdm

from utils.metrics import iouEval


def get_args():
    parser = argparse.ArgumentParser("FarMOS Evaluation (pred vs GT)")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/semantic-kitti-mos.yaml")
    parser.add_argument("--sequences", type=int, nargs="+", default=[8])
    parser.add_argument("--range_max", type=int, default=50)
    parser.add_argument("--range_step", type=int, default=10)
    return parser.parse_args()


def load_labels(path, learning_map):
    raw = np.fromfile(path, dtype=np.uint32)
    sem_label = raw & 0xFFFF
    mapped = np.zeros_like(sem_label, dtype=np.int32)
    for orig, target in learning_map.items():
        mapped[sem_label == orig] = target
    return mapped


def load_pointcloud(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def evaluate(args, task_cfg):
    learning_map = task_cfg["learning_map"]
    ignore = [cl for cl, ign in task_cfg["learning_ignore"].items() if ign]
    n_classes = len(task_cfg["learning_map_inv"])

    # Range bins
    range_bins = [(i, i + args.range_step) for i in range(0, args.range_max, args.range_step)]

    overall_eval = iouEval(n_classes, ignore)
    range_evals = {r: iouEval(n_classes, ignore) for r in range_bins}

    for seq_num in args.sequences:
        seq_id = f"{seq_num:02d}"
        pred_seq_dir = os.path.join(args.pred_dir, "sequences", seq_id, "predictions")
        gt_label_dir = os.path.join(args.sequence_dir, seq_id, "labels")
        velodyne_dir = os.path.join(args.sequence_dir, seq_id, "velodyne")

        pred_files = sorted(glob.glob(os.path.join(pred_seq_dir, "*.label")))
        if not pred_files:
            print(f"No predictions found for seq {seq_id} in {pred_seq_dir}")
            continue

        for pred_path in tqdm(pred_files, desc=f"Eval seq {seq_id}", dynamic_ncols=True):
            file_id = os.path.splitext(os.path.basename(pred_path))[0]
            gt_path = os.path.join(gt_label_dir, f"{file_id}.label")
            pc_path = os.path.join(velodyne_dir, f"{file_id}.bin")

            pred = load_labels(pred_path, learning_map)
            gt = load_labels(gt_path, learning_map)

            # pred may be shorter than gt (padded points excluded during save)
            n = min(len(pred), len(gt))
            pred, gt = pred[:n], gt[:n]

            overall_eval.addBatch(pred, gt)

            # Range-wise
            xyz = load_pointcloud(pc_path)[:n]
            depth = np.linalg.norm(xyz, axis=1)
            for (rmin, rmax), reval in range_evals.items():
                mask = (depth >= rmin) & (depth < rmax)
                if mask.any():
                    reval.addBatch(pred[mask], gt[mask])

    # Print results
    _, overall_iou = overall_eval.getIoU()
    name = {1: "static", 2: "moving"}

    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  Sequences: {[f'{s:02d}' for s in args.sequences]}")
    print()
    print("  [Overall]")
    for i in range(n_classes):
        if i not in ignore:
            print(f"    IoU {name.get(i, i):>10s}: {overall_iou[i].item():.6f}")

    print()
    print("  [Range-wise Moving IoU]")
    print(f"  {'Range':>10s} | {'static':>12s} | {'moving':>12s} | {'points':>10s}")
    print("  " + "-" * 52)
    for (rmin, rmax), reval in range_evals.items():
        tp, fp, fn = reval.getStats()
        n_points = int(reval.conf_matrix.sum())
        _, r = reval.getIoU()
        print(f"  {rmin:>2d}-{rmax:>2d}m    | {r[1].item():>12.6f} | {r[2].item():>12.6f} | {n_points:>10d}")
    print("=" * 60)


if __name__ == "__main__":
    args = get_args()

    with open(args.config) as f:
        task_cfg = yaml.load(f, Loader=yaml.FullLoader)

    evaluate(args, task_cfg)
