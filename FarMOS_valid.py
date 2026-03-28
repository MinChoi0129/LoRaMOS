import argparse
import os
import numpy as np
import yaml
import torch
from tqdm import tqdm

from networks.MainNetwork import FarMOS
from core.checkpoint import load_checkpoint
from core.builder import build_val_loader, build_test_loader


def get_args():
    parser = argparse.ArgumentParser("FarMOS Prediction (save .label files)")
    parser.add_argument("--mode", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/semantic-kitti-mos.yaml")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    return parser.parse_args()


def save_predictions(pred_labels, num_valid, seq_id, file_id, pred_dir, inv_map):
    seq_pred_dir = os.path.join(pred_dir, "sequences", seq_id, "predictions")
    os.makedirs(seq_pred_dir, exist_ok=True)

    pred_valid = pred_labels[:num_valid]
    original_labels = np.zeros_like(pred_valid, dtype=np.uint32)
    for mapped_cls, original_cls in inv_map.items():
        original_labels[pred_valid == mapped_cls] = original_cls

    original_labels.tofile(os.path.join(seq_pred_dir, f"{file_id}.label"))


def run_predict(args, model, task_cfg, loader, desc):
    inv_map = task_cfg["learning_map_inv"]

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
            if args.mode == "val":
                pcd_input, rv_input, bev_coord, rv_coord, _, _, _, _, num_valid, seq_ids, file_ids = batch
            else:
                pcd_input, rv_input, bev_coord, rv_coord, num_valid, seq_ids, file_ids = batch

            pcd_input, rv_input = pcd_input.cuda(), rv_input.cuda()
            bev_coord, rv_coord = bev_coord.cuda(), rv_coord.cuda()

            output = model.infer(pcd_input, rv_input, bev_coord, rv_coord)
            pred_cls = output["moving_logit_3d"].squeeze(-1).argmax(dim=1).cpu().numpy()

            for b in range(pred_cls.shape[0]):
                save_predictions(pred_cls[b], num_valid[b].item(), seq_ids[b], file_ids[b], args.pred_dir, inv_map)


if __name__ == "__main__":
    args = get_args()

    with open(args.config) as f:
        task_cfg = yaml.load(f, Loader=yaml.FullLoader)

    model = FarMOS().cuda()
    ckpt = load_checkpoint(model, args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    if args.mode == "val":
        loader = build_val_loader(args.sequence_dir, args.config, args.num_workers)
        run_predict(args, model, task_cfg, loader, "Predicting (val)")
    elif args.mode == "test":
        for seq_num in task_cfg["split"]["test"]:
            loader = build_test_loader(args.sequence_dir, seq_num, args.batch_size, args.num_workers)
            run_predict(args, model, task_cfg, loader, f"Predicting (test seq {seq_num:02d})")

    print(f"Predictions saved to: {args.pred_dir}")
