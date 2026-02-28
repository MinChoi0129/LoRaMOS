import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from networks.MainNetwork import FarMOS
from datasets.dataloader import DataloadVal
from utils.checkpoint import load_checkpoint
from utils.pretty_print_and_pretty_image import save_feature_as_img


def get_args():
    parser = argparse.ArgumentParser("FarMOS Speed Benchmark")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--frame_id", type=int, default=4017)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda")

    # Dataset
    dataset = DataloadVal(args.sequence_dir)

    # Load real data once from dataloader
    print("Loading sample from dataloader...")
    xyzi, des_coord, sph_coord, _, label_2d, _, _, _ = dataset[args.frame_id]

    # Add batch dimension
    xyzi = xyzi.unsqueeze(0).to(device)
    des_coord = des_coord.unsqueeze(0).to(device)
    sph_coord = sph_coord.unsqueeze(0).to(device)
    label_2d = label_2d.unsqueeze(0).to(device)

    print(
        f"xyzi: {xyzi.shape}, des_coord: {des_coord.shape}, sph_coord: {sph_coord.shape}, label_2d: {label_2d.shape}"
    )

    # Model
    model = FarMOS().to(device)
    model.eval()
    ckpt = load_checkpoint(model, args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    with torch.no_grad():
        save_feature_as_img([label_2d], ["label_2d"], "max")
        model.infer(xyzi, des_coord, sph_coord)

    print("Image saved in 'images'.")
