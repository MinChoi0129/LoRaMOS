import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from networks.MainNetwork import LoRaMOS
from datasets.dataloader import DataloadVal


def get_args():
    parser = argparse.ArgumentParser("LoRaMOS Speed Benchmark")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/semantic-kitti-mos.yaml")
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--num_iters", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda")
    dataset = DataloadVal(args.sequence_dir, args.config)
    pcd_input, rv_input, bev_coord, rv_coord, *_ = dataset[4017]
    pcd_input = pcd_input.unsqueeze(0).to(device)
    rv_input = rv_input.unsqueeze(0).to(device)
    bev_coord = bev_coord.unsqueeze(0).to(device)
    rv_coord = rv_coord.unsqueeze(0).to(device)

    model = LoRaMOS().to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    for _ in tqdm(range(args.warmup_iters), desc="Warmup"):
        with torch.no_grad():
            model.infer(pcd_input, rv_input, bev_coord, rv_coord)
    torch.cuda.synchronize()

    times = []
    pbar = tqdm(range(args.num_iters), desc="Benchmark")
    for i in pbar:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            model.infer(pcd_input, rv_input, bev_coord, rv_coord)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)
        pbar.set_postfix(mean=f"{np.mean(times):.2f}ms")

    times = np.array(times)
    print(f"\nResult ({args.num_iters} iters): {times.mean():.2f} +/- {times.std():.2f} ms")
