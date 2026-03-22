import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from networks.MainNetwork import FarMOS
from datasets.dataloader import DataloadVal


def get_args():
    parser = argparse.ArgumentParser("FarMOS Speed Benchmark")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--num_iters", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda")

    # Dataset
    dataset = DataloadVal(args.sequence_dir)

    # Load real data once from dataloader
    print("Loading sample from dataloader...")
    pcd_input, rv_input, bev_coord, rv_coord, *_ = dataset[4017]

    # Add batch dimension
    pcd_input = pcd_input.unsqueeze(0).to(device)
    rv_input = rv_input.unsqueeze(0).to(device)
    bev_coord = bev_coord.unsqueeze(0).to(device)
    rv_coord = rv_coord.unsqueeze(0).to(device)

    # Model
    model = FarMOS().to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Warmup
    for _ in tqdm(range(args.warmup_iters), desc="Warmup"):
        with torch.no_grad():
            model.infer(pcd_input, rv_input, bev_coord, rv_coord)
    torch.cuda.synchronize()

    # Benchmark
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
