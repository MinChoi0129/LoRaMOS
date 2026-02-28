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
    xyzi, des_coord, sph_coord, _, _, _, _, _ = dataset[4017]

    # Add batch dimension
    xyzi = xyzi.unsqueeze(0).to(device)
    des_coord = des_coord.unsqueeze(0).to(device)
    sph_coord = sph_coord.unsqueeze(0).to(device)

    print(f"xyzi: {xyzi.shape}, des_coord: {des_coord.shape}, sph_coord: {sph_coord.shape}")

    # Model
    model = FarMOS().to(device)
    model.eval()

    # Warmup
    for _ in tqdm(range(args.warmup_iters), desc="Warmup"):
        with torch.no_grad():
            model.infer(xyzi, des_coord, sph_coord)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    pbar = tqdm(range(args.num_iters), desc="Benchmark")
    for i in pbar:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            model.infer(xyzi, des_coord, sph_coord)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)
        pbar.set_postfix(mean=f"{np.mean(times):.2f}ms")

    times = np.array(times)
    print(f"\nResult ({args.num_iters} iters): {times.mean():.2f} +/- {times.std():.2f} ms")
