import argparse
import os
import time
from urllib.parse import quote

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from tqdm import tqdm

from core.checkpoint import load_checkpoint
from datasets.pointcloud import parse_calibration, parse_poses
from datasets.preprocessing import (
    build_input_tensors,
    build_sequence_filelist,
    load_sequence,
    pad_to_max,
)
from networks.MainNetwork import LoRaMOS


STATIC_COLOR = (150, 150, 150)   # Gray
MOVING_COLOR = (150, 230, 100)   # Light green
IGNORE_COLOR = (150, 80, 200)    # Purple
INPUT_COLOR = (220, 220, 220)    # Near-white (input pcd tab only)


def get_args():
    parser = argparse.ArgumentParser("LoRaMOS Rerun Visualization")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--config", type=str, default="config/semantic-kitti-mos.yaml")
    parser.add_argument("--web_port", type=int, default=9090)
    parser.add_argument("--grpc_port", type=int, default=9876)
    parser.add_argument("--max_points", type=int, default=32000,
                        help="Random-subsample points per frame to this cap (0 = no cap).")
    return parser.parse_args()


def colorize(labels):
    colors = np.tile(np.array(IGNORE_COLOR, dtype=np.uint8), (len(labels), 1))
    colors[labels == 1] = STATIC_COLOR
    colors[labels == 2] = MOVING_COLOR
    return colors


def main():
    args = get_args()
    device = torch.device("cuda")

    rr.init("LoRaMOS", spawn=False)
    grpc_uri = rr.serve_grpc(grpc_port=args.grpc_port, server_memory_limit="2GB")
    rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=grpc_uri)

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(name="Camera (image_2)", origin="/camera"),
                rrb.Spatial3DView(name="Pred", origin="/pred"),
                row_shares=[1, 3],
            ),
            collapse_panels=True,
        )
    )

    viewer_url = f"http://localhost:{args.web_port}/?url={quote(grpc_uri, safe='')}"
    print()
    print("=" * 70)
    print("[Rerun] Servers up. Before pressing Enter, do the following:")
    print(f"  1. In VSCode 'Ports' panel, forward port {args.grpc_port} (9090 is auto).")
    print("  2. Open this URL in your host browser:")
    print(f"     {viewer_url}")
    print("  3. Wait for viewer to show 'Connected'.")
    print("  4. Press Enter here to start live streaming (no data before Enter).")
    print("     Ctrl+C to abort.")
    print("=" * 70)
    try:
        input()
    except EOFError:
        pass

    seq_id = f"{args.sequence:02d}"
    seq_path = os.path.join(args.sequence_dir, seq_id)
    image_dir = os.path.join(seq_path, "image_2")
    has_image = os.path.isdir(image_dir) and len(os.listdir(image_dir)) > 0
    print(f"[Seq {seq_id}] image_2 {'available' if has_image else 'missing'}")

    calib = parse_calibration(os.path.join(seq_path, "calib.txt"))
    poses = parse_poses(os.path.join(seq_path, "poses.txt"), calib)
    flist = build_sequence_filelist(args.sequence_dir, seq_id, poses, include_labels=False)
    print(f"[Seq {seq_id}] {len(flist)} frames")

    model = LoRaMOS().to(device)
    model.eval()
    ckpt = load_checkpoint(model, args.checkpoint)
    print(f"[Model] Loaded: {args.checkpoint} (epoch {ckpt['epoch']})")

    with torch.no_grad():
        for frame_idx, meta_list in enumerate(tqdm(flist, desc="Computing", dynamic_ncols=True)):
            file_id = meta_list[-1][3]
            point_clouds, _, _, _ = load_sequence(meta_list)
            point_clouds, valid_counts = pad_to_max(point_clouds)

            pcd_input, bev_coord, rv_coord, _, rv_input = build_input_tensors(point_clouds)
            num_valid = valid_counts[-1]

            pcd_input_b = pcd_input.unsqueeze(0).to(device, non_blocking=True)
            rv_input_b = rv_input.unsqueeze(0).to(device, non_blocking=True)
            bev_coord_b = bev_coord.unsqueeze(0).to(device, non_blocking=True)
            rv_coord_b = rv_coord.unsqueeze(0).to(device, non_blocking=True)

            output = model.infer(pcd_input_b, rv_input_b, bev_coord_b, rv_coord_b)
            pred = output["moving_logit_3d"].squeeze(-1).squeeze(0).argmax(dim=0).cpu().numpy()

            xyz = pcd_input[-1, :3, :, 0].numpy().T  # [N, 3]
            xyz_valid = xyz[:num_valid]
            pred_valid = pred[:num_valid]

            if args.max_points and num_valid > args.max_points:
                stride = num_valid // args.max_points
                xyz_valid = xyz_valid[::stride][: args.max_points]
                pred_valid = pred_valid[::stride][: args.max_points]

            rr.set_time("frame", sequence=frame_idx)

            if has_image:
                img_path = os.path.join(image_dir, f"{file_id}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rr.log("camera/image", rr.Image(img))

            rr.log("pred/points", rr.Points3D(xyz_valid, colors=colorize(pred_valid)))

    print("[Rerun] Streaming finished. Viewer stays alive; Ctrl+C to exit.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
