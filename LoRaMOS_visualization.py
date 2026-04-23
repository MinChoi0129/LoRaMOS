import argparse
import torch
from networks.MainNetwork import LoRaMOS
from datasets.dataloader import DataloadVal, DataloadTest
from core.checkpoint import load_checkpoint
from core.projector_unprojector import project
from core.pretty_printer_and_saver import save_feature_as_img


def get_args():
    parser = argparse.ArgumentParser("LoRaMOS Visualization")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/semantic-kitti-mos.yaml")
    parser.add_argument("--mode", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--frame_id", type=int, default=4017)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda")

    model = LoRaMOS().to(device)
    model.eval()
    try:
        ckpt = load_checkpoint(model, args.checkpoint)
        print(f"Successfully Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Model structure mismatch, proceeding with current structure")

    with torch.no_grad():
        if args.mode == "val":
            dataset = DataloadVal(args.sequence_dir, args.config)
            sample = dataset[args.frame_id]
            pcd_input, rv_input, bev_coord, rv_coord, label_moving_3d, label_movable_3d, label_movable_rv, label_moving_bev = [
                t.unsqueeze(0).to(device) for t in sample[:8]
            ]

            moving_label_bev = project(label_moving_3d.view(1, 1, -1, 1).float(), bev_coord[:, -1], view="bev")
            movable_label_bev = project(label_movable_3d.view(1, 1, -1, 1).float(), bev_coord[:, -1], view="bev")
            moving_label_rv = project(label_moving_3d.view(1, 1, -1, 1).float(), rv_coord[:, -1], view="rv")

            save_feature_as_img(
                [
                    (moving_label_bev, "GT_moving_bev"),
                    (moving_label_rv, "GT_moving_rv"),
                    (label_movable_rv.unsqueeze(1).float(), "GT_movable_rv"),
                    (label_moving_bev.unsqueeze(1).float(), "GT_moving_2d_bev"),
                    (movable_label_bev, "GT_movable_bev"),
                ],
                channel_pool="max",
            )
            print("Label saved.")

        else:  # Test split
            dataset = DataloadTest(args.sequence_dir, args.seq_id)
            sample = dataset[args.frame_id]
            pcd_input, rv_input, bev_coord, rv_coord = [
                t.unsqueeze(0).to(device) for t in sample[:4]
            ]

        output = model.infer(pcd_input, rv_input, bev_coord, rv_coord)
        save_feature_as_img(output["visualization"], channel_pool="max")
        print("Feature saved.")
