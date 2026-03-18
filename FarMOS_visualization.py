import argparse
import torch
from networks.MainNetwork import FarMOS
from datasets.dataloader import DataloadVal
from core.checkpoint import load_checkpoint
from core.projector_unprojector import project
from core.pretty_printer_and_saver import save_feature_as_img


def get_args():
    parser = argparse.ArgumentParser("FarMOS Visualization")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--frame_id", type=int, default=4017)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda")

    dataset = DataloadVal(args.sequence_dir)

    (
        pcd_input,
        rv_input,
        bev_coord,
        rv_coord,
        label_moving_3d,
        label_movable_rv,
        label_moving_bev,
        num_valid_t0,
        seq_id,
        file_id,
    ) = dataset[args.frame_id]

    # Add batch dimension
    pcd_input = pcd_input.unsqueeze(0).to(device)
    rv_input = rv_input.unsqueeze(0).to(device)
    bev_coord = bev_coord.unsqueeze(0).to(device)
    rv_coord = rv_coord.unsqueeze(0).to(device)
    label_moving_3d = label_moving_3d.unsqueeze(0).to(device)
    label_movable_rv = label_movable_rv.unsqueeze(0).to(device)
    label_moving_bev = label_moving_bev.unsqueeze(0).to(device)

    # Generate label views
    moving_label_bev = project(label_moving_3d.view(1, 1, -1, 1).float(), bev_coord[:, -1], view="bev")
    moving_label_rv = project(label_moving_3d.view(1, 1, -1, 1).float(), rv_coord[:, -1], view="rv")

    # Model
    model = FarMOS().to(device)
    model.eval()
    try:
        ckpt = load_checkpoint(model, args.checkpoint)
        print(f"Successfully Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("모델 구조가 달라 그냥 현재 구조로 진행합니다")

    with torch.no_grad():
        # Labels
        save_feature_as_img(
            [moving_label_bev, moving_label_rv, label_movable_rv, label_moving_bev.unsqueeze(1).float()],
            ["GT_moving_bev", "GT_moving_rv", "GT_movable_rv", "GT_moving_2d_bev"],
            "max",
        )
        print("Label saved.")

        # Inference + intermediate features
        output = model.infer(pcd_input, rv_input, bev_coord, rv_coord)
        tensors, names = zip(*output["visualization"])
        save_feature_as_img(list(tensors), list(names), "max")
        print("Feature saved.")
