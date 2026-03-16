import argparse
import torch
from networks.MainNetwork import FarMOS
from datasets.dataloader import DataloadVal
from utils.checkpoint import load_checkpoint
from utils.projector_unprojector import project, unproject
from utils.pretty_printer_and_saver import save_feature_as_img


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
        xyzi,
        bev_coord,
        rv_coord,
        rv_input,
        moving_label_3d,
        movable_label_2d,
        num_valid_t0,
        current_seq_id,
        current_file_id,
    ) = dataset[args.frame_id]

    # Add batch dimension
    xyzi = xyzi.unsqueeze(0).to(device)
    bev_coord = bev_coord.unsqueeze(0).to(device)
    rv_coord = rv_coord.unsqueeze(0).to(device)
    rv_input = rv_input.unsqueeze(0).to(device)
    moving_label_3d = moving_label_3d.unsqueeze(0).to(device)
    movable_label_rv = movable_label_2d.unsqueeze(0).to(device)

    # Generate label views
    moving_label_bev = project(moving_label_3d.view(1, 1, -1, 1), bev_coord[:, -1], view="bev")
    moving_label_rv = project(moving_label_3d.view(1, 1, -1, 1), rv_coord[:, -1], view="rv")

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
            [moving_label_bev, moving_label_rv, movable_label_rv],
            ["GT_moving_bev", "GT_moving_rv", "GT_movable_rv"],
            "max",
        )
        print("Label saved.")

        # Inference + intermediate features
        output = model.infer(xyzi, bev_coord, rv_coord, rv_input)
        tensors, names = zip(*output["visualization"])
        save_feature_as_img(list(tensors), list(names), "max")
        print("Feature saved.")
