import argparse
import torch
from networks.MainNetwork import FarMOS
from datasets.dataloader import DataloadVal
from utils.checkpoint import load_checkpoint
from utils.projector_unprojector import project_to_bev, project_to_rv, unprojectors
from utils.pretty_printer_and_saver import save_feature_as_img


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
    # dataset = DataloadTrain(args.sequence_dir)
    dataset = DataloadVal(args.sequence_dir)

    (
        xyzi,  # [T, 7, N, 1]
        des_coord,  # [T, N, 3, 1]
        sph_coord,  # [T, N, 3, 1]
        rv_input,  # [5, 64, 2048]
        moving_label_3d,  # [N]
        movable_label_2d,  # [64, 2048]
        num_valid_t0,  # int
        current_seq_id,  # str
        current_file_id,  # str
    ) = dataset[args.frame_id]

    # Add batch dimension
    xyzi = xyzi.unsqueeze(0).to(device)
    des_coord = des_coord.unsqueeze(0).to(device)
    sph_coord = sph_coord.unsqueeze(0).to(device)
    rv_input = rv_input.unsqueeze(0).to(device)
    moving_label_3d = moving_label_3d.unsqueeze(0).to(device)
    movable_label_rv = movable_label_2d.unsqueeze(0).to(device)

    # Generate else label
    moving_label_bev = project_to_bev(moving_label_3d.view(1, 1, -1, 1), des_coord[:, -1:])
    moving_label_rv = project_to_rv(moving_label_3d.view(1, 1, -1, 1), sph_coord[:, -1])
    movable_label_3d = unprojectors["full"](movable_label_rv.unsqueeze(1).float(), sph_coord[:, -1, :, :2, :].flip(-2))
    movable_label_bev = project_to_bev(movable_label_3d, des_coord[:, -1:])

    # Model
    model = FarMOS().to(device)
    model.eval()
    try:
        ckpt = load_checkpoint(model, args.checkpoint)
        print(f"Successfully Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")
    except:
        print("모델 구조가 달라 그냥 현재 구조로 진행합니다")

    with torch.no_grad():
        save_feature_as_img(
            [moving_label_bev, moving_label_rv, movable_label_bev, movable_label_rv],
            ["moving_label_bev", "moving_label_rv", "movable_label_bev", "movable_label_rv"],
            "max",
        )
        print("Label Saved.")
        model.infer(xyzi, des_coord, sph_coord, rv_input)
        print("Feature Saved.")
