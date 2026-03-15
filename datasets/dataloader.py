"""Train / Val / Test Dataset 클래스."""

import os
import yaml
import numpy as np
from torch.utils.data import Dataset

from datasets.config import TASK_CONFIG_PATH, STATIC_FRAMES_PATH, OBJECT_BANK_DIR
from datasets.augmentation import DataAugment, SequenceCopyPaste
from datasets.pointcloud import parse_calibration, parse_poses
from datasets.preprocessing import (
    build_sequence_filelist,
    load_sequence,
    pad_to_max,
    build_input_tensors,
    build_label_tensors,
)


# ============================================================
# Dataset: Train
# ============================================================


class DataloadTrain(Dataset):
    def __init__(self, sequence_dir):
        self.sequence_dir = sequence_dir
        self.flist = []

        with open(TASK_CONFIG_PATH, "r") as f:
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.movable_learning_map = self.task_cfg["movable_learning_map"]
        self.augmentor = DataAugment()
        self.copy_paste = SequenceCopyPaste(OBJECT_BANK_DIR, paste_max_obj_num=3)

        train_sequences = [str(s).zfill(2) for s in self.task_cfg["split"]["train"]]
        for seq_id in train_sequences:
            seq_path = os.path.join(sequence_dir, seq_id)
            calib = parse_calibration(os.path.join(seq_path, "calib.txt"))
            poses = parse_poses(os.path.join(seq_path, "poses.txt"), calib)
            self.flist.extend(build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=True))

        print(f"[Train] Before static removal: {len(self.flist)} samples")
        self._remove_static_frames()
        print(f"[Train] After static removal:  {len(self.flist)} samples")

    def _remove_static_frames(self):
        """dynamic point가 존재하는 프레임만 유지 (txt 파일 기반 필터링)"""
        if not os.path.exists(STATIC_FRAMES_PATH):
            print(f"  Warning: {STATIC_FRAMES_PATH} not found, skipping static removal.")
            return

        with open(STATIC_FRAMES_PATH, "r") as f:
            lines = f.readlines()

        keep_set = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            seq_id, file_id, _ = parts
            if seq_id not in keep_set:
                keep_set[seq_id] = set()
            keep_set[seq_id].add(file_id)

        new_flist = []
        for meta_list in self.flist:
            current_seq_id = meta_list[-1][3]
            current_file_id = meta_list[-1][4]
            if current_seq_id in keep_set and current_file_id in keep_set[current_seq_id]:
                new_flist.append(meta_list)

        self.flist = new_flist

    def __getitem__(self, index):
        meta_list = self.flist[index]

        point_clouds, label_list, movable_label_list, raw_semantic_list = load_sequence(
            meta_list, self.task_cfg["learning_map"], self.movable_learning_map
        )
        point_clouds, label_list, movable_label_list = self.copy_paste(
            point_clouds, label_list, movable_label_list, raw_semantic_list
        )
        point_clouds, label_list, movable_label_list, valid_point_counts = pad_to_max(
            point_clouds, label_list, movable_label_list
        )

        xyzi, bev_coord, rv_coord, spherical_coords_raw, rv_input = build_input_tensors(point_clouds, augmentor=self.augmentor)
        num_valid_t0 = valid_point_counts[-1]

        label_3d, label_2d = build_label_tensors(label_list, spherical_coords_raw, movable_label_list)

        return xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid_t0

    def __len__(self):
        return len(self.flist)


# ============================================================
# Dataset: Validation
# ============================================================


class DataloadVal(Dataset):
    def __init__(self, sequence_dir):
        self.sequence_dir = sequence_dir
        self.flist = []

        with open(TASK_CONFIG_PATH, "r") as f:
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.movable_learning_map = self.task_cfg["movable_learning_map"]

        val_sequences = [str(s).zfill(2) for s in self.task_cfg["split"]["valid"]]
        for seq_id in val_sequences:
            seq_path = os.path.join(sequence_dir, seq_id)
            calib = parse_calibration(os.path.join(seq_path, "calib.txt"))
            poses = parse_poses(os.path.join(seq_path, "poses.txt"), calib)
            self.flist.extend(build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=True))

        print(f"[Val] Total samples: {len(self.flist)}")

    def __getitem__(self, index):
        meta_list = self.flist[index]

        point_clouds, label_list, movable_label_list, _ = load_sequence(
            meta_list, self.task_cfg["learning_map"], self.movable_learning_map
        )
        point_clouds, label_list, movable_label_list, valid_point_counts = pad_to_max(
            point_clouds, label_list, movable_label_list
        )
        xyzi, bev_coord, rv_coord, spherical_coords_raw, rv_input = build_input_tensors(point_clouds)
        moving_label_3d, movable_label_2d = build_label_tensors(label_list, spherical_coords_raw, movable_label_list)

        num_valid_t0 = valid_point_counts[-1]
        current_seq_id = meta_list[-1][3]
        current_file_id = meta_list[-1][4]

        return (
            xyzi,
            bev_coord,
            rv_coord,
            rv_input,
            moving_label_3d,
            movable_label_2d,
            num_valid_t0,
            current_seq_id,
            current_file_id,
        )

    def __len__(self):
        return len(self.flist)


# ============================================================
# Dataset: Test
# ============================================================


class DataloadTest(Dataset):
    def __init__(self, sequence_dir, target_sequence):
        self.sequence_dir = sequence_dir
        self.flist = []

        seq_id = str(target_sequence).zfill(2)
        seq_path = os.path.join(sequence_dir, seq_id)
        calib = parse_calibration(os.path.join(seq_path, "calib.txt"))
        poses = parse_poses(os.path.join(seq_path, "poses.txt"), calib)
        self.flist = build_sequence_filelist(sequence_dir, seq_id, poses, include_labels=False)

        print(f"[Test] Sequence {seq_id}: {len(self.flist)} samples")

    def __getitem__(self, index):
        meta_list = self.flist[index]

        point_clouds, _, _, _ = load_sequence(meta_list)
        point_clouds, valid_point_counts = pad_to_max(point_clouds)
        xyzi, bev_coord, rv_coord, _, rv_input = build_input_tensors(point_clouds)

        num_valid_t0 = valid_point_counts[-1]
        current_seq_id = meta_list[-1][2]
        current_file_id = meta_list[-1][3]

        return xyzi, bev_coord, rv_coord, rv_input, num_valid_t0, current_seq_id, current_file_id

    def __len__(self):
        return len(self.flist)
