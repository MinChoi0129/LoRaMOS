TASK_CONFIG_PATH = "config/semantic-kitti-mos.yaml"
STATIC_FRAMES_PATH = "config/train_split_dynamic_pointnumber.txt"
OBJECT_BANK_DIR = "/home/ssd_data/ROOT_KITTI/object_bank_semkitti/"

MAX_POINTS = 160000
NUM_TEMPORAL_FRAMES = 3

BEV_RANGE_X = (-50.0, 50.0)
BEV_RANGE_Y = (-50.0, 50.0)
BEV_RANGE_Z = (-4.0, 2.0)
BEV_GRID_SIZE = (512, 512, 30)  # (H, W, depth)

RV_RANGE_PHI = (-180.0, 180.0)
RV_RANGE_THETA = (-25.0, 3.0)
RV_RANGE_R = (2.0, 50.0)
RV_GRID_SIZE = (64, 2048, 50)  # (H, W, depth)

RANGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
