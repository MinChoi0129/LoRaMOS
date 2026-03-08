TASK_CONFIG_PATH = "config/semantic-kitti-mos.yaml"
STATIC_FRAMES_PATH = "config/train_split_dynamic_pointnumber.txt"

MAX_POINTS = 160000
NUM_TEMPORAL_FRAMES = 5

BEV_RANGE_X = (-50.0, 50.0)
BEV_RANGE_Y = (-50.0, 50.0)
BEV_RANGE_Z = (-4.0, 2.0)
BEV_GRID_SIZE = (512, 512, 30)

RV_RANGE_PHI = (-180.0, 180.0)
RV_RANGE_THETA = (-25.0, 3.0)
RV_RANGE_R = (2.0, 50.0)
RV_GRID_SIZE = (64, 2048, 30)
