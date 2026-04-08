import os

RUN_NAME = "conditional_reverse_synthetic"
VERSION_NAME = "resnet50"

# Files
DATA_DIR = "data"
INITIAL_DIR = os.path.join(DATA_DIR, "initial")
TARGET_DIR = os.path.join(DATA_DIR, "target")
TRAIN_CSV = os.path.join(DATA_DIR, "training_features_reverse.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation_features.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_features.csv")
OUTPUT_DIR = os.path.join("outputs", RUN_NAME, VERSION_NAME)

# Dataset
IMG_WIDTH = 224
IMG_HEIGHT = 224
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
CHANNELS = 3
FEATURE_COLUMNS = ["yarn_number", "frequency", "fabric_elasticity", "cielab_l_raw", "cielab_a_raw", "cielab_b_raw"]
FEATURE_NORMALIZATION = True
FEATURE_MINS = [5.6,15,0,16.87,-1.91,-16.75]
FEATURE_MAXS = [30,42,95,40,1.62,-0.5]
TARGET_FEATURE_COLUMNS = ["concentration"] #["bleaching", "duration", "concentration"]
TARGET_MINS = [1000] #[1,1,1000]
TARGET_MAXS = [7000] #[4,7,7000]

KNOWN_TARGET_COLUMNS = ["bleaching", "duration"]
KNOWN_TARGET_MINS = [1, 1]
KNOWN_TARGET_MAXS = [4, 7]
INITIAL_IMAGE = False
IMAGE_ENCODER = "resnet50"
FREEZE_BACKBONE = True

# training defaults
LR = 0.001
GRAD_CLIP = 0.5
LOG_INTERVAL = 100
VAL_INTERVAL = 10

# ddp
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)
DDP_TIMEOUT = 30
BATCH_SIZE_PER_GPU = 8
NUM_WORKERS = 4