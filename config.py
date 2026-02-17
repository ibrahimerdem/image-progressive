import os

# Files
DATA_DIR = "data"
INITIAL_DIR = os.path.join(DATA_DIR, "initial")
TARGET_DIR = os.path.join(DATA_DIR, "target")
TRAIN_CSV = os.path.join(DATA_DIR, "training_features.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation_features.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_features.csv")

# Dataset
IMG_WIDTH = 128
IMG_HEIGHT = 128
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
CHANNELS = 3
EMBEDDING_OUT_DIM = 512
FEATURE_COLUMNS = ["type", "replication"]
NUM_TYPES = 50
NUM_REPLICATIONS = 3
USE_EMBEDDING = True
INITIAL_IMAGE = True

# gan training defaults
NOISE_DIM = 128
FIXED_G_LR = 0.0001
FIXED_D_LR = 0.0001
BCE_FACTOR = 1.0
L1_FACTOR = 50.0
L2_FACTOR = 150.0
VAL_EPOCH = 5

# ddp
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)
BATCH_SIZE_PER_GPU = 8
NUM_WORKERS = 4