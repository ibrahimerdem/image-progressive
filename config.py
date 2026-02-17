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
FEATURE_COLUMNS = ["yarn_number", "frequency", "fabric_elasticity", "cielab_l_raw", "cielab_a_raw", "cielab_b_raw", "bleaching", "duration", "concentration"]
FEATURE_NORMALIZATION = True
FEATURE_MINS = [5.6,15,0,16.87,-1.91,-16.75,1,1,1000]
FEATURE_MAXS = [30,42,95,40,1.62,-0.5,4,7,7000]
CATEGORICAL_FEATURES = ["coloring_type", "knitting"]
CATEGORICAL_DIMS = [5, 6]
TOTAL_FEATURE_DIM = len(FEATURE_COLUMNS) + sum(CATEGORICAL_DIMS)
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