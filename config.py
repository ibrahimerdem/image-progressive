import os

# Files
DATA_DIR = "data"
INITIAL_DIR = os.path.join(DATA_DIR, "cropped_initial")
TARGET_DIR = os.path.join(DATA_DIR, "cropped_target")
TRAIN_CSV = os.path.join(DATA_DIR, "training_features_cross.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation_features_cross.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_features_cross.csv")

# Dataset
IMG_WIDTH = 128
IMG_HEIGHT = 128
TARGET_WIDTH = 256
TARGET_HEIGHT = 256
CHANNELS = 3
EMBEDDING_OUT_DIM = 512
FEATURE_COLUMNS = ["yarn_number", 
                   "frequency", 
                   "fabric_elasticity", 
                   "cielab_l_raw",
                   "cielab_a_raw", 
                   "cielab_b_raw", 
                   "bleaching", 
                   "duration", 
                   "concentration"]
FEATURE_NORMALIZATION = True
FEATURE_MINS = [5.6,
                15,
                0,
                0,
                -127,
                -127,
                1,
                1,
                5]
FEATURE_MAXS = [30,
                42,
                95,
                100,
                127,
                127,
                4,
                7,
                35]
TOTAL_FEATURE_DIM = len(FEATURE_COLUMNS)
INITIAL_IMAGE = True

# training defaults
NOISE_DIM = 128
FIXED_G_LR = 0.00001
FIXED_D_LR = 0.00001
BCE_FACTOR = 1.0
L1_FACTOR = 50.0
L2_FACTOR = 100.0
VAL_EPOCH = 10

# ddp
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)
BATCH_SIZE_PER_GPU = 8
NUM_WORKERS = 4