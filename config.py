import os

# Files
DATA_DIR = "data"
INITIAL_DIR = os.path.join(DATA_DIR, "initial")
TARGET_DIR = os.path.join(DATA_DIR, "target")
TRAIN_CSV = os.path.join(DATA_DIR, "training_features.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation_features.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_features.csv")

# Dataset
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
CHANNELS = 3

# training
VAE_LR = 0.0001
VAE_KL_WEIGHT = 0.0001
VAL_EPOCH = 10

# DDP
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)
BATCH_SIZE_PER_GPU = 4
NUM_WORKERS = 4
