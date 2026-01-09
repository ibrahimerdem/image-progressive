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
TARGET_WIDTH = 256
TARGET_HEIGHT = 256
CHANNELS = 3
EMBEDDING_DIM = 128
EMBEDDING_OUT_DIM = 256
FEATURE_COLUMNS = ["yarn_number", "frequency", "fabric_elasticity", "cielab_l_raw", "cielab_a_raw", "cielab_b_raw", "bleaching", "duration", "concentration"]
FEATURE_NORMALIZATION = True
FEATURE_MINS = [5.6,15,0,16.87,-1.91,-16.75,1,1,1000]
FEATURE_MAXS = [30,42,95,40,1.62,-0.5,4,7,7000]
INITIAL_IMAGE = True
ENCODER_PATH = "checkpoints/encoder_epoch_50.pth"
FREEZE_ENCODER = True
# Stable diffusion initial image encoder options
SD_INITIAL_ENCODER_CKPT = "checkpoints/encoder_epoch_50.pth"
SD_FREEZE_INITIAL_ENCODER = False

# Model / training defaults
NOISE_DIM = 128
FIXED_G_LR = 0.0001
FIXED_D_LR = 0.0001
BCE_FACTOR = 1.0
L1_FACTOR = 50.0
L2_FACTOR = 150.0
VAL_EPOCH = 5

# Stable diffusion defaults
SD_LR = 0.0002
SD_TIMESTEPS = 1000
SD_SAMPLE_STEPS = 256
SD_EMB_DIM = 384
SD_VAL_STEPS = 1
SD_SAMPLE_BATCH = 2
SD_LOG_INTERVAL = 10
SD_BASE_CHANNELS = 96
SD_ATTENTION_HEADS = 4
SD_ATTENTION_RESOLUTION = [32, 16]
SD_EMA_DECAY = 0.9995
SD_GRAD_CLIP = 1.0

# Devices for training (indices as seen by the system)
# Used by run_train.sh and DDP world size.
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)

# Per-GPU batch size and dataloader workers
BATCH_SIZE_PER_GPU = 2
NUM_WORKERS = 4

