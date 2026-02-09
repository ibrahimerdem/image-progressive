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
FEATURE_COLUMNS = ["yarn_number", "frequency", "fabric_elasticity", "cielab_l_raw", "cielab_a_raw", "cielab_b_raw", "bleaching", "duration", "concentration"]
FEATURE_NORMALIZATION = True
FEATURE_MINS = [5.6,15,0,16.87,-1.91,-16.75,1,1,1000]
FEATURE_MAXS = [30,42,95,40,1.62,-0.5,4,7,7000]
INITIAL_IMAGE = True

# Stable diffusion defaults
SD_LR = 0.00005
SD_TIMESTEPS = 1000      
SD_SAMPLE_STEPS = 200     
SD_EMB_DIM = 768
SD_BASE_CHANNELS = 128
SD_VAL_STEPS = 50
SD_VAL_EPOCH = 10
SD_SAMPLE_BATCH = 2
SD_LOG_INTERVAL = 100   
SD_ATTENTION_HEADS = 2
SD_EMA_DECAY = 0.9995
SD_GRAD_CLIP = 0.5
SD_DDP_TIMEOUT_MINUTES = 30
SD_VAE_CKPT = "checkpoints/vae_epoch_70.pth"
SD_FREEZE_VAE = False

# ddp
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)
BATCH_SIZE_PER_GPU = 4
NUM_WORKERS = 4