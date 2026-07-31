import os

RUN_NAME = "diffusion"
VERSION_NAME = "regional_t1_s50"

# Files
DATA_DIR = "data"
INITIAL_DIR = os.path.join(DATA_DIR, "local_initial")
TARGET_DIR = os.path.join(DATA_DIR, "local_target")
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

# diffusion defaults
LR = 0.00005
TIMESTEPS = 1000      
SAMPLE_STEPS = 50     
EMB_DIM = 768
BASE_CHANNELS = 256
VAL_EPOCH = 20
SAMPLE_TEMPERATURE = 1.0
SAMPLER_ETA = 1.0
LOG_INTERVAL = 100   
ATTENTION_HEADS = 4
EMA_DECAY = 0.999
GRAD_CLIP = 1.0
DDP_TIMEOUT_MINUTES = 30

# vae
VAE_SCALE = 0.2372
VAE_CKPT = "checkpoints/diffusion/diffusion_pytorch_model.bin"
FREEZE_VAE = False

# ddp
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)
BATCH_SIZE_PER_GPU = 4
NUM_WORKERS = 4