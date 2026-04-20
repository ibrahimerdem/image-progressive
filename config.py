import os

RUN_NAME = "vae"
VERSION_NAME = "load_and_retrain"

# Files
DATA_DIR = "data"
TARGET_DIR = os.path.join(DATA_DIR, "target")
TRAIN_CSV = os.path.join(DATA_DIR, "defect_free.csv")
VAL_CSV = os.path.join(DATA_DIR, "defected_validation.csv")

# Dataset
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
CHANNELS = 3

# vae
VAE_SCALE = 0.17146
VAE_CKPT = "checkpoints/diffusion/diffusion_pytorch_model.bin"
VAE_RETRAIN_DIR = "checkpoints/vae_retrain"
FREEZE_VAE = False

# training
LR = 1e-5
KL_WEIGHT   = 0.001   # regularises latent toward N(0,1); 1e-6 is too weak (copies defects through)
SSIM_WEIGHT = 0.5     # structural similarity loss alongside L1; keeps fabric texture sharp
VAL_EPOCH = 5
SAVE_EVERY = 10

# ddp
DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)
BATCH_SIZE_PER_GPU = 4
NUM_WORKERS = 4