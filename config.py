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
EMBEDDING_DIM = 128
EMBEDDING_OUT_DIM = 256
FEATURE_COLUMNS = ["yarn_number", "frequency", "fabric_elasticity", "cielab_l_raw", "cielab_a_raw", "cielab_b_raw", "bleaching", "duration", "concentration"]
FEATURE_NORMALIZATION = True
FEATURE_MINS = [5.6,15,0,16.87,-1.91,-16.75,1,1,1000]
FEATURE_MAXS = [30,42,95,40,1.62,-0.5,4,7,7000]
FEATURE_SEQUENCE_LENGTH = 16  # 0 = single vector [D], >0 = sequence [N, D] for cross-attention
INITIAL_IMAGE = False
ENCODER_PATH = "checkpoints/encoder_epoch_50.pth"
FREEZE_ENCODER = True
# Stable diffusion initial image encoder options
SD_INITIAL_ENCODER_CKPT = "checkpoints/vae_epoch_10.pth"  # Path to pretrained VAE encoder
SD_FREEZE_INITIAL_ENCODER = False

# Model / training defaults
NOISE_DIM = 128
FIXED_G_LR = 0.0001
FIXED_D_LR = 0.0001
BCE_FACTOR = 1.0
L1_FACTOR = 50.0
L2_FACTOR = 150.0
VAL_EPOCH = 10 

# VAE defaults (balanced for memory and capacity)
VAE_BASE_CHANNELS = 64  # Compromise: not too small (32) or large (128)
VAE_LATENT_DIM = 256     # Sufficient for learning
VAE_HIDDEN_DIM = 512     # Better feature encoding
VAE_LR = 0.0001          # Conservative learning rate
VAE_KL_WEIGHT = 0.00001  # Small KL weight to avoid posterior collapse

# Stable diffusion defaults
SD_LR = 0.00005         # Lower learning rate to prevent NaN
SD_TIMESTEPS = 1000      # Full diffusion schedule  
SD_SAMPLE_STEPS = 100     
SD_EMB_DIM = 768
SD_VAL_STEPS = 50
SD_SAMPLE_BATCH = 2
SD_LOG_INTERVAL = 100   
SD_BASE_CHANNELS = 128
SD_ATTENTION_HEADS = 2
SD_ATTENTION_RESOLUTION = [32, 16]
SD_EMA_DECAY = 0.9995
SD_GRAD_CLIP = 0.5       # More aggressive gradient clipping
SD_DDP_TIMEOUT_MINUTES = 30  # Increase DDP timeout for slow validation
SD_CFG_DROPOUT = 0.0  # DISABLED: Was teaching model to ignore features!
SD_X0_LOSS_WEIGHT = 0.0  # DISABLED: Causing NaN - use noise prediction only
SD_PERCEPTUAL_WEIGHT = 0.5  # Perceptual L1 loss for better visual alignment
SD_USE_CROSS_ATTN = True  # Enable cross-attention for sequence features [B, N, D]

DEVICE_IDS = [0, 1]
WORLD_SIZE = len(DEVICE_IDS)

# Per-GPU batch size and dataloader workers
BATCH_SIZE_PER_GPU = 2
NUM_WORKERS = 4

