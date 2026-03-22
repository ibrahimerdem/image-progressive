import torch
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder
import config as cfg

ckpt = torch.load(cfg.SD_VAE_CKPT, map_location="cpu", weights_only=False)
quant_keys = [k for k in ckpt.keys() if "quant" in k]
print(quant_keys)