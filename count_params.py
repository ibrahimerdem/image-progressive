#!/usr/bin/env python3
"""Count parameters for SD models"""

import torch
import config as cfg
from models.stable_diffusion import StableDiffusionConditioned, GaussianDiffusion
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder


def count_parameters(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}:")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Size (MB):            {total * 4 / (1024**2):.2f}")
    return total, trainable


print("="*60)
print("MODEL PARAMETER COUNTS")
print("="*60)

# SD Models
print("\n" + "="*60)
print("STABLE DIFFUSION MODELS:")
print("-"*60)

sd_model = StableDiffusionConditioned(
    latent_channels=4,
    emb_dim=cfg.SD_EMB_DIM,
    base_channels=cfg.SD_BASE_CHANNELS,
    use_initial_image=cfg.INITIAL_IMAGE
)
diffusion = GaussianDiffusion(timesteps=cfg.SD_TIMESTEPS)

sd_total, sd_train = count_parameters(sd_model, "SD Model")
diff_total, diff_train = count_parameters(diffusion, "Diffusion Schedule")

print(f"\nSD Total:")
print(f"  Combined parameters: {sd_total + diff_total:,}")
print(f"  Combined size (MB):  {(sd_total + diff_total) * 4 / (1024**2):.2f}")

# VAE Models
print("\n" + "="*60)
print("VAE MODELS:")
print("-"*60)

vae_encoder = VAE_Encoder()
vae_decoder = VAE_Decoder()

enc_total, enc_train = count_parameters(vae_encoder, "VAE Encoder")
dec_total, dec_train = count_parameters(vae_decoder, "VAE Decoder")

print(f"\nVAE Total:")
print(f"  Combined parameters: {enc_total + dec_total:,}")
print(f"  Combined size (MB):  {(enc_total + dec_total) * 4 / (1024**2):.2f}")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"SD (UNet only):      {sd_total:,} params ({sd_total * 4 / (1024**2):.2f} MB)")
print(f"SD + VAE (full):     {sd_total + enc_total + dec_total:,} params ({(sd_total + enc_total + dec_total) * 4 / (1024**2):.2f} MB)")
print("="*60)
