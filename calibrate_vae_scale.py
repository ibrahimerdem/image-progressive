"""
Compute the empirical VAE_SCALE = 1 / std(encoder_mean) over the training set.
Run after training to verify / update config.VAE_SCALE.

Usage:
    python calibrate_vae_scale.py --checkpoint checkpoints/vae_retrain/vae_load_and_retrain_epoch_0050.pth
    python calibrate_vae_scale.py   # uses cfg.VAE_CKPT (pre-trained weights)
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

import config as cfg
from utils.vae_loader import load_vae


def _run_encoder_mean(encoder, x):
    """Extract unscaled mean from encoder (replicates VAEWrapper._run_encoder)."""
    for module in encoder:
        if getattr(module, "stride", None) == (2, 2):
            x = F.pad(x, (0, 1, 0, 1))
        x = module(x)
    mean, log_var = torch.chunk(x, 2, dim=1)
    return mean  # unscaled, no noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint (defaults to cfg.VAE_RETRAIN_DIR latest)")
    parser.add_argument("--batches", type=int, default=50,
                        help="Number of batches to sample (default 50 ≈ 200 images at batch=4)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        # pick the latest checkpoint in VAE_RETRAIN_DIR, fall back to pre-trained
        retrain_ckpts = sorted(
            [f for f in os.listdir(cfg.VAE_RETRAIN_DIR) if f.endswith(".pth")]
        ) if os.path.isdir(cfg.VAE_RETRAIN_DIR) else []
        ckpt = os.path.join(cfg.VAE_RETRAIN_DIR, retrain_ckpts[-1]) if retrain_ckpts else cfg.VAE_CKPT
    print(f"[calibrate] Using checkpoint: {ckpt}")
    encoder, _ = load_vae(ckpt, device, freeze=True)
    encoder.eval()

    tf = transforms.Compose([
        transforms.Resize((cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    df = pd.read_csv(cfg.TRAIN_CSV)
    paths = [os.path.join(cfg.TARGET_DIR, p) for p in df["target_filename"].tolist()]

    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, paths, tf):
            self.paths = paths
            self.tf = tf
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.tf(img)

    loader = DataLoader(SimpleDS(paths, tf), batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    all_means = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            batch = batch.to(device)
            mean = _run_encoder_mean(encoder, batch)
            all_means.append(mean.cpu())

    all_means = torch.cat(all_means, dim=0)  # (N, 4, H/8, W/8)
    std = all_means.std().item()
    scale = 1.0 / std

    print(f"\nSampled {all_means.shape[0]} images")
    print(f"  Latent mean std : {std:.5f}")
    print(f"  Recommended VAE_SCALE = 1/std = {scale:.5f}")
    print(f"  Current  VAE_SCALE            = {cfg.VAE_SCALE}")
    if abs(scale - cfg.VAE_SCALE) / cfg.VAE_SCALE > 0.10:
        print("  >> >10% difference — consider updating config.VAE_SCALE")
    else:
        print("  >> Within 10% of current value — current scale is fine")


if __name__ == "__main__":
    main()
