"""
VAE defect evaluation: load trained VAE, run defected_validation images through it,
compute per-pixel residual, overlay defect heatmap on original and save results.

Usage:
    python vae_evaluate.py                        # latest checkpoint in VAE_RETRAIN_DIR
    python vae_evaluate.py --checkpoint checkpoints/vae_retrain/vae_load_and_retrain_epoch_0020.pth
    python vae_evaluate.py --sigma 7.0            # tighten adaptive threshold
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import binary_dilation
from tqdm import tqdm

import config as cfg
from utils.vae_loader import load_vae


def _encode_mean(encoder, x):
    for module in encoder:
        if getattr(module, "stride", None) == (2, 2):
            x = F.pad(x, (0, 1, 0, 1))
        x = module(x)
    mean, log_var = torch.chunk(x, 2, dim=1)
    return mean


def reconstruct(encoder, decoder, x):
    mean = _encode_mean(encoder, x)
    z = mean * cfg.VAE_SCALE
    return decoder(z)


def denorm(t):
    """[-1,1] tensor → [0,1] numpy HWC"""
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def adaptive_threshold(residual_np, sigma):
    """mean + sigma*std per-image threshold."""
    return float(residual_np.mean() + sigma * residual_np.std())


def overlay_heatmap(orig_np, residual_np, threshold):
    heat = cm.hot(residual_np)[:, :, :3]
    mask = residual_np > threshold
    alpha = 0.55
    composite = orig_np.copy()
    composite[mask] = (1 - alpha) * orig_np[mask] + alpha * heat[mask]
    outline = binary_dilation(mask, iterations=2) & ~mask
    composite[outline] = [1.0, 0.0, 0.0]
    return (composite * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sigma", type=float, default=5.0,
                        help="Threshold = mean + sigma*std of residual (default 5.0)")
    parser.add_argument("--out_dir", default=os.path.join("outputs", "vae_defects"))
    args = parser.parse_args()

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpts = sorted(
            f for f in os.listdir(cfg.VAE_RETRAIN_DIR) if f.endswith(".pth")
        ) if os.path.isdir(cfg.VAE_RETRAIN_DIR) else []
        ckpt = os.path.join(cfg.VAE_RETRAIN_DIR, ckpts[-1]) if ckpts else cfg.VAE_CKPT
    print(f"[eval] checkpoint : {ckpt}")
    print(f"[eval] sigma      : {args.sigma}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_vae(ckpt, device, freeze=True)
    encoder.eval()
    decoder.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    df = pd.read_csv(cfg.VAL_CSV)
    filenames = df["target_filename"].tolist()

    anomaly_scores = []

    with torch.no_grad():
        for fname in tqdm(filenames, desc="Evaluating", ncols=90):
            img_path = os.path.join(cfg.TARGET_DIR, fname)
            pil_img  = Image.open(img_path).convert("RGB")
            x        = tf(pil_img).unsqueeze(0).to(device)

            recon = reconstruct(encoder, decoder, x)

            residual = (x - recon).abs().mean(dim=1).squeeze(0)
            residual_np = residual.cpu().numpy()
            r_max = residual_np.max() + 1e-8
            residual_norm = residual_np / r_max

            orig_np  = denorm(x.squeeze(0))
            recon_np = denorm(recon.squeeze(0))

            thr = adaptive_threshold(residual_norm, args.sigma)
            composite = overlay_heatmap(orig_np, residual_norm, thr)

            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            axes[0].imshow(orig_np);   axes[0].set_title("Original (defected)"); axes[0].axis("off")
            axes[1].imshow(recon_np);  axes[1].set_title("VAE Reconstruction");  axes[1].axis("off")
            im = axes[2].imshow(residual_np, cmap="hot", vmin=0, vmax=r_max * 0.5)
            axes[2].set_title("Residual");                                        axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            axes[3].imshow(composite); axes[3].set_title(f"Defect Overlay (σ={args.sigma}, thr={thr:.3f})"); axes[3].axis("off")

            fig.suptitle(fname, fontsize=9)
            plt.tight_layout()
            stem = os.path.splitext(fname)[0]
            plt.savefig(os.path.join(args.out_dir, f"{stem}_defect.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)

            anomaly_scores.append({
                "filename": fname,
                "mean_residual": float(residual_norm.mean()),
                "max_residual":  float(residual_norm.max()),
                "threshold_used": float(thr),
                "defect_area_pct": float((residual_norm > thr).mean() * 100),
            })

    score_df = pd.DataFrame(anomaly_scores).sort_values("mean_residual", ascending=False)
    csv_path = os.path.join(args.out_dir, "anomaly_scores.csv")
    score_df.to_csv(csv_path, index=False)
    print(f"\n[eval] Saved {len(filenames)} overlay images → {args.out_dir}/")
    print(f"[eval] Anomaly scores → {csv_path}")
    print(score_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
