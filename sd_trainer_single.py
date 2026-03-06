import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import config as cfg
from models.stable_diffusion import (
    GaussianDiffusion,
    ModelEMA,
    StableDiffusionConditioned,
    StableDiffusionPipeline,
)
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder
from utils.dataset import create_dataloaders
from utils.training import (
    MetricsLogger,
    calculate_avg_rgb_distance,
    calculate_psnr,
    calculate_ssim,
    compute_clip_metrics_batch,
    load_clip_model,
    save_random_sample_pairs,
)


# ── VAE loading ────────────────────────────────────────────────────────────────

def _load_vae(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")

    print(f"[SD] Loading pretrained VAE from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    vae_encoder = VAE_Encoder().to(device)
    vae_decoder = VAE_Decoder().to(device)

    if isinstance(ckpt, dict) and "encoder" in ckpt and "decoder" in ckpt:
        vae_encoder.load_state_dict(ckpt["encoder"], strict=True)
        vae_decoder.load_state_dict(ckpt["decoder"], strict=True)
    else:
        raise RuntimeError(
            f"Unrecognised VAE checkpoint format. Keys: {list(ckpt.keys())[:8]}"
        )

    for p in vae_encoder.parameters():
        p.requires_grad = False
    for p in vae_decoder.parameters():
        p.requires_grad = False

    vae_encoder.eval()
    vae_decoder.eval()
    print("[SD] VAE loaded and frozen.")
    return vae_encoder, vae_decoder


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, ema_model, epoch, save_dir, version):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"sd_{version}_epoch_{epoch:04d}.pth")
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict":      ema_model.state_dict() if ema_model else None,
    }, path)
    return path


def _load_checkpoint(model, optimizer, ema_model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ema_model and "ema_state_dict" in ckpt and ckpt["ema_state_dict"]:
        ema_model.ema.load_state_dict(ckpt["ema_state_dict"])
    return ckpt.get("epoch", 0)


# ── Validation ─────────────────────────────────────────────────────────────────

def _run_validation(model, pipeline, diffusion, val_loader, device,
                    clip_model, clip_preprocess, sample_dir, epoch, vae_encoder):
    if val_loader is None or clip_model is None:
        return None

    model.eval()
    diffusion.eval()
    l1_fn = nn.L1Loss(reduction="mean")

    total_l1 = total_psnr = total_ssim = total_clip = total_rgb = 0.0
    clip_count = total_samples = 0

    for idx, (initial_images, features, target_images, _) in enumerate(val_loader):
        initial_images = initial_images.to(device)
        features       = features.to(device)
        targets        = target_images.to(device)

        with torch.no_grad():
            if cfg.INITIAL_IMAGE:
                samples = pipeline.sample(features, steps=cfg.SD_SAMPLE_STEPS,
                                          save_intermediates=False,
                                          initial_images=initial_images)
            else:
                samples = pipeline.sample(features, steps=cfg.SD_SAMPLE_STEPS,
                                          save_intermediates=False)

        B = targets.size(0)
        total_samples += B
        total_l1      += l1_fn(samples, targets).item() * B
        total_psnr    += calculate_psnr(samples, targets) * B
        total_ssim    += calculate_ssim(samples, targets) * B
        clip_sum, clip_bs = compute_clip_metrics_batch(
            samples, targets, clip_model, clip_preprocess, device)
        total_clip  += clip_sum
        clip_count  += clip_bs
        total_rgb   += calculate_avg_rgb_distance(samples, targets) * B

        if idx == 0:
            save_random_sample_pairs(
                initial_images, samples, targets,
                sample_dir, epoch,
                prefix=f"sd_{cfg.VERSION_NAME.lower()}",
                num_samples=B,
            )

    if total_samples == 0:
        return None

    return {
        "val_l1":       total_l1   / total_samples,
        "val_psnr":     total_psnr / total_samples,
        "val_ssim":     total_ssim / total_samples,
        "val_clip":     total_clip / max(clip_count, 1),
        "val_rgb_dist": total_rgb  / total_samples,
    }


# ── Main training loop ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Stable Diffusion on a single GPU")
    parser.add_argument("--epochs",     type=int, required=True)
    parser.add_argument("--device",     type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--retrain",    type=int, default=0,
                        help="1 = resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint (optional with --retrain 1)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    print(f"[SD] Training on {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=False,
    )
    print(f"[SD] Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── VAE (frozen) ──────────────────────────────────────────────────────────
    vae_encoder, vae_decoder = _load_vae(cfg.SD_VAE_CKPT, device)

    # ── Diffusion model ───────────────────────────────────────────────────────
    base_model = StableDiffusionConditioned(
        latent_channels=4,
        emb_dim=cfg.SD_EMB_DIM,
        base_channels=cfg.SD_BASE_CHANNELS,
        use_initial_image=cfg.INITIAL_IMAGE,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=cfg.SD_TIMESTEPS).to(device)
    pipeline  = StableDiffusionPipeline(base_model, diffusion, vae_encoder, vae_decoder)

    if cfg.INITIAL_IMAGE:
        print("[SD] Initial image conditioning ENABLED")
    else:
        print("[SD] Initial image conditioning DISABLED")

    # ── Optimiser / AMP ───────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(base_model.parameters(), lr=cfg.SD_LR)
    scaler    = GradScaler("cuda")
    amp_ctx   = lambda: autocast(device_type="cuda")

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_helper  = ModelEMA(base_model, cfg.SD_EMA_DECAY)
    ema_helper.to(device)
    ema_pipeline = StableDiffusionPipeline(
        ema_helper.ema, diffusion, vae_encoder, vae_decoder)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    save_dir    = os.path.join("checkpoints", "sd")

    if args.retrain:
        ckpt_path = args.checkpoint
        if not ckpt_path:
            # Auto-detect latest checkpoint
            if os.path.exists(save_dir):
                ckpts = sorted(
                    [f for f in os.listdir(save_dir) if f.endswith(".pth")],
                    key=lambda x: int(x.split("_")[-1].replace(".pth", ""))
                )
                if ckpts:
                    ckpt_path = os.path.join(save_dir, ckpts[-1])
        if ckpt_path and os.path.exists(ckpt_path):
            start_epoch = _load_checkpoint(
                base_model, optimizer, ema_helper, ckpt_path, device)
            print(f"[SD] Resumed from {ckpt_path}, starting at epoch {start_epoch + 1}")
        else:
            print("[SD] No checkpoint found, starting from scratch")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir    = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(log_dir,     exist_ok=True)
    os.makedirs(sample_dir,  exist_ok=True)

    metrics_logger = MetricsLogger(log_dir, f"diffusion_{cfg.VERSION_NAME}_single_log.csv")
    clip_model, clip_preprocess = load_clip_model(device)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        base_model.train()
        diffusion.train()

        epoch_loss = 0.0
        steps      = 0
        start_time = time.time()

        for batch_idx, (initial_images, features, target_images, _) in enumerate(train_loader):
            initial_images = initial_images.to(device)
            features       = features.to(device)
            targets        = target_images.to(device)

            with amp_ctx():
                if cfg.INITIAL_IMAGE:
                    loss_dict = diffusion.p_loss(
                        base_model, targets, features,
                        vae_encoder=vae_encoder,
                        initial_images=initial_images,
                    )
                else:
                    loss_dict = diffusion.p_loss(
                        base_model, targets, features,
                        vae_encoder=vae_encoder,
                    )
                loss = loss_dict["loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if cfg.SD_GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), cfg.SD_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            ema_helper.update(base_model)

            epoch_loss += loss.item()
            steps      += 1

            if (batch_idx + 1) % cfg.SD_LOG_INTERVAL == 0:
                x0 = loss_dict.get("metrics", {}).get("x0_loss", 0.0)
                print(
                    f"[SD] Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)} "
                    f"Loss: {epoch_loss/steps:.4f}  X0: {x0:.4f}"
                )

        avg_loss = epoch_loss / max(steps, 1)
        elapsed  = time.time() - start_time
        print(f"[SD] Epoch {epoch} done — Loss: {avg_loss:.4f}  Time: {elapsed:.1f}s")

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = None
        if epoch % cfg.SD_VAL_EPOCH == 0:
            val_metrics = _run_validation(
                ema_helper.ema, ema_pipeline, diffusion,
                val_loader, device,
                clip_model, clip_preprocess,
                sample_dir, epoch, vae_encoder,
            )
            if val_metrics:
                print(
                    f"[SD] Val — L1: {val_metrics['val_l1']:.4f}  "
                    f"PSNR: {val_metrics['val_psnr']:.2f}  "
                    f"SSIM: {val_metrics['val_ssim']:.4f}  "
                    f"CLIP: {val_metrics['val_clip']:.4f}  "
                    f"RGB: {val_metrics['val_rgb_dist']:.4f}"
                )

        # ── Log ───────────────────────────────────────────────────────────────
        log_dict = {"epoch": epoch, "train_loss": avg_loss}
        if val_metrics:
            log_dict.update(val_metrics)
        metrics_logger.log(log_dict)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % cfg.SD_VAL_EPOCH == 0:
            path = _save_checkpoint(
                base_model, optimizer, ema_helper, epoch,
                save_dir, cfg.VERSION_NAME,
            )
            print(f"[SD] Checkpoint saved: {path}")

    print("[SD] Training complete.")


if __name__ == "__main__":
    main()
