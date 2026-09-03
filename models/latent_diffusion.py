import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import config as cfg
from torch.amp import autocast, GradScaler

from models.latent_diffusion import (
    GaussianDiffusion,
    ModelEMA,
    LatentDiffusionConditioned,
    LatentDiffusionPipeline,
)

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

from utils.vae_loader import load_vae


def _measure_denoising_quality(
    model,
    diffusion,
    features,
    targets,
    device,
    vae_encoder=None,
    timesteps_to_test=[100, 200, 400, 600, 800],
    initial_images=None,
):
    model.eval()
    losses_by_timestep = {}
    prediction_stats = {}
    
    with torch.no_grad():
        
        if vae_encoder is not None:
            noise_for_vae = torch.randn(
                targets.size(0), 4,
                targets.size(2) // 8, targets.size(3) // 8,
                device=device
            )
            targets_latent = vae_encoder(targets, noise_for_vae)
        else:
            targets_latent = targets
        
        for t_val in timesteps_to_test:
            t = torch.full((targets_latent.size(0),), t_val, dtype=torch.long, device=device)
            noise = torch.randn_like(targets_latent)
            
            noisy = diffusion.q_sample(targets_latent, t, noise)

            pred_noise = model(noisy, t, features, initial_images)

            mse = F.mse_loss(pred_noise, noise).item()
            losses_by_timestep[t_val] = mse

            pred_std = pred_noise.std().item()
            pred_mean = pred_noise.mean().item()
            prediction_stats[t_val] = {
                'std': pred_std,
                'mean': pred_mean
            }
    
    return losses_by_timestep, prediction_stats


def _setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    timeout_minutes = getattr(cfg, "DDP_TIMEOUT_MINUTES", 10)
    timeout = torch.distributed.timedelta(minutes=timeout_minutes)
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    device_id = cfg.DEVICE_IDS[rank]
    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _save_checkpoint(model, optimizer, epoch, save_dir, version, ema_model=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    
    filename = os.path.join(save_dir, f"diffusion_{version}_epoch_{epoch:04d}.pth")
    torch.save(checkpoint, filename)
    return filename


def _load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)


def _run_validation(
    model,
    pipeline,
    diffusion,
    val_loader,
    device,
    clip_model,
    clip_preprocess,
    sample_dir,
    epoch,
    vae_encoder=None,
    rank=0,
):
    if val_loader is None or clip_model is None or clip_preprocess is None:
        return None

    if hasattr(val_loader, "sampler") and isinstance(val_loader.sampler, DistributedSampler):
        val_loader.sampler.set_epoch(epoch)

    model.eval()
    
    l1_loss_fn = nn.L1Loss(reduction="mean")

    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_clip = 0.0
    total_rgb_dist = 0.0
    clip_count = 0
    total_samples = 0

    for idx, (initial_images, features, target_images, _) in enumerate(val_loader):
        
        initial_images = initial_images.to(device)
        features = features.to(device)
        targets = target_images.to(device)

        with torch.no_grad():
            val_steps = cfg.SAMPLE_STEPS

            if cfg.INITIAL_IMAGE:
                samples = pipeline.sample(features, steps=val_steps, save_intermediates=False,
                                          initial_images=initial_images,
                                          temperature=cfg.SAMPLE_TEMPERATURE,
                                          eta=cfg.SAMPLER_ETA)
            else:
                samples = pipeline.sample(features, steps=val_steps, save_intermediates=False,
                                          temperature=cfg.SAMPLE_TEMPERATURE,
                                          eta=cfg.SAMPLER_ETA)
                
        batch_size = targets.size(0)

        total_samples += batch_size
        total_l1 += l1_loss_fn(samples, targets).item() * batch_size
        total_psnr += calculate_psnr(samples, targets) * batch_size
        total_ssim += calculate_ssim(samples, targets) * batch_size

        clip_sum, clip_bs = compute_clip_metrics_batch(samples, targets, clip_model, clip_preprocess, device)
        total_clip += clip_sum
        clip_count += clip_bs
        total_rgb_dist += calculate_avg_rgb_distance(samples, targets) * batch_size

        if idx == 0 and rank == 0:
            save_random_sample_pairs(
                initial_images,  
                samples,         
                targets,         
                sample_dir,
                epoch,
                prefix=f"d_{cfg.VERSION_NAME.lower()}",
                num_samples=batch_size,
            )
            
            if cfg.INITIAL_IMAGE:
                timestep_losses, pred_stats = _measure_denoising_quality(
                    model, diffusion, features, targets, device, vae_encoder,
                    timesteps_to_test=[100, 200, 400, 600, 800],
                    initial_images=initial_images
                )
            else:
                timestep_losses, pred_stats = _measure_denoising_quality(
                    model, diffusion, features, targets, device, vae_encoder,
                    timesteps_to_test=[100, 200, 400, 600, 800]
                )

    if total_samples == 0:
        return None

    metrics = {
        "val_l1": total_l1 / total_samples,
        "val_psnr": total_psnr / total_samples,
        "val_ssim": total_ssim / total_samples,
        "val_clip": total_clip / max(clip_count, 1),
        "val_rgb_dist": total_rgb_dist / total_samples,
    }

    if timestep_losses:
        for t, loss in timestep_losses.items():
            metrics[f"val_loss_t{t}"] = loss
        avg_pred_std = sum(pred_stats[t]['std'] for t in pred_stats) / len(pred_stats)
        metrics["pred_variance"] = avg_pred_std
    
    return metrics


def _ddp_worker(rank, world_size, epochs, retrain, checkpoint_path, version):
    device = _setup_ddp(rank, world_size)

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    if len(train_loader) == 0:
        raise ValueError(f"[D] ERROR: Training dataloader is empty! Check dataset configuration.")
    if val_loader and len(val_loader) == 0:
        print(f"[D] WARNING: Validation dataloader is empty!")
    
    if rank == 0:
        print(f"[D] Training batches per epoch: {len(train_loader)}")
        print(f"[D] Validation batches per epoch: {len(val_loader) if val_loader else 0}")

    sample_train_features = train_loader.dataset.input_data[0]
    sample_val_features = val_loader.dataset.input_data[0]
    
    print(f"[D] Training features: {sample_train_features.shape}")
    print(f"[D] Validation features: {sample_val_features.shape}")
    
    vae_encoder, vae_decoder = load_vae(cfg.VAE_CKPT, device)

    _, val_loader_diag, _ = create_dataloaders(
        batch_size=1,
        num_workers=0,          # <-- critical: no forked workers for this quick check
        pin_memory=False,
        distributed=False,
    )

    raw_stds = []
    raw_means = []

    for i, batch in enumerate(val_loader_diag):
        if i >= 50: 
            break
        img = batch[2][:1].to(device)  # [1, 3, 512, 512]

        noise = torch.zeros(1, 4, img.shape[2] // 8, img.shape[3] // 8, device=device)

        with torch.no_grad():
            latent_scaled = vae_encoder(img, noise)
            latent_raw = latent_scaled / cfg.VAE_SCALE

        raw_stds.append(latent_raw.std().item())
        raw_means.append(latent_raw.mean().item())

    mean_raw_std  = sum(raw_stds)  / len(raw_stds)
    mean_raw_mean = sum(raw_means) / len(raw_means)
    correct_scale = 1.0 / mean_raw_std

    print(f"[VAE] Raw latent mean (avg over dataset) : {mean_raw_mean:.4f}  (target ~0.0)")
    print(f"[VAE] Raw latent std  (avg over dataset) : {mean_raw_std:.4f}")
    print(f"[VAE] Correct VAE_SCALE                  : {correct_scale:.4f}  (current: {cfg.VAE_SCALE})")
    print(f"[VAE] Verification - scaled std will be  : {mean_raw_std * correct_scale:.4f}  (target 1.0)")

    num_features = len(cfg.FEATURE_COLUMNS)
    base_model = LatentDiffusionConditioned(
        latent_channels=4,
        emb_dim=cfg.EMB_DIM,
        base_channels=cfg.BASE_CHANNELS,
        use_initial_image=cfg.INITIAL_IMAGE,
    )

    feature_dim = num_features * cfg.EMB_DIM
    if cfg.INITIAL_IMAGE:
        print(f"[D] Initial image conditioning ENABLED "
              f"(feature_dim={feature_dim} → concat → {feature_dim * 2})")
    else:
        print(f"[D] Initial image conditioning DISABLED (feature_dim={feature_dim})")

    diffusion = GaussianDiffusion(timesteps=cfg.TIMESTEPS).to(device)
    model = DDP(base_model.to(device), device_ids=[cfg.DEVICE_IDS[rank]])
    
    pipeline = LatentDiffusionPipeline(model.module, diffusion, vae_encoder, vae_decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else nullcontext()

    ema_helper = ModelEMA(base_model, cfg.EMA_DECAY)
    ema_helper.to(device)
    ema_pipeline = LatentDiffusionPipeline(ema_helper.ema, diffusion, vae_encoder, vae_decoder) if rank == 0 else None

    rgb_loss_weight = getattr(cfg, "RGB_LOSS_WEIGHT", 0.05)
    if rank == 0:
        print(f"[D] RGB difference loss weight: {rgb_loss_weight}")

    save_dir   = os.path.join("checkpoints", "diffusion")
    log_dir    = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    if rank == 0:
        os.makedirs(save_dir,   exist_ok=True)
        os.makedirs(log_dir,    exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 0
    if retrain and checkpoint_path:
        if os.path.exists(checkpoint_path):
            start_epoch = _load_checkpoint(model, optimizer, checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "ema_state_dict" in ckpt:
                ema_helper.ema.load_state_dict(ckpt["ema_state_dict"])
                if rank == 0:
                    print(f"[D] Loaded EMA weights from checkpoint")
            if rank == 0:
                print(f"[D] Resumed from {checkpoint_path} at epoch {start_epoch + 1}")
        elif rank == 0:
            print(f"[D] Checkpoint {checkpoint_path} not found, starting from scratch")
    elif retrain and not checkpoint_path:
        print(f"[D] No checkpoint directory found, starting from scratch")

    metrics_logger = MetricsLogger(log_dir, f"diffusion_{version}_log.csv") if rank == 0 else None
    clip_model = clip_preprocess = None
    if rank == 0:
        clip_model, clip_preprocess = load_clip_model(device)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        steps = 0
        model.train()
    
        start_time = time.time()
        for batch_idx, (initial_images, features, target_images, _) in enumerate(train_loader):
            initial_images = initial_images.to(device)
            features = features.to(device)
            targets = target_images.to(device)

            if batch_idx == 0 and rank == 0:
                print(f"[D] Epoch {epoch} - batch 0/{len(train_loader)} "
                      f"features: {features.shape}, targets: {targets.shape}")
                if cfg.INITIAL_IMAGE:
                    print(f"[D] Initial images: {initial_images.shape}")

            optimizer.zero_grad()
            with amp_ctx():
                if cfg.INITIAL_IMAGE:
                    loss_dict = diffusion.p_loss(model, targets, features,
                                                 vae_encoder=vae_encoder,
                                                 vae_decoder=vae_decoder,
                                                 rgb_loss_weight=rgb_loss_weight,
                                                 initial_images=initial_images)
                else:
                    loss_dict = diffusion.p_loss(model, targets, features,
                                                 vae_encoder=vae_encoder,
                                                 vae_decoder=vae_decoder,
                                                 rgb_loss_weight=rgb_loss_weight)
                loss = loss_dict['loss']
                loss_metrics = loss_dict.get('metrics', {})

            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                else:
                    scaler.unscale_(optimizer)
                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                else:
                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                optimizer.step()

            ema_helper.update(model)
            epoch_loss += loss.item()
            steps += 1

            if (batch_idx + 1) % cfg.LOG_INTERVAL == 0 and rank == 0:
                noise_loss_val = loss_metrics.get('noise_loss', loss.item())
                perceptual_loss_val = loss_metrics.get('perceptual_loss', 0.0)
                rgb_loss_val = loss_metrics.get('rgb_loss', 0.0)
                print(
                    f"[D] Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} | "
                    f"Noise: {noise_loss_val:.4f} | "
                    f"Percep: {perceptual_loss_val:.4f} | "
                    f"RGB: {rgb_loss_val:.4f} | "
                    f"Grad Norm: {grad_norm:.4f}"
                )

        if rank == 0:
            print(f"[D] Epoch {epoch} - Completed all {len(train_loader)} batches in {time.time() - start_time:.2f}s")

        loss_tensor = torch.tensor([epoch_loss, steps], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps = max(int(loss_tensor[1].item()), 1)
        avg_loss = (loss_tensor[0] / total_steps).item()

        if rank == 0:
            print(f"[D] Rank {rank} finished all_reduce for epoch {epoch}")

        if dist.is_initialized():
            if rank == 0:
                print(f"[D] Rank {rank} entering first barrier before validation")
            dist.barrier()
            if rank == 0:
                print(f"[D] Rank {rank} passed first barrier")

        val_metrics = None
        should_validate = (
            rank == 0
            and val_loader is not None
            and (cfg.VAL_EPOCH <= 1 or epoch % cfg.VAL_EPOCH == 0)
        )
        if should_validate:
            print(f"[D] Rank {rank} starting validation")

            val_model = ema_helper.ema if rank == 0 else model
            val_pipeline = ema_pipeline if rank == 0 else pipeline
            
            val_metrics = _run_validation(
                val_model,
                val_pipeline,
                diffusion,
                val_loader,
                device,
                clip_model,
                clip_preprocess,
                sample_dir,
                epoch,
                vae_encoder,
                rank,
            )

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            elapsed = time.time() - start_time
            if val_metrics:
                
                print(
                    f"[D] Epoch {epoch} Loss: {avg_loss:.4f} | "
                    f"Val L1: {val_metrics['val_l1']:.4f}, PSNR: {val_metrics['val_psnr']:.2f}, "
                    f"SSIM: {val_metrics['val_ssim']:.4f}, CLIP: {val_metrics['val_clip']:.4f}, "
                    f"RGB Dist: {val_metrics['val_rgb_dist']:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )

                timestep_keys = [k for k in val_metrics.keys() if k.startswith('val_loss_t')]
                if timestep_keys:
                    print(f"[D] Denoising quality by timestep:")
                    for key in sorted(timestep_keys, key=lambda x: int(x.split('t')[1])):
                        t = int(key.split('t')[1])
                        loss = val_metrics[key]
                        quality = "Good" if loss < 0.05 else "Learning" if loss < 0.1 else "Poor"
                        print(f"     t={t:3d}: loss={loss:.4f} {quality}")

                    pred_var = val_metrics.get('pred_variance', 0)
                    if pred_var < 0.1:
                        var_status = "MODE COLLAPSE!"
                    elif pred_var < 0.3:
                        var_status = "Low (risky)"
                    else:
                        var_status = "Healthy"
                    print(f"[D] Pred Variance: {pred_var:.4f} {var_status}")

                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_l1": val_metrics["val_l1"],
                    "val_psnr": val_metrics["val_psnr"],
                    "val_ssim": val_metrics["val_ssim"],
                    "val_clip": val_metrics["val_clip"],
                    "val_rgb_dist": val_metrics["val_rgb_dist"],
                }
  
                for key in timestep_keys:
                    log_dict[key] = val_metrics[key]
                
                metrics_logger.log(log_dict)
                
            else:
                print(f"[D] Epoch {epoch} Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
                metrics_logger.log({"epoch": epoch, "train_loss": avg_loss})

            if should_validate:
                saved_path = _save_checkpoint(
                    model, optimizer, epoch, save_dir, version,
                    ema_model=ema_helper.ema,
                )
                print(f"[D] Checkpoint saved: {saved_path}")

    _cleanup_ddp()


def train_distributed(epochs, retrain, checkpoint_path, version):
    device_ids = getattr(cfg, "DEVICE_IDS", None)
    if not device_ids or len(device_ids) < 2:
        raise RuntimeError("Training requires at least two devices listed in DEVICE_IDS")

    mp.spawn(
        _ddp_worker,
        args=(len(device_ids), epochs, retrain, checkpoint_path, version),
        nprocs=len(device_ids),
        join=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Training script for Latent Diffusion with DDP")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    retrain_flag = bool(args.retrain)
    checkpoint_path = args.checkpoint or None

    print(f"Launching Latent Diffusion DDP training on devices {cfg.DEVICE_IDS}")
    train_distributed(
        epochs=args.epochs,
        retrain=retrain_flag,
        checkpoint_path=checkpoint_path,
        version=cfg.VERSION_NAME,
    )


if __name__ == "__main__":
    main()