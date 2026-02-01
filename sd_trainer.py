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
    calculate_psnr,
    calculate_ssim,
    compute_clip_metrics_batch,
    load_clip_model,
    save_random_sample_pairs,
    save_diffusion_intermediates,
)


def _load_vae(checkpoint_path: str, device: torch.device):
    """Load pretrained VAE encoder and decoder from checkpoint and freeze them"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    
    print(f"[SD] Loading pretrained VAE from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Debug: print checkpoint structure
    print(f"[SD] Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"[SD] Checkpoint keys: {list(checkpoint.keys())[:10]}")
        
        # Check if encoder/decoder are stored as nested dicts (common format)
        if 'encoder' in checkpoint and 'decoder' in checkpoint:
            print(f"[SD] Found 'encoder' and 'decoder' as top-level keys")
            encoder_weights = checkpoint['encoder']
            decoder_weights = checkpoint['decoder']
            
            # Check if they are state_dicts or need further unwrapping
            if isinstance(encoder_weights, dict):
                print(f"[SD] Encoder has {len(encoder_weights)} keys")
                print(f"[SD] Sample encoder keys: {list(encoder_weights.keys())[:3]}")
            if isinstance(decoder_weights, dict):
                print(f"[SD] Decoder has {len(decoder_weights)} keys")
                print(f"[SD] Sample decoder keys: {list(decoder_weights.keys())[:3]}")
            
            vae_encoder = VAE_Encoder().to(device)
            vae_decoder = VAE_Decoder().to(device)
            
            # Load the state dicts
            try:
                vae_encoder.load_state_dict(encoder_weights, strict=True)
                print(f"[SD] ✓ Loaded encoder weights ({len(encoder_weights)} keys)")
            except Exception as e:
                print(f"[SD] ✗ Failed to load encoder: {e}")
                raise
            
            try:
                vae_decoder.load_state_dict(decoder_weights, strict=True)
                print(f"[SD] ✓ Loaded decoder weights ({len(decoder_weights)} keys)")
            except Exception as e:
                print(f"[SD] ✗ Failed to load decoder: {e}")
                raise
            
            # Freeze VAE parameters
            for param in vae_encoder.parameters():
                param.requires_grad = False
            for param in vae_decoder.parameters():
                param.requires_grad = False
            
            vae_encoder.eval()
            vae_decoder.eval()
            
            print(f"[SD] VAE loaded and frozen successfully")
            return vae_encoder, vae_decoder
        
        # Check for common checkpoint formats
        if 'model_state_dict' in checkpoint:
            print(f"[SD] Found 'model_state_dict', using it")
            checkpoint = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            print(f"[SD] Found 'state_dict', using it")
            checkpoint = checkpoint['state_dict']
    
    vae_encoder = VAE_Encoder().to(device)
    vae_decoder = VAE_Decoder().to(device)
    
    # Try multiple key patterns
    # Pattern 1: encoder.X.Y, decoder.X.Y
    encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.')}
    decoder_state = {k.replace('decoder.', ''): v for k, v in checkpoint.items() if k.startswith('decoder.')}
    
    # Pattern 2: Direct weights (no prefix)
    if not encoder_state and not decoder_state:
        print(f"[SD] No 'encoder.'/'decoder.' prefix found, checking sample keys...")
        sample_keys = list(checkpoint.keys())[:5]
        print(f"[SD] Sample keys: {sample_keys}")
        
        # Try to split by checking if keys match encoder/decoder structure
        encoder_param_names = set(n for n, _ in vae_encoder.named_parameters())
        decoder_param_names = set(n for n, _ in vae_decoder.named_parameters())
        
        encoder_state = {k: v for k, v in checkpoint.items() if k in encoder_param_names}
        decoder_state = {k: v for k, v in checkpoint.items() if k in decoder_param_names}
        
        print(f"[SD] Matched by parameter names: encoder={len(encoder_state)}, decoder={len(decoder_state)}")
    
    # Load encoder weights
    if encoder_state:
        try:
            vae_encoder.load_state_dict(encoder_state, strict=True)
            print(f"[SD] ✓ Loaded encoder weights ({len(encoder_state)} keys)")
        except Exception as e:
            print(f"[SD] ✗ Failed to load encoder: {e}")
            raise
    else:
        print(f"[SD] ✗ WARNING: No encoder weights found in checkpoint")
        print(f"[SD] Expected encoder keys like: {list(vae_encoder.state_dict().keys())[:3]}")
    
    # Load decoder weights
    if decoder_state:
        try:
            vae_decoder.load_state_dict(decoder_state, strict=True)
            print(f"[SD] ✓ Loaded decoder weights ({len(decoder_state)} keys)")
        except Exception as e:
            print(f"[SD] ✗ Failed to load decoder: {e}")
            raise
    else:
        print(f"[SD] ✗ WARNING: No decoder weights found in checkpoint")
        print(f"[SD] Expected decoder keys like: {list(vae_decoder.state_dict().keys())[:3]}")
    
    if not encoder_state or not decoder_state:
        raise RuntimeError("Failed to load VAE weights. Check checkpoint format.")
    
    # Freeze VAE parameters
    for param in vae_encoder.parameters():
        param.requires_grad = False
    for param in vae_decoder.parameters():
        param.requires_grad = False
    
    vae_encoder.eval()
    vae_decoder.eval()
    
    print(f"[SD] VAE loaded and frozen successfully")
    return vae_encoder, vae_decoder


def _measure_denoising_quality(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    vae_encoder=None,
    timesteps_to_test: list = [100, 200, 400, 600, 800],
):
    """
    Measure model's denoising ability at different noise levels.
    Returns MSE loss for each tested timestep + prediction statistics.
    
    Lower loss = better denoising at that noise level
    < 0.05: Model denoises well at this timestep
    0.05-0.1: Model is learning
    > 0.1: Model struggles at this noise level
    
    Also returns prediction variance to detect mode collapse.
    Healthy model: variance > 0.5
    Collapsed model: variance < 0.1 (predicting near-constant values)
    """
    model.eval()
    losses_by_timestep = {}
    prediction_stats = {}
    
    with torch.no_grad():
        # Encode targets to latent space if VAE is provided
        if vae_encoder is not None:
            noise_for_vae = torch.randn(
                targets.size(0), 4,
                targets.size(2) // 8, targets.size(3) // 8,
                device=device
            )
            targets_latent = vae_encoder(targets, noise_for_vae)
            # Note: VAE encoder already scales by 0.18215
        else:
            targets_latent = targets
        
        for t_val in timesteps_to_test:
            t = torch.full((targets_latent.size(0),), t_val, dtype=torch.long, device=device)
            noise = torch.randn_like(targets_latent)
            
            # Add noise at this timestep
            noisy = diffusion.q_sample(targets_latent, t, noise)
            
            # Predict noise
            pred_noise = model(noisy, t, features)
            
            # Calculate MSE
            mse = F.mse_loss(pred_noise, noise).item()
            losses_by_timestep[t_val] = mse
            
            # Calculate prediction statistics to detect collapse
            pred_std = pred_noise.std().item()
            pred_mean = pred_noise.mean().item()
            prediction_stats[t_val] = {
                'std': pred_std,
                'mean': pred_mean
            }
    
    return losses_by_timestep, prediction_stats


def _setup_ddp(rank: int, world_size: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    # Set DDP timeout for slow validation
    timeout_minutes = getattr(cfg, "SD_DDP_TIMEOUT_MINUTES", 10)
    timeout = torch.distributed.timedelta(minutes=timeout_minutes)
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    device_id = cfg.DEVICE_IDS[rank]
    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, save_dir: str, version: str, ema_model=None) -> str:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    # Save EMA model if provided
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    
    filename = os.path.join(save_dir, f"sd_{version}_epoch_{epoch:04d}.pth")
    torch.save(checkpoint, filename)
    return filename


def _load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)


def _run_validation(
    model: nn.Module,
    pipeline: StableDiffusionPipeline,
    diffusion: GaussianDiffusion,
    val_loader,
    device,
    clip_model,
    clip_preprocess,
    sample_dir: str,
    epoch: int,
    vae_encoder=None,
    rank: int = 0,
):
    if val_loader is None or clip_model is None or clip_preprocess is None:
        return None

    if hasattr(val_loader, "sampler") and isinstance(val_loader.sampler, DistributedSampler):
        val_loader.sampler.set_epoch(epoch)

    model.eval()
    diffusion.eval()
    l1_loss_fn = nn.L1Loss(reduction="mean")

    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_clip = 0.0
    clip_count = 0
    total_samples = 0

    autocast_ctx = (lambda: autocast(device_type="cuda")) if device.type == "cuda" else (lambda: nullcontext())

    # VALIDATION SET: Calculate metrics over entire validation dataset
    # Features are [B, 9] tensor (9 continuous features normalized [0-1])
    for idx, (_, features, target_images, _) in enumerate(val_loader):
        
        features = features.to(device)
        targets = target_images.to(device)

        with torch.no_grad():
            with autocast_ctx():
                val_steps = cfg.SD_SAMPLE_STEPS
                
                # Generate from pure noise (full generation)
                samples = pipeline.sample(features, steps=val_steps, save_intermediates=False)
                
        batch_size = targets.size(0)

        total_samples += batch_size
        total_l1 += l1_loss_fn(samples, targets).item() * batch_size
        total_psnr += calculate_psnr(samples, targets) * batch_size
        total_ssim += calculate_ssim(samples, targets) * batch_size

        clip_sum, clip_bs = compute_clip_metrics_batch(samples, targets, clip_model, clip_preprocess, device)
        total_clip += clip_sum
        clip_count += clip_bs

        # Save sample generated images with target pairs (only on rank 0)
        if idx == 0 and rank == 0:
            save_random_sample_pairs(
                targets,
                samples,
                targets,
                sample_dir,
                epoch,
                prefix="sd_val",
                num_samples=batch_size,
            )
            
            # Measure denoising quality at different timesteps (first batch only)
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
    }
    
    # Add per-timestep losses and prediction stats
    if timestep_losses:
        for t, loss in timestep_losses.items():
            metrics[f"val_loss_t{t}"] = loss
        # Add prediction variance for collapse detection
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

    # Safety check: ensure dataloaders are not empty
    if len(train_loader) == 0:
        raise ValueError(f"[SD] ERROR: Training dataloader is empty! Check dataset configuration.")
    if val_loader and len(val_loader) == 0:
        print(f"[SD] WARNING: Validation dataloader is empty!")
    
    if rank == 0:
        print(f"[SD] Training batches per epoch: {len(train_loader)}")
        print(f"[SD] Validation batches per epoch: {len(val_loader) if val_loader else 0}")

    sample_train_features = train_loader.dataset.input_data[0]
    sample_val_features = val_loader.dataset.input_data[0]
    
    print(f"[SD] Training features: {sample_train_features.shape}")
    print(f"[SD] Validation features: {sample_val_features.shape}")
    
    # Load pretrained VAE encoder and decoder (frozen)
    vae_encoder, vae_decoder = _load_vae(cfg.SD_VAE_CKPT, device)
    
    # Model works in latent space (4 channels) not RGB space
    base_model = StableDiffusionConditioned(latent_channels=4,
                                            emb_dim=cfg.SD_EMB_DIM,
                                            base_channels=cfg.SD_BASE_CHANNELS)

    diffusion = GaussianDiffusion(timesteps=cfg.SD_TIMESTEPS).to(device)
    model = DDP(base_model.to(device), device_ids=[cfg.DEVICE_IDS[rank]])
    
    # Pipeline needs VAE for encoding/decoding
    pipeline = StableDiffusionPipeline(model.module, diffusion, vae_encoder, vae_decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SD_LR)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else nullcontext()

    ema_helper = ModelEMA(base_model, cfg.SD_EMA_DECAY)
    ema_helper.to(device)
    ema_pipeline = StableDiffusionPipeline(ema_helper.ema, diffusion, vae_encoder, vae_decoder) if rank == 0 else None

    start_epoch = 0
    if retrain and checkpoint_path:
        if os.path.exists(checkpoint_path):
            start_epoch = _load_checkpoint(model, optimizer, checkpoint_path)
            # Load EMA weights if available
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "ema_state_dict" in checkpoint:
                ema_helper.ema.load_state_dict(checkpoint["ema_state_dict"])
                if rank == 0:
                    print(f"[SD] Loaded EMA weights from checkpoint")
            if rank == 0:
                print(f"[SD] Resumed from checkpoint {checkpoint_path} starting at epoch {start_epoch + 1}")
        elif rank == 0:
            print(f"[SD] Checkpoint {checkpoint_path} not found, starting from scratch")
    elif retrain and not checkpoint_path:
        # Auto-detect latest checkpoint
        if os.path.exists(save_dir):
            checkpoints = [f for f in os.listdir(save_dir) if f.startswith("sd_") and f.endswith(".pth")]
            if checkpoints:
                # Sort by epoch number
                checkpoints.sort(key=lambda x: int(x.split("_")[-1].replace(".pth", "")))
                latest_ckpt = os.path.join(save_dir, checkpoints[-1])
                start_epoch = _load_checkpoint(model, optimizer, latest_ckpt)
                # Load EMA weights if available
                checkpoint = torch.load(latest_ckpt, map_location="cpu")
                if "ema_state_dict" in checkpoint:
                    ema_helper.ema.load_state_dict(checkpoint["ema_state_dict"])
                    if rank == 0:
                        print(f"[SD] Loaded EMA weights from checkpoint")
                if rank == 0:
                    print(f"[SD] Auto-resumed from {latest_ckpt} starting at epoch {start_epoch + 1}")
            elif rank == 0:
                print(f"[SD] No checkpoints found in {save_dir}, starting from scratch")
        elif rank == 0:
            print(f"[SD] No checkpoint directory found, starting from scratch")

    save_dir = os.path.join("checkpoints", "sd")
    log_dir = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    metrics_logger = MetricsLogger(log_dir, f"stable_diffusion_{version}_log.csv") if rank == 0 else None
    clip_model = clip_preprocess = None
    if rank == 0:
        clip_model, clip_preprocess = load_clip_model(device)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        steps = 0
        model.train()
        diffusion.train()

        start_time = time.time()
        for batch_idx, (_, features, target_images, _) in enumerate(train_loader):
            # Features are [B, 9] vector - 9 continuous features normalized [0-1]
            features = features.to(device)
            targets = target_images.to(device)
            
            # Debug: Print first batch info
            if batch_idx == 0 and rank == 0:
                print(f"[SD] Epoch {epoch} - Processing batch {batch_idx}/{len(train_loader)}, features shape: {features.shape}")

            with amp_ctx():
                loss_dict = diffusion.p_loss(model, targets, features, vae_encoder=vae_encoder)
                loss = loss_dict['loss']
                loss_metrics = loss_dict.get('metrics', {})

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.SD_GRAD_CLIP and cfg.SD_GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SD_GRAD_CLIP)
                else:
                    scaler.unscale_(optimizer)
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.SD_GRAD_CLIP and cfg.SD_GRAD_CLIP > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SD_GRAD_CLIP)
                else:
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                optimizer.step()

            ema_helper.update(model)
            epoch_loss += loss.item()
            steps += 1

            if (batch_idx + 1) % cfg.SD_LOG_INTERVAL == 0 and rank == 0:
                print(
                    f"[SD] Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {epoch_loss / steps:.4f} | Grad Norm: {grad_norm:.4f}"
                )
        
        # End of batch loop - log completion
        if rank == 0:
            print(f"[SD] Epoch {epoch} - Completed all {len(train_loader)} batches in {time.time() - start_time:.2f}s")

        loss_tensor = torch.tensor([epoch_loss, steps], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps = max(int(loss_tensor[1].item()), 1)
        avg_loss = (loss_tensor[0] / total_steps).item()

        if rank == 0:
            print(f"[SD] Rank {rank} finished all_reduce for epoch {epoch}")
        
        # Keep all ranks in sync before validation to avoid collective timeouts
        if dist.is_initialized():
            if rank == 0:
                print(f"[SD] Rank {rank} entering first barrier before validation")
            dist.barrier()
            if rank == 0:
                print(f"[SD] Rank {rank} passed first barrier")

        val_metrics = None
        should_validate = (
            rank == 0
            and val_loader is not None
            and (cfg.VAL_EPOCH <= 1 or epoch % cfg.VAL_EPOCH == 0)
        )
        if should_validate:
            print(f"[SD] Rank {rank} starting validation")
            # Use EMA model for validation on rank 0, raw model on other ranks
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
                # Print main metrics
                print(
                    f"[SD] Epoch {epoch} Loss: {avg_loss:.4f} | "
                    f"Val L1: {val_metrics['val_l1']:.4f}, PSNR: {val_metrics['val_psnr']:.2f}, "
                    f"SSIM: {val_metrics['val_ssim']:.4f}, CLIP: {val_metrics['val_clip']:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )
                
                # Print per-timestep denoising losses
                timestep_keys = [k for k in val_metrics.keys() if k.startswith('val_loss_t')]
                if timestep_keys:
                    print(f"[SD] Denoising quality by timestep:")
                    for key in sorted(timestep_keys, key=lambda x: int(x.split('t')[1])):
                        t = int(key.split('t')[1])
                        loss = val_metrics[key]
                        quality = "✓ Good" if loss < 0.05 else "⚠ Learning" if loss < 0.1 else "✗ Poor"
                        print(f"     t={t:3d}: loss={loss:.4f} {quality}")
                    
                    # Always print prediction variance for collapse detection
                    pred_var = val_metrics.get('pred_variance', 0)
                    if pred_var < 0.1:
                        var_status = "⚠️ MODE COLLAPSE!"
                    elif pred_var < 0.3:
                        var_status = "⚠ Low (risky)"
                    else:
                        var_status = "✓ Healthy"
                    print(f"[SD] Pred Variance: {pred_var:.4f} {var_status}")
                
                # Log all metrics including per-timestep losses
                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_l1": val_metrics["val_l1"],
                    "val_psnr": val_metrics["val_psnr"],
                    "val_ssim": val_metrics["val_ssim"],
                    "val_clip": val_metrics["val_clip"],
                }
                # Add timestep losses
                for key in timestep_keys:
                    log_dict[key] = val_metrics[key]
                
                metrics_logger.log(log_dict)
                
                # Print progress summary at key milestones
                if epoch in [10, 20, 30, 50, 75, 100]:
                    print(f"\n{'='*60}")
                    print(f"PROGRESS SUMMARY - Epoch {epoch}/100")
                    print(f"{'='*60}")
                    print(f"Training Loss:  {avg_loss:.4f}")
                    print(f"PSNR:          {val_metrics['val_psnr']:.2f} dB  (target: >25 dB)")
                    print(f"SSIM:          {val_metrics['val_ssim']:.4f}  (target: >0.75)")
                    print(f"CLIP Sim:      {val_metrics['val_clip']:.4f}  (target: >0.70)")
                    
                    # Prediction variance (collapse detection)
                    pred_var = val_metrics.get('pred_variance', 0)
                    if pred_var < 0.1:
                        var_status = "⚠️  MODE COLLAPSE"
                    elif pred_var < 0.3:
                        var_status = "⚠ Low (risk)"
                    else:
                        var_status = "✓ Healthy"
                    print(f"Pred Variance: {pred_var:.4f}  {var_status}")
                    
                    # Count how many timesteps are learned well
                    good_timesteps = sum(1 for k in timestep_keys if val_metrics[k] < 0.05)
                    total_timesteps = len(timestep_keys)
                    print(f"Learned steps: {good_timesteps}/{total_timesteps} timesteps < 0.05 loss")
                    print(f"{'='*60}\n")
            else:
                print(f"[SD] Epoch {epoch} Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
                metrics_logger.log({"epoch": epoch, "train_loss": avg_loss})

            if should_validate:
                checkpoint_path = _save_checkpoint(model, optimizer, epoch, save_dir, version, ema_model=ema_helper.ema if rank == 0 else None)
                print(f"[SD] Checkpoint saved: {checkpoint_path}")

    _cleanup_ddp()


def train_distributed(epochs, retrain, checkpoint_path, version):
    device_ids = getattr(cfg, "DEVICE_IDS", None)
    if not device_ids or len(device_ids) < 2:
        raise RuntimeError("Stable diffusion training requires at least two devices listed in DEVICE_IDS")

    mp.spawn(
        _ddp_worker,
        args=(len(device_ids), epochs, retrain, checkpoint_path, version),
        nprocs=len(device_ids),
        join=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stable diffusion conditioned on features and an optional initial image")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    retrain_flag = bool(args.retrain)
    checkpoint_path = args.checkpoint or None
    # Note: retrain=1 means RESUME training, retrain=0 means start fresh
    # Checkpoint path is optional - if not provided, auto-detects latest

    print(f"Launching stable diffusion DDP training on devices {cfg.DEVICE_IDS}")
    train_distributed(
        epochs=args.epochs,
        retrain=retrain_flag,
        checkpoint_path=checkpoint_path,
        version="ddp",
    )


if __name__ == "__main__":
    main()