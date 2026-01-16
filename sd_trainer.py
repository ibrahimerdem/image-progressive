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


def _measure_denoising_quality(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    timesteps_to_test: list = [50, 100, 200, 300, 400],
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
        for t_val in timesteps_to_test:
            t = torch.full((targets.size(0),), t_val, dtype=torch.long, device=device)
            noise = torch.randn_like(targets)
            
            # Add noise at this timestep
            noisy = diffusion.q_sample(targets, t, noise)
            
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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device_id = cfg.DEVICE_IDS[rank]
    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, save_dir: str, version: str) -> str:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
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
    device: torch.device,
    clip_model,
    clip_preprocess,
    sample_dir: str,
    epoch: int,
    rank: int = 0,
    max_batches: int = 10,  # Limit validation to first 10 batches
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
    
    # Save intermediates at key epochs: 5, 10, 20, 50, 100
    save_intermediates = epoch in [5, 10, 20, 50, 100]
    
    if epoch in [5, 10, 20, 50, 100]:
        print(f"[SD] Epoch {epoch}: Saving intermediate diffusion steps")
    
    if epoch == 5:
        print(f"[SD] Validation using {cfg.SD_SAMPLE_STEPS} sampling steps from {cfg.SD_TIMESTEPS} trained timesteps")

    for idx, (_, features, target_images, _) in enumerate(val_loader):
        if idx >= max_batches: 
            break
            
        features = features.to(device)
        targets = target_images.to(device)

        with torch.no_grad():
            with autocast_ctx():
                val_steps = cfg.SD_SAMPLE_STEPS
                
                # Save intermediates at key epochs
                if idx == 0 and save_intermediates:
                    result = pipeline.sample(features, steps=val_steps, save_intermediates=True)
                    if isinstance(result, tuple):
                        samples, intermediates = result
                        save_diffusion_intermediates(intermediates, sample_dir, epoch, sample_idx=0)
                        
                        # Print timestep schedule for debugging (epoch 5 only)
                        if epoch == 5:
                            timesteps = [t for t, _ in intermediates]
                            print(f"[SD] Saved intermediates at timesteps: {timesteps}")
                    else:
                        samples = result
                else:
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
                model, diffusion, features, targets, device,
                timesteps_to_test=[50, 100, 200, 300, 400]
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

    feature_dim = train_loader.dataset.input_data.shape[1]

    base_model = StableDiffusionConditioned(
        input_dim=feature_dim,
    )
    diffusion = GaussianDiffusion(timesteps=cfg.SD_TIMESTEPS).to(device)

    model = DDP(base_model.to(device), device_ids=[cfg.DEVICE_IDS[rank]])
    pipeline = StableDiffusionPipeline(model.module, diffusion)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SD_LR)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else nullcontext()

    ema_helper = ModelEMA(base_model, cfg.SD_EMA_DECAY)
    ema_helper.to(device)
    ema_pipeline = StableDiffusionPipeline(ema_helper.ema, diffusion) if rank == 0 else None

    start_epoch = 0
    if retrain and checkpoint_path:
        if os.path.exists(checkpoint_path):
            start_epoch = _load_checkpoint(model, optimizer, checkpoint_path)
            if rank == 0:
                print(f"[SD] Resumed from checkpoint {checkpoint_path} starting at epoch {start_epoch}")
        elif rank == 0:
            print(f"[SD] Checkpoint {checkpoint_path} not found, starting from scratch")

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
        for batch_idx, (initial_images, features, target_images, _) in enumerate(train_loader):
            features = features.to(device)
            targets = target_images.to(device)

            with amp_ctx():
                loss = diffusion.p_loss(model, targets, features)

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
            val_metrics = _run_validation(
                model,
                pipeline,
                diffusion,
                val_loader,
                device,
                clip_model,
                clip_preprocess,
                sample_dir,
                epoch,
                rank,
            )
            print(f"[SD] Rank {rank} finished validation")

        # Keep all ranks in sync after validation before next epoch
        if dist.is_initialized():
            if rank == 0:
                print(f"[SD] Rank {rank} entering second barrier after validation")
            dist.barrier()
            if rank == 0:
                print(f"[SD] Rank {rank} passed second barrier")

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
                checkpoint_path = _save_checkpoint(model, optimizer, epoch, save_dir, version)
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
    if retrain_flag and not checkpoint_path:
        parser.error("--checkpoint is required when --retrain=1")

    print(f"Launching stable diffusion DDP training on devices {cfg.DEVICE_IDS}")
    train_distributed(
        epochs=args.epochs,
        retrain=retrain_flag,
        checkpoint_path=checkpoint_path,
        version="ddp",
    )


if __name__ == "__main__":
    main()
