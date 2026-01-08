import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import config as cfg
from models.stable_diffusion import GaussianDiffusion, StableDiffusionConditioned, StableDiffusionPipeline
from utils.dataset import create_dataloaders
from utils.training import (
    MetricsLogger,
    calculate_psnr,
    calculate_ssim,
    compute_clip_metrics_batch,
    load_clip_model,
    save_random_sample_pairs,
)


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

    for idx, (initial_images, features, target_images, _) in enumerate(val_loader):
        if idx >= cfg.SD_VAL_STEPS:
            break

        features = features.to(device)
        targets = target_images.to(device)
        initials = initial_images.to(device)
        cond_initial = initials if cfg.INITIAL_IMAGE else None

        samples = pipeline.sample(features, cond_initial, steps=cfg.SD_SAMPLE_STEPS)
        batch_size = targets.size(0)

        total_samples += batch_size
        total_l1 += l1_loss_fn(samples, targets).item() * batch_size
        total_psnr += calculate_psnr(samples, targets) * batch_size
        total_ssim += calculate_ssim(samples, targets) * batch_size

        clip_sum, clip_bs = compute_clip_metrics_batch(samples, targets, clip_model, clip_preprocess, device)
        total_clip += clip_sum
        clip_count += clip_bs

        if idx == 0:
            save_random_sample_pairs(
                initials if cfg.INITIAL_IMAGE else targets,
                samples,
                targets,
                sample_dir,
                epoch,
                prefix="sd_val",
                num_samples=min(cfg.SD_SAMPLE_BATCH, batch_size),
            )

    if total_samples == 0:
        return None

    return {
        "val_l1": total_l1 / total_samples,
        "val_psnr": total_psnr / total_samples,
        "val_ssim": total_ssim / total_samples,
        "val_clip": total_clip / max(clip_count, 1),
    }


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

    base_model = StableDiffusionConditioned(input_dim=feature_dim, use_initial=cfg.INITIAL_IMAGE)
    diffusion = GaussianDiffusion(timesteps=cfg.SD_TIMESTEPS).to(device)

    model = DDP(base_model.to(device), device_ids=[cfg.DEVICE_IDS[rank]])
    pipeline = StableDiffusionPipeline(model.module, diffusion)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SD_LR)

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
            initials = initial_images.to(device)
            cond_initial = initials if cfg.INITIAL_IMAGE else None

            loss = diffusion.p_loss(model, targets, features, cond_initial)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

            if (batch_idx + 1) % cfg.SD_LOG_INTERVAL == 0 and rank == 0:
                print(
                    f"[SD] Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {epoch_loss / steps:.4f}"
                )

        loss_tensor = torch.tensor([epoch_loss, steps], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps = max(int(loss_tensor[1].item()), 1)
        avg_loss = (loss_tensor[0] / total_steps).item()

        val_metrics = None
        if rank == 0 and val_loader is not None:
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
            )

        if rank == 0:
            elapsed = time.time() - start_time
            if val_metrics:
                print(
                    f"[SD] Epoch {epoch} Loss: {avg_loss:.4f} | "
                    f"Val L1: {val_metrics['val_l1']:.4f}, PSNR: {val_metrics['val_psnr']:.2f}, "
                    f"SSIM: {val_metrics['val_ssim']:.4f}, CLIP: {val_metrics['val_clip']:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )
                metrics_logger.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_l1": val_metrics["val_l1"],
                    "val_psnr": val_metrics["val_psnr"],
                    "val_ssim": val_metrics["val_ssim"],
                    "val_clip": val_metrics["val_clip"],
                })
            else:
                print(f"[SD] Epoch {epoch} Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
                metrics_logger.log({"epoch": epoch, "train_loss": avg_loss.item()})

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
