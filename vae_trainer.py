import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.conditional_vae import ConditionalVAE
from utils.dataset import create_dataloaders
from utils.training import (
    calculate_psnr,
    calculate_ssim,
    MetricsLogger,
    save_random_sample_pairs,
    load_clip_model,
    compute_clip_metrics_batch,
)
import config as cfg


def get_device() -> torch.device:
    device_ids = getattr(cfg, "DEVICE_IDS", [0])
    if torch.cuda.is_available() and device_ids:
        return torch.device(f"cuda:{device_ids[0]}")
    return torch.device("cpu")



def save_checkpoint(path: str, vae: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    payload = {
        "epoch": epoch,
        "vae_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, vae: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    vae.load_state_dict(checkpoint["vae_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)


def _setup_ddp(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device_id = getattr(cfg, "DEVICE_IDS", list(range(torch.cuda.device_count())))[rank]
    torch.cuda.set_device(device_id)


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate(
    vae: nn.Module,
    val_loader,
    device: torch.device,
    epoch: int,
    sample_dir: str,
    clip_model,
    clip_preprocess,
):
    vae.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_l1 = 0.0
    total_clip = 0.0
    total_samples = 0
    l1_loss = nn.L1Loss()

    with torch.no_grad():
        for batch_idx, (initial, meta, target, _) in enumerate(val_loader):
            initial = initial.to(device)
            meta = meta.to(device)
            target = target.to(device)

            noise = torch.randn(target.shape[0], cfg.VAE_NOISE_DIM, device=device)
            outputs = vae(initial, meta, noise)
            reconstruction = outputs["reconstruction"]
            vae_module = vae.module if isinstance(vae, nn.parallel.DistributedDataParallel) else vae
            rec_loss = l1_loss(reconstruction, target).item()
            kl = vae_module.kl_loss(outputs["mu"], outputs["logvar"]).item()
            batch_size = target.shape[0]
            total_loss += (cfg.VAE_RECON_FACTOR * rec_loss + cfg.VAE_KL_FACTOR * kl) * batch_size
            total_l1 += rec_loss * batch_size

            batch_psnr = calculate_psnr(reconstruction, target)
            batch_ssim = calculate_ssim(reconstruction, target)
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_samples += batch_size

            if clip_model is not None and clip_preprocess is not None:
                clip_sum, _ = compute_clip_metrics_batch(reconstruction, target, clip_model, clip_preprocess, device)
                total_clip += clip_sum

            if batch_idx == 0:
                save_random_sample_pairs(
                    initial.cpu(),
                    reconstruction.detach().cpu(),
                    target.detach().cpu(),
                    sample_dir,
                    epoch,
                    prefix="vae_val",
                    num_samples=4,
                )

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_l1 = total_l1 / total_samples
    avg_clip = total_clip / max(total_samples, 1)
    return avg_loss, avg_psnr, avg_ssim, avg_l1, avg_clip


def train_single(
    vae: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | None,
    retrain: bool,
    name: str,
    version: str,
):
    save_dir = "checkpoints"
    log_dir = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    metrics_logger = MetricsLogger(log_dir, f"vae_{version}_training_log.csv")

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR, betas=(0.5, 0.999))
    l1_loss = nn.L1Loss()

    start_epoch = 0
    if retrain and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        start_epoch = load_checkpoint(checkpoint_path, vae, optimizer)

    clip_model, clip_preprocess = load_clip_model(device)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        vae.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (initial, meta, target, _) in enumerate(train_loader):
            initial = initial.to(device)
            meta = meta.to(device)
            target = target.to(device)

            noise = torch.randn(target.shape[0], cfg.VAE_NOISE_DIM, device=device)
            outputs = vae(initial, meta, noise)
            reconstruction = outputs["reconstruction"]

            rec_loss = l1_loss(reconstruction, target)
            vae_module = vae.module if isinstance(vae, nn.parallel.DistributedDataParallel) else vae
            kl = vae_module.kl_loss(outputs["mu"], outputs["logvar"])
            loss = cfg.VAE_RECON_FACTOR * rec_loss + cfg.VAE_KL_FACTOR * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)} - "
                    f"loss={loss.item():.4f}, rec={rec_loss.item():.4f}, kl={kl.item():.4f}"
                )

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = (0.0, 0.0, 0.0, 0.0, 0.0)

        if val_loader and (epoch % cfg.VAL_EPOCH == 0):
            val_metrics = evaluate(vae, val_loader, device, epoch, sample_dir, clip_model, clip_preprocess)

        avg_val_loss, avg_psnr, avg_ssim, avg_val_l1, avg_val_clip = val_metrics
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch} finished (train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
            f"L1={avg_val_l1:.4f}, CLIP={avg_val_clip:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}) | Time: {elapsed:.2f}s"
        )

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
            "val_l1": avg_val_l1,
            "val_clip": avg_val_clip,
        }
        metrics_logger.log(log_entry)

        if val_loader and (epoch % cfg.VAL_EPOCH == 0):
            ckpt_path = os.path.join(save_dir, f"vae_{version}_epoch_{epoch}.pth")
            save_checkpoint(ckpt_path, vae, optimizer, epoch)
            print(f"Checkpoint saved to {ckpt_path}")


def evaluate_ddp(
    vae: nn.Module,
    val_loader,
    device: torch.device,
    epoch: int,
    sample_dir: str,
    rank: int,
    clip_model,
    clip_preprocess,
) -> tuple[float, float, float, float, float]:
    vae.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_psnr = torch.tensor(0.0, device=device)
    total_ssim = torch.tensor(0.0, device=device)
    total_l1 = torch.tensor(0.0, device=device)
    total_clip = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0, device=device)
    l1_loss = nn.L1Loss()
    first_saved = False

    with torch.no_grad():
        for batch_idx, (initial, meta, target, _) in enumerate(val_loader):
            initial = initial.to(device)
            meta = meta.to(device)
            target = target.to(device)

            noise = torch.randn(target.shape[0], cfg.VAE_NOISE_DIM, device=device)
            outputs = vae(initial, meta, noise)
            reconstruction = outputs["reconstruction"]
            vae_module = vae.module if isinstance(vae, nn.parallel.DistributedDataParallel) else vae
            rec_loss = l1_loss(reconstruction, target).item()
            kl = vae_module.kl_loss(outputs["mu"], outputs["logvar"]).item()
            batch_size = target.shape[0]

            local_loss = (cfg.VAE_RECON_FACTOR * rec_loss + cfg.VAE_KL_FACTOR * kl) * batch_size
            total_loss += local_loss
            total_l1 += rec_loss * batch_size

            batch_psnr = calculate_psnr(reconstruction, target)
            batch_ssim = calculate_ssim(reconstruction, target)
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_samples += batch_size

            if clip_model is not None and clip_preprocess is not None:
                clip_sum, _ = compute_clip_metrics_batch(reconstruction, target, clip_model, clip_preprocess, device)
                total_clip += clip_sum

            if not first_saved and rank == 0:
                save_random_sample_pairs(
                    initial.cpu(),
                    reconstruction.detach().cpu(),
                    target.detach().cpu(),
                    sample_dir,
                    epoch,
                    prefix="vae_val",
                    num_samples=4,
                )
                first_saved = True

    _tensor = torch.stack([
        total_loss,
        total_psnr,
        total_ssim,
        total_l1,
        total_clip,
        total_samples.to(torch.float32),
    ])
    dist.all_reduce(_tensor, op=dist.ReduceOp.SUM)

    total_loss, total_psnr, total_ssim, total_l1, total_clip, total_samples = _tensor.tolist()
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_l1 = total_l1 / total_samples
    avg_clip = total_clip / max(total_samples, 1)
    return avg_loss, avg_psnr, avg_ssim, avg_l1, avg_clip


def _ddp_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    _setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    feature_dim = train_loader.dataset.input_data.shape[1]

    clip_model, clip_preprocess = load_clip_model(device)

    vae = ConditionalVAE(
        encoded_dim=cfg.VAE_ENCODER_DIM,
        meta_dim=feature_dim,
        noise_dim=cfg.VAE_NOISE_DIM,
        latent_dim=cfg.VAE_LATENT_DIM,
        hidden_dim=cfg.VAE_HIDDEN_DIM,
        channels=cfg.CHANNELS,
    ).to(device)
    vae = nn.SyncBatchNorm.convert_sync_batchnorm(vae)
    ddp_device_ids = [cfg.DEVICE_IDS[rank]] if hasattr(cfg, "DEVICE_IDS") else [rank]
    vae = DDP(vae, device_ids=ddp_device_ids)

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR, betas=(0.5, 0.999))

    start_epoch = 0
    if args.retrain and args.checkpoint and os.path.exists(args.checkpoint):
        start_epoch = load_checkpoint(args.checkpoint, vae.module, optimizer)

    save_dir = "checkpoints"
    log_dir = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    metrics_logger = MetricsLogger(log_dir, f"vae_ddp_training_log.csv") if rank == 0 else None

    try:
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            vae.train()
            total_loss = 0.0
            total_samples = 0
            epoch_start = time.time()

            for batch_idx, (initial, meta, target, _) in enumerate(train_loader):
                initial = initial.to(device)
                meta = meta.to(device)
                target = target.to(device)

                noise = torch.randn(target.shape[0], cfg.VAE_NOISE_DIM, device=device)
                outputs = vae(initial, meta, noise)
                reconstruction = outputs["reconstruction"]

                vae_module = vae.module if isinstance(vae, nn.parallel.DistributedDataParallel) else vae
                rec_loss = nn.L1Loss()(reconstruction, target)
                kl = vae_module.kl_loss(outputs["mu"], outputs["logvar"])
                loss = cfg.VAE_RECON_FACTOR * rec_loss + cfg.VAE_KL_FACTOR * kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = target.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            loss_tensor = torch.tensor([total_loss, total_samples], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            summed_loss, summed_samples = loss_tensor.tolist()
            avg_train_loss = summed_loss / max(summed_samples, 1)

            # Ensure both ranks execute validation or skip it together
            val_loss, val_psnr, val_ssim, val_l1, val_clip = (0.0, 0.0, 0.0, 0.0, 0.0)
            should_validate = val_loader is not None and (epoch % cfg.VAL_EPOCH == 0)
            if should_validate:
                val_loss, val_psnr, val_ssim, val_l1, val_clip = evaluate_ddp(
                    vae,
                    val_loader,
                    device,
                    epoch,
                    sample_dir,
                    rank,
                    clip_model,
                    clip_preprocess,
                )

            if rank == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"DDP Epoch {epoch} finished (train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"L1={val_l1:.4f}, CLIP={val_clip:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}) | Time: {elapsed:.2f}s"
                )

                metrics_logger.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_l1": val_l1,
                        "val_clip": val_clip,
                    }
                )

                if should_validate:
                    ckpt_path = os.path.join(save_dir, f"vae_ddp_epoch_{epoch}.pth")
                    save_checkpoint(ckpt_path, vae.module, optimizer, epoch)
                    print(f"Checkpoint saved to {ckpt_path}")

            # keep ranks in sync each epoch to avoid hanging collectives
            dist.barrier()
    finally:
        _cleanup_ddp()


def train_distributed(args: argparse.Namespace) -> None:
    device_ids = getattr(cfg, "DEVICE_IDS", list(range(torch.cuda.device_count())))
    world_size = len(device_ids)

    if world_size <= 1:
        raise RuntimeError("Distributed training requires more than one device")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested world_size={world_size} but only {torch.cuda.device_count()} CUDA devices available"
        )

    mp.spawn(
        _ddp_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the conditional VAE")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--name", type=str, default="conditional_vae")
    parser.add_argument("--version", type=str, default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_ids = getattr(cfg, "DEVICE_IDS", [0])
    if torch.cuda.is_available() and len(device_ids) > 1:
        print(
            f"Launching DDP VAE training on devices {device_ids} for {args.epochs} epochs "
            f"(retrain={bool(args.retrain)}, checkpoint={args.checkpoint})"
        )
        train_distributed(args)
        return

    device = get_device()

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    feature_dim = train_loader.dataset.input_data.shape[1]

    vae = ConditionalVAE(
        encoded_dim=cfg.VAE_ENCODER_DIM,
        meta_dim=feature_dim,
        noise_dim=cfg.VAE_NOISE_DIM,
        latent_dim=cfg.VAE_LATENT_DIM,
        hidden_dim=cfg.VAE_HIDDEN_DIM,
        channels=cfg.CHANNELS
    ).to(device)

    print("Starting VAE training")
    train_single(
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint or None,
        retrain=bool(args.retrain),
        name=args.name,
        version=args.version,
    )


if __name__ == "__main__":
    main()
