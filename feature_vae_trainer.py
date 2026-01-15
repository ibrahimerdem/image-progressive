import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.experimental_vae import FeatureVAE
from utils.dataset import create_dataloaders
from utils.training import (
    calculate_psnr,
    calculate_ssim,
    MetricsLogger,
    save_random_sample_pairs,
)
import config as cfg


def get_device() -> torch.device:
    device_ids = getattr(cfg, "DEVICE_IDS", [0])
    if torch.cuda.is_available() and device_ids:
        return torch.device(f"cuda:{device_ids[0]}")
    return torch.device("cpu")


def generate_images_from_features(
    vae: nn.Module,
    val_loader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 16,
):
    """Generate images from features and save them."""
    vae.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    generated_images = []
    initial_images = []
    target_images = []
    
    with torch.no_grad():
        for batch_idx, (initial, features, target, _) in enumerate(val_loader):
            if len(generated_images) >= num_samples:
                break
                
            initial = initial.to(device)
            features = features.to(device)
            target = target.to(device)
            
            generated = vae.generate(features)
            
            batch_size = min(generated.shape[0], num_samples - len(generated_images))
            generated_images.append(generated[:batch_size].cpu())
            initial_images.append(initial[:batch_size].cpu())
            target_images.append(target[:batch_size].cpu())
    
    if generated_images:
        generated_images = torch.cat(generated_images, dim=0)
        initial_images = torch.cat(initial_images, dim=0)
        target_images = torch.cat(target_images, dim=0)
        
        save_random_sample_pairs(
            initial_images,
            generated_images,
            target_images,
            output_dir,
            epoch=0,
            prefix="final_generation",
            num_samples=num_samples,
        )
        
        print(f"Generated {len(generated_images)} images saved to {output_dir}")


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


def _setup_ddp(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device_ids = getattr(cfg, "DEVICE_IDS", list(range(world_size)))
    torch.cuda.set_device(device_ids[rank])


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate(
    vae: nn.Module,
    val_loader,
    device: torch.device,
    epoch: int,
    sample_dir: str,
    prefix: str,
):
    vae.eval()
    rec_loss_total = 0.0
    rec_psnr_total = 0.0
    rec_ssim_total = 0.0
    gen_loss_total = 0.0
    gen_psnr_total = 0.0
    gen_ssim_total = 0.0
    total_samples = 0
    l1_loss = nn.L1Loss()

    with torch.no_grad():
        for batch_idx, (initial, features, target, _) in enumerate(val_loader):
            initial = initial.to(device)
            features = features.to(device)
            target = target.to(device)

            # Reconstruction metrics
            outputs = vae(features, target)
            reconstruction = outputs["reconstruction"]
            rec_loss = l1_loss(reconstruction, target).item()
            batch_size = target.shape[0]
            rec_loss_total += rec_loss * batch_size
            rec_psnr_total += calculate_psnr(reconstruction, target) * batch_size
            rec_ssim_total += calculate_ssim(reconstruction, target) * batch_size
            
            # Generation metrics
            generated = vae.generate(features)
            gen_loss = l1_loss(generated, target).item()
            gen_loss_total += gen_loss * batch_size
            gen_psnr_total += calculate_psnr(generated, target) * batch_size
            gen_ssim_total += calculate_ssim(generated, target) * batch_size
            
            total_samples += batch_size

            if batch_idx == 0:
                save_random_sample_pairs(
                    initial.cpu(),
                    reconstruction.detach().cpu(),
                    target.detach().cpu(),
                    sample_dir,
                    epoch,
                    prefix=f"{prefix}_rec",
                    num_samples=4,
                )
                save_random_sample_pairs(
                    initial.cpu(),
                    generated.detach().cpu(),
                    target.detach().cpu(),
                    sample_dir,
                    epoch,
                    prefix=f"{prefix}_gen",
                    num_samples=4,
                )

    if total_samples == 0:
        return {}

    return {
        "rec_loss": rec_loss_total / total_samples,
        "rec_psnr": rec_psnr_total / total_samples,
        "rec_ssim": rec_ssim_total / total_samples,
        "gen_loss": gen_loss_total / total_samples,
        "gen_psnr": gen_psnr_total / total_samples,
        "gen_ssim": gen_ssim_total / total_samples,
    }


def evaluate_ddp(
    vae: nn.Module,
    val_loader,
    device: torch.device,
    epoch: int,
    sample_dir: str,
    rank: int,
    prefix: str,
):
    vae.eval()
    rec_loss_total = torch.tensor(0.0, device=device)
    rec_psnr_total = torch.tensor(0.0, device=device)
    rec_ssim_total = torch.tensor(0.0, device=device)
    gen_loss_total = torch.tensor(0.0, device=device)
    gen_psnr_total = torch.tensor(0.0, device=device)
    gen_ssim_total = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0, device=device)
    l1_loss = nn.L1Loss()
    first_saved = False

    with torch.no_grad():
        for batch_idx, (initial, features, target, _) in enumerate(val_loader):
            initial = initial.to(device)
            features = features.to(device)
            target = target.to(device)

            # Reconstruction metrics
            outputs = vae(features, target)
            reconstruction = outputs["reconstruction"]
            rec_loss = l1_loss(reconstruction, target).item()
            batch_size = target.shape[0]
            rec_loss_total += rec_loss * batch_size
            rec_psnr_total += calculate_psnr(reconstruction, target) * batch_size
            rec_ssim_total += calculate_ssim(reconstruction, target) * batch_size
            
            # Generation metrics
            generated = vae.generate(features)
            gen_loss = l1_loss(generated, target).item()
            gen_loss_total += gen_loss * batch_size
            gen_psnr_total += calculate_psnr(generated, target) * batch_size
            gen_ssim_total += calculate_ssim(generated, target) * batch_size
            
            total_samples += batch_size

            if rank == 0 and not first_saved:
                save_random_sample_pairs(
                    initial.cpu(),
                    reconstruction.detach().cpu(),
                    target.detach().cpu(),
                    sample_dir,
                    epoch,
                    prefix=f"{prefix}_rec",
                    num_samples=4,
                )
                save_random_sample_pairs(
                    initial.cpu(),
                    generated.detach().cpu(),
                    target.detach().cpu(),
                    sample_dir,
                    epoch,
                    prefix=f"{prefix}_gen",
                    num_samples=4,
                )
                first_saved = True

    _tensor = torch.stack([
        rec_loss_total,
        rec_psnr_total,
        rec_ssim_total,
        gen_loss_total,
        gen_psnr_total,
        gen_ssim_total,
        total_samples.to(torch.float32),
    ])
    dist.all_reduce(_tensor, op=dist.ReduceOp.SUM)
    rec_loss, rec_psnr, rec_ssim, gen_loss, gen_psnr, gen_ssim, total_samples = _tensor.tolist()

    if total_samples == 0:
        return {}

    return {
        "rec_loss": rec_loss / total_samples,
        "rec_psnr": rec_psnr / total_samples,
        "rec_ssim": rec_ssim / total_samples,
        "gen_loss": gen_loss / total_samples,
        "gen_psnr": gen_psnr / total_samples,
        "gen_ssim": gen_ssim / total_samples,
    }


def train_single(
    vae: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | None,
    retrain: bool,
    version: str,
):
    save_dir = "checkpoints"
    log_dir = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    metrics_logger = MetricsLogger(log_dir, f"feature_vae_{version}_training_log.csv")

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR, betas=(0.5, 0.999))
    l1_loss = nn.L1Loss()

    start_epoch = 0
    if retrain and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        start_epoch = load_checkpoint(checkpoint_path, vae, optimizer)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        vae.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        kl_epoch_total = 0.0
        total_samples = 0
        for batch_idx, (initial, features, target, _) in enumerate(train_loader):
            initial = initial.to(device)
            features = features.to(device)
            target = target.to(device)

            outputs = vae(features, target)
            reconstruction = outputs["reconstruction"]
            mu = outputs["mu"]
            logvar = outputs["logvar"]

            rec_loss = l1_loss(reconstruction, target)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / max(mu.shape[0], 1)
            loss = rec_loss + cfg.VAE_KL_FACTOR * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = target.shape[0]
            epoch_loss += loss.item()
            kl_epoch_total += kl_loss.item() * batch_size
            total_samples += batch_size

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)} - loss={loss.item():.4f}"
                )

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        avg_kl_loss = kl_epoch_total / max(total_samples, 1)
        val_metrics = {}

        if val_loader and (epoch % cfg.VAL_EPOCH == 0):
            val_metrics = evaluate(vae, val_loader, device, epoch, sample_dir, prefix="feature_vae_val")

        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch} finished (train_loss={avg_train_loss:.4f}, kl_loss={avg_kl_loss:.6f}) | "
            f"Rec(loss={val_metrics.get('rec_loss', 0.0):.4f}, PSNR={val_metrics.get('rec_psnr', 0.0):.2f}, SSIM={val_metrics.get('rec_ssim', 0.0):.4f}) | "
            f"Gen(loss={val_metrics.get('gen_loss', 0.0):.4f}, PSNR={val_metrics.get('gen_psnr', 0.0):.2f}, SSIM={val_metrics.get('gen_ssim', 0.0):.4f}) | "
            f"Time: {elapsed:.2f}s"
        )

        log_dict = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "kl_loss": avg_kl_loss,
        }
        log_dict.update(val_metrics)
        metrics_logger.log(log_dict)

        if val_loader and (epoch % cfg.VAL_EPOCH == 0):
            ckpt_path = os.path.join(save_dir, f"feature_vae_{version}_epoch_{epoch}.pth")
            save_checkpoint(ckpt_path, vae, optimizer, epoch)
            print(f"Checkpoint saved to {ckpt_path}")
    
    # Generate images after training
    print("\nGenerating final images from features...")
    generate_images_from_features(
        vae,
        val_loader,
        device,
        output_dir=os.path.join(sample_dir, "final_generations"),
        num_samples=16,
    )


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

    log_dir = os.path.join("checkpoints", "logs")
    sample_dir = os.path.join("checkpoints", "samples")
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    feature_dim = train_loader.dataset.input_data.shape[1]
    vae = FeatureVAE(feature_dim=feature_dim, image_channels=cfg.CHANNELS).to(device)
    vae = nn.SyncBatchNorm.convert_sync_batchnorm(vae)
    ddp_device_ids = [cfg.DEVICE_IDS[rank]] if hasattr(cfg, "DEVICE_IDS") else [rank]
    vae = DDP(vae, device_ids=ddp_device_ids)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR, betas=(0.5, 0.999))

    start_epoch = 0
    if args.retrain and args.checkpoint and os.path.exists(args.checkpoint):
        start_epoch = load_checkpoint(args.checkpoint, vae.module, optimizer)

    metrics_logger = MetricsLogger(log_dir, f"feature_vae_ddp_training_log.csv") if rank == 0 else None

    try:
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            if hasattr(train_loader, "sampler") and isinstance(
                train_loader.sampler,
                torch.utils.data.distributed.DistributedSampler,
            ):
                train_loader.sampler.set_epoch(epoch)

            vae.train()
            total_loss = 0.0
            total_samples = 0
            kl_batch_total = 0.0
            epoch_start = time.time()

            for batch_idx, (initial, features, target, _) in enumerate(train_loader):
                initial = initial.to(device)
                features = features.to(device)
                target = target.to(device)

                outputs = vae(features, target)
                reconstruction = outputs["reconstruction"]
                mu = outputs["mu"]
                logvar = outputs["logvar"]

                rec_loss = nn.L1Loss()(reconstruction, target)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / max(mu.shape[0], 1)
                loss = rec_loss + cfg.VAE_KL_FACTOR * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = target.shape[0]
                total_loss += loss.item() * batch_size
                kl_batch_total += kl_loss.item() * batch_size
                total_samples += batch_size

            loss_tensor = torch.tensor(
                [total_loss, total_samples, kl_batch_total],
                device=device,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            summed_loss, summed_samples, summed_kl = loss_tensor.tolist()
            avg_train_loss = summed_loss / max(summed_samples, 1)
            avg_kl_loss = summed_kl / max(summed_samples, 1)

            val_metrics = {}
            if val_loader and (epoch % cfg.VAL_EPOCH == 0):
                val_metrics = evaluate_ddp(
                    vae,
                    val_loader,
                    device,
                    epoch,
                    sample_dir,
                    rank,
                    prefix="feature_vae_val",
                )

            if rank == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"DDP Epoch {epoch} finished (train_loss={avg_train_loss:.4f}, kl_loss={avg_kl_loss:.6f}) | "
                    f"Rec(loss={val_metrics.get('rec_loss', 0.0):.4f}, PSNR={val_metrics.get('rec_psnr', 0.0):.2f}, SSIM={val_metrics.get('rec_ssim', 0.0):.4f}) | "
                    f"Gen(loss={val_metrics.get('gen_loss', 0.0):.4f}, PSNR={val_metrics.get('gen_psnr', 0.0):.2f}, SSIM={val_metrics.get('gen_ssim', 0.0):.4f}) | "
                    f"Time: {elapsed:.2f}s"
                )

                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "kl_loss": avg_kl_loss,
                }
                log_dict.update(val_metrics)
                metrics_logger.log(log_dict)

                if val_loader and (epoch % cfg.VAL_EPOCH == 0):
                    ckpt_path = os.path.join("checkpoints", f"feature_vae_ddp_epoch_{epoch}.pth")
                    save_checkpoint(ckpt_path, vae.module, optimizer, epoch)
                    print(f"Checkpoint saved to {ckpt_path}")

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
    parser = argparse.ArgumentParser(description="Train the feature-conditioned VAE")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--name", type=str, default="feature_vae")
    parser.add_argument("--version", type=str, default="ddp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_ids = getattr(cfg, "DEVICE_IDS", [0])
    if torch.cuda.is_available() and len(device_ids) > 1:
        print(
            f"Launching DDP FeatureVAE training on devices {device_ids} for {args.epochs} epochs "
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
    vae = FeatureVAE(feature_dim=feature_dim, image_channels=cfg.CHANNELS).to(device)

    print("Starting FeatureVAE training")
    train_single(
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint or None,
        retrain=bool(args.retrain),
        version=args.version,
    )


if __name__ == "__main__":
    main()
