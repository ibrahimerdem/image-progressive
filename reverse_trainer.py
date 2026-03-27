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

from models.reverse_prediction import Basic_Triplet

from utils.dataset import create_dataloaders
from utils.training import MetricsLogger


def _setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    timeout_minutes = getattr(cfg, "DDP_TIMEOUT", 10)
    timeout = torch.distributed.timedelta(minutes=timeout_minutes)
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    device_id = cfg.DEVICE_IDS[rank]
    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _save_checkpoint(model, optimizer, epoch, save_dir, name):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    filename = os.path.join(save_dir, f"{name}_epoch_{epoch:04d}.pth")
    torch.save(checkpoint, filename)
    return filename


def _load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)


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
        raise ValueError(f"[E] Training dataloader is empty! Check dataset configuration.")
    if val_loader and len(val_loader) == 0:
        print(f"[W] Validation dataloader is empty!")
    
    if rank == 0:
        print(f"[S] Training batches per epoch: {len(train_loader)}")
        print(f"[S] Validation batches per epoch: {len(val_loader) if val_loader else 0}")

    sample_train_features = train_loader.dataset.input_data[0]
    sample_val_features = val_loader.dataset.input_data[0]
    
    print(f"[S] Training features: {sample_train_features.shape}")
    print(f"[S] Validation features: {sample_val_features.shape}")
    
    base_model = Basic_Triplet()

    model = DDP(base_model.to(device), device_ids=[cfg.DEVICE_IDS[rank]])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else nullcontext()

    save_dir = os.path.join("checkpoints", "reverse")
    log_dir = os.path.join(save_dir, "logs")

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    start_epoch = 0
    if retrain and checkpoint_path:
        if os.path.exists(checkpoint_path):
            start_epoch = _load_checkpoint(model, optimizer, checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if rank == 0:
                print(f"[S] Resumed from {checkpoint_path} at epoch {start_epoch + 1}")
        elif rank == 0:
            print(f"[S] Checkpoint {checkpoint_path} not found, starting from scratch")
    elif retrain and not checkpoint_path:
        print(f"[W] No checkpoint directory found, starting from scratch")

    metrics_logger = MetricsLogger(log_dir, f"{version}_log.csv") if rank == 0 else None

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        steps = 0
        model.train()
    
        start_time = time.time()
        for batch_idx, (features, initial_images, target_images, targets) in enumerate(train_loader):
            initial_images = initial_images.to(device)
            features = features.to(device)
            target_images = target_images.to(device)
            targets = targets.to(device)

            if batch_idx == 0 and rank == 0:
                print(f"[S] Epoch {epoch} - batch 0/{len(train_loader)} "
                      f"features: {features.shape}, targets: {targets.shape}")
                if cfg.INITIAL_IMAGE:
                    print(f"[S] Initial images: {initial_images.shape}")

            optimizer.zero_grad()
            with amp_ctx():
                loss_dict = nn.L1Loss(model, targets, features)
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

            epoch_loss += loss.item()
            steps += 1

            if (batch_idx + 1) % cfg.LOG_INTERVAL == 0 and rank == 0:
                mae_loss_val = loss_metrics.get('mae_loss', loss.item())
                print(
                    f"[S] Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} | "
                    f"MAE: {mae_loss_val:.4f} | "
                    f"Grad Norm: {grad_norm:.4f}"
                )

        if rank == 0:
            print(f"[S] Epoch {epoch} - Completed all {len(train_loader)} batches in {time.time() - start_time:.2f}s")

        loss_tensor = torch.tensor([epoch_loss, steps], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps = max(int(loss_tensor[1].item()), 1)
        avg_loss = (loss_tensor[0] / total_steps).item()

        if rank == 0:
            print(f"[S] Rank {rank} finished all_reduce for epoch {epoch}")

        if dist.is_initialized():
            if rank == 0:
                print(f"[S] Rank {rank} entering first barrier before validation")
            dist.barrier()
            if rank == 0:
                print(f"[S] Rank {rank} passed first barrier")

        val_metrics = None
        should_validate = (
            rank == 0
            and val_loader is not None
            and (cfg.VAL_EPOCH <= 1 or epoch % cfg.VAL_EPOCH == 0)
        )
        if should_validate:
            print(f"[S] Rank {rank} starting validation")
            
            # run validation

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            elapsed = time.time() - start_time
            if val_metrics:
                
                print(
                    f"[S] Epoch {epoch} Loss: {avg_loss:.4f} | "
                    f"Val MAE: {val_metrics['val_mae']:.4f} | "
                    f"Val MSE: {val_metrics['val_mse']:.4f} | "
                    f"Val MAPE: {val_metrics['val_mape']:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )

                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_mae": val_metrics["val_mae"],
                    "val_mse": val_metrics["val_mse"],
                    "val_mape": val_metrics["val_mape"]
                }
                  
                metrics_logger.log(log_dict)
                
            else:
                print(f"[S] Epoch {epoch} Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
                metrics_logger.log({"epoch": epoch, "train_loss": avg_loss})

            if should_validate:
                saved_path = _save_checkpoint(model, optimizer, epoch, save_dir, version)
                print(f"[S] Checkpoint saved: {saved_path}")

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
    parser = argparse.ArgumentParser(description="Training script with DDP")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    retrain_flag = bool(args.retrain)
    checkpoint_path = args.checkpoint or None

    print(f"Launching DDP training on devices {cfg.DEVICE_IDS}")
    train_distributed(
        epochs=args.epochs,
        retrain=retrain_flag,
        checkpoint_path=checkpoint_path,
        version=cfg.VERSION_NAME,
    )


if __name__ == "__main__":
    main()