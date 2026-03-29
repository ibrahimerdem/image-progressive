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


def _denormalize(tensor, mins, maxs):
    mins = mins.to(tensor.device)
    maxs = maxs.to(tensor.device)
    real_denorm = (tensor + 1) / 2 * (maxs - mins) + mins
    return real_denorm
 
 
def _denorm_metrics(preds_norm, targets_norm, t_mins, t_maxs):
    preds   = _denormalize(preds_norm,   t_mins, t_maxs)
    targets = _denormalize(targets_norm, t_mins, t_maxs)
 
    mae  = F.l1_loss(preds, targets).item()
    mse  = F.mse_loss(preds, targets).item()
    mape = (torch.abs(preds - targets) / (torch.abs(targets) + 1e-8)).mean().item() * 100.0
    return {"mae": mae, "mse": mse, "mape": mape}

def _print_sample_predictions(preds_norm, targets_norm, t_mins, t_maxs,
                               col_names, n=5):

    preds   = _denormalize(preds_norm.cpu().float(),   t_mins.cpu(), t_maxs.cpu())
    targets = _denormalize(targets_norm.cpu().float(), t_mins.cpu(), t_maxs.cpu())
 
    n = min(n, preds.size(0))
    D = preds.size(1)
 
    col_w = 12
    # Header
    header = f"{'#':>3}  "
    header += "  ".join(
        f"{'pred_' + name:>{col_w}}  {'true_' + name:>{col_w}}"
        for name in col_names
    )
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)
 
    for i in range(n):
        row = f"{i:>3}  "
        row += "  ".join(
            f"{preds[i, d].item():{col_w}.4f}  {targets[i, d].item():{col_w}.4f}"
            for d in range(D)
        )
        print(row)
    print()
 


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


def _run_validation(model, val_loader, device, amp_ctx, t_mins, t_maxs, col_names):

    model.eval()
 
    total_mae  = 0.0
    total_mse  = 0.0
    total_mape = 0.0
    steps = 0
    first_batch_shown = False
 
    with torch.no_grad():
        for features, initial_images, target_images, targets in val_loader:
            features       = features.to(device)
            initial_images = initial_images.to(device)
            target_images  = target_images.to(device)
            targets        = targets.to(device)
 
            with amp_ctx():
                preds = model(features, target_images)
 
            # Clamp to valid range before denormalizing
            preds_clamped = preds.clamp(-1.0, 1.0)
 
            m = _denorm_metrics(preds_clamped, targets, t_mins, t_maxs)
            total_mae  += m["mae"]
            total_mse  += m["mse"]
            total_mape += m["mape"]
            steps += 1
 
            # Show first-batch sample table (first 5 rows) once per validation
            if not first_batch_shown:
                print("[VAL] Sample predictions vs ground-truth (denormalized, first batch):")
                _print_sample_predictions(
                    preds_clamped, targets,
                    t_mins, t_maxs,
                    col_names, n=5,
                )
                first_batch_shown = True
 
    model.train()
 
    n = max(steps, 1)
    return {
        "val_mae":  total_mae  / n,
        "val_mse":  total_mse  / n,
        "val_mape": total_mape / n,
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

    if len(train_loader) == 0:
        raise ValueError(f"[E] Training dataloader is empty! Check dataset configuration.")
    if val_loader and len(val_loader) == 0:
        print(f"[W] Validation dataloader is empty!")
    
    if rank == 0:
        print(f"[S] Training batches per epoch:   {len(train_loader)}")
        print(f"[S] Validation batches per epoch: {len(val_loader) if val_loader else 0}")
        print(f"[S] Training features shape:      {train_loader.dataset.input_data[0].shape}")
        print(f"[S] Validation features shape:    {val_loader.dataset.input_data[0].shape}")

    t_mins = torch.tensor(cfg.TARGET_MINS, dtype=torch.float32)
    t_maxs = torch.tensor(cfg.TARGET_MAXS, dtype=torch.float32)
    col_names = cfg.TARGET_FEATURE_COLUMNS
    
    base_model = Basic_Triplet()
    model = DDP(base_model.to(device), device_ids=[cfg.DEVICE_IDS[rank]])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
    criterion = nn.L1Loss()

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
            if rank == 0:
                print(f"[S] Resumed from {checkpoint_path} at epoch {start_epoch + 1}")
        elif rank == 0:
            print(f"[S] Checkpoint {checkpoint_path} not found — starting from scratch")
    elif retrain and not checkpoint_path:
        if rank == 0:
            print("[W] --retrain set but no checkpoint path given — starting from scratch")
 
    metrics_logger = MetricsLogger(log_dir, f"{version}_log.csv") if rank == 0 else None

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
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
                preds = model(features, target_images)
                loss = criterion(preds, targets)

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
                with torch.no_grad():
                    denorm_m = _denorm_metrics(
                        preds.detach().clamp(-1.0, 1.0), targets, t_mins, t_maxs
                    )
                print(
                    f"[S] Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss (norm MAE): {loss.item():.4f} | "
                    f"MAE: {denorm_m['mae']:.4f} | "
                    f"MSE: {denorm_m['mse']:.4f} | "
                    f"MAPE: {denorm_m['mape']:.2f}% | "
                    f"Grad Norm: {grad_norm:.4f}"
                )
 

        if rank == 0:
           print(
                f"[S] Epoch {epoch} — all {len(train_loader)} batches done "
                f"in {time.time() - start_time:.2f}s"
            )

        loss_tensor = torch.tensor([epoch_loss, float(steps)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps   = max(int(loss_tensor[1].item()), 1)
        avg_norm_loss = (loss_tensor[0] / total_steps).item()

        do_validate = val_loader is not None and (
            cfg.VAL_INTERVAL <= 1 or epoch % cfg.VAL_INTERVAL == 0
        )

        do_validate_tensor = torch.tensor(int(do_validate), device=device)
        dist.broadcast(do_validate_tensor, src=0)
        do_validate = bool(do_validate_tensor.item())

        val_metrics = None
        if do_validate:
            # ALL ranks participate in validation (DDP requires this)
            if rank == 0:
                print(f"\n[S] === Validation — Epoch {epoch} ===")

            model.eval()
            total_mae = total_mse = total_mape = 0.0
            steps = 0
            first_batch_shown = False
            with torch.no_grad():
                for features, initial_images, target_images, targets in val_loader:
                    features       = features.to(device)
                    initial_images = initial_images.to(device)
                    target_images  = target_images.to(device)
                    targets        = targets.to(device)

                    with amp_ctx():
                        preds = model(features, target_images)  # all ranks forward

                    if rank == 0:
                        preds_clamped = preds.clamp(-1.0, 1.0)
                        m = _denorm_metrics(preds_clamped, targets, t_mins, t_maxs)
                        total_mae  += m["mae"]
                        total_mse  += m["mse"]
                        total_mape += m["mape"]
                        steps += 1

                        if not first_batch_shown:
                            print("[VAL] Sample predictions vs ground-truth (denormalized, first batch):")
                            _print_sample_predictions(
                                preds_clamped, targets, t_mins, t_maxs, col_names, n=5,
                            )
                            first_batch_shown = True

            model.train()

            if rank == 0:
                n = max(steps, 1)
                val_metrics = {
                    "val_mae":  total_mae  / n,
                    "val_mse":  total_mse  / n,
                    "val_mape": total_mape / n,
                }
                print(
                    f"[S] Validation summary — "
                    f"MAE: {val_metrics['val_mae']:.4f} | "
                    f"MSE: {val_metrics['val_mse']:.4f} | "
                    f"MAPE: {val_metrics['val_mape']:.2f}%"
                )

        dist.barrier()

        if rank == 0:
            elapsed = time.time() - start_time
            if val_metrics:
                print(
                    f"[S] Epoch {epoch} | "
                    f"Train Loss (norm MAE): {avg_norm_loss:.4f} | "
                    f"Val MAE: {val_metrics['val_mae']:.4f} | "
                    f"Val MSE: {val_metrics['val_mse']:.4f} | "
                    f"Val MAPE: {val_metrics['val_mape']:.2f}% | "
                    f"Time: {elapsed:.2f}s"
                )
                metrics_logger.log({
                    "epoch":      epoch,
                    "train_loss": avg_norm_loss,
                    "val_mae":    val_metrics["val_mae"],
                    "val_mse":    val_metrics["val_mse"],
                    "val_mape":   val_metrics["val_mape"],
                })
                saved_path = _save_checkpoint(model, optimizer, epoch, save_dir, version)
                print(f"[S] Checkpoint saved: {saved_path}")
            else:
                print(
                    f"[S] Epoch {epoch} | "
                    f"Train Loss (norm MAE): {avg_norm_loss:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )
                metrics_logger.log({"epoch": epoch, "train_loss": avg_norm_loss})

            

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