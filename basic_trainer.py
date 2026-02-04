import argparse
import time
import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from models.multimodal_basic import Generator, Discriminator
from utils.training import (
    calculate_psnr,
    calculate_ssim,
    MetricsLogger,
    load_clip_model,
    compute_clip_metrics_batch,
    save_random_sample_pairs,
    load_checkpoint,
)
from utils.dataset import create_dataloaders
import config as cfg


def _configure_torch_perf():
    """Enable safe performance knobs for conv-heavy models.

    - cudnn.benchmark: speeds up fixed-size convolutions
    - TF32: improves matmul/conv throughput on Ampere+ with minimal quality impact
    """
    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def train(
    generator,
    discriminator,
    train_loader,
    val_loader,
    num_epochs,
    device,
    reload,
    checkpoint_path=None,
    name=None,
    version=None
):

    _configure_torch_perf()

    save_dir = "checkpoints"
    log_dir = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    model_name = name if name else "model"
    ver = version if version else "default"

    generator.to(device)
    discriminator.to(device)

    metrics_logger = MetricsLogger(log_dir, f"{model_name}_{ver}_training_log.csv")

    criterion = nn.BCEWithLogitsLoss().to(device)  # Changed for AMP compatibility
    l2_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.FIXED_G_LR, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.FIXED_D_LR, betas=(0.5, 0.999))

    start_epoch = 0
    if reload and checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path} for single-GPU training")
            start_epoch, _ = load_checkpoint(
                checkpoint_path,
                generator.module if isinstance(generator, nn.DataParallel) else generator,
                discriminator.module if isinstance(discriminator, nn.DataParallel) else discriminator,
                optimizer_g=g_optimizer,
                optimizer_d=d_optimizer,
            )
        else:
            print(f"Checkpoint path {checkpoint_path} not found, starting from scratch")

    # Prepare CLIP for validation metrics
    clip_model, clip_preprocess = load_clip_model(device)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        start_time = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        generator.train()
        discriminator.train()

        for idx, (input_image, input_feat, target_image, wrong_image) in enumerate(train_loader):
            batch_time = time.time()

            # Move data to device
            input_image = input_image.to(device) if cfg.INITIAL_IMAGE else None
            images = target_image.to(device)
            wrong_images = wrong_image.to(device)
            features = input_feat.to(device)  # Now called "features" not "embeddings"
            batch_size = images.size(0)

            # --- Train Discriminator ---
            d_optimizer.zero_grad()

            noise = torch.randn(batch_size, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, features, input_image)

            real_out, real_act = discriminator(images, features)
            d_loss_real = criterion(real_out, torch.full_like(real_out, 1.0, device=device))

            wrong_out, _ = discriminator(wrong_images, features)
            d_loss_wrong = criterion(wrong_out, torch.full_like(wrong_out, 0.0, device=device))

            fake_out, _ = discriminator(fake_images.detach(), features)
            d_loss_fake = criterion(fake_out, torch.full_like(fake_out, 0.0, device=device))

            d_loss = d_loss_real + d_loss_wrong + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # --- Train Generator ---
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, features, input_image)

            out_fake, act_fake = discriminator(fake_images, features)
            out_real, act_real = discriminator(images, features)

            g_bce = cfg.BCE_FACTOR * criterion(out_fake, torch.full_like(out_fake, 1.0, device=device))
            g_l1 = cfg.L1_FACTOR * l1_loss(fake_images, images)
            g_l2 = cfg.L2_FACTOR * l2_loss(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())

            g_loss = g_bce + g_l1 + g_l2
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            if (idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch}/{start_epoch + num_epochs}] Batch [{idx+1}/{len(train_loader)}] "
                    f"Loss_D: {d_loss.item():.4f}, Loss_G: {g_loss.item():.4f} "
                    f"Time: {time.time() - batch_time:.2f}s"
                )

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)

        if (val_loader is not None) and (cfg.VAL_EPOCH <= 1 or (epoch) % cfg.VAL_EPOCH == 0):
            # --- Validation: reconstruction + CLIP score + PSNR/SSIM ---
            val_l1_total = 0.0
            val_psnr_sum = 0.0
            val_ssim_sum = 0.0
            val_clip_sum = 0.0
            val_count = 0

            generator.eval()
            discriminator.eval()

            with torch.no_grad():
                for val_batch_idx, (v_input_image, v_input_feat, v_target_image, _) in enumerate(val_loader):
                    # Move data to device
                    v_input_image = v_input_image.to(device) if cfg.INITIAL_IMAGE else None
                    v_images = v_target_image.to(device)
                    v_features = v_input_feat.to(device)  # Now called "features"
                    bs = v_images.size(0)

                    noise = torch.randn(bs, 128, 1, 1, device=device)
                    v_fake = generator(noise, v_features, v_input_image)

                    # Reconstruction L1
                    val_l1 = l1_loss(v_fake, v_images).item()
                    val_l1_total += val_l1 * bs

                    # PSNR / SSIM (batch-averaged, weight by batch size)
                    batch_psnr = calculate_psnr(v_fake, v_images)
                    batch_ssim = calculate_ssim(v_fake, v_images)
                    val_psnr_sum += batch_psnr * bs
                    val_ssim_sum += batch_ssim * bs

                    # CLIP similarity
                    clip_sum_batch, _ = compute_clip_metrics_batch(v_fake, v_images, clip_model, clip_preprocess, device)
                    val_clip_sum += clip_sum_batch

                    # Save random sample pairs once per validation
                    if val_batch_idx == 0:
                        save_random_sample_pairs(
                            v_input_image if v_input_image is not None else v_images,
                            v_fake,
                            v_images,
                            sample_dir,
                            epoch,
                            prefix="val",
                            num_samples=4,
                        )

                    val_count += bs

            avg_val_l1 = val_l1_total / max(val_count, 1)
            avg_val_psnr = val_psnr_sum / max(val_count, 1)
            avg_val_ssim = val_ssim_sum / max(val_count, 1)
            avg_val_clip = val_clip_sum / max(val_count, 1)

            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}/{start_epoch + num_epochs}] - "
                f"Train D: {avg_d_loss:.4f}, Train G: {avg_g_loss:.4f}, "
                f"Val L1: {avg_val_l1:.4f}, Val CLIP: {avg_val_clip:.4f} "
                f"(PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}) | "
                f"Time: {elapsed:.2f}s\n"
            )

            metrics_logger.log(
                {
                    "epoch": epoch,
                    "train_d_loss": avg_d_loss,
                    "train_g_loss": avg_g_loss,
                    "val_l1": avg_val_l1,
                    "val_psnr": avg_val_psnr,
                    "val_ssim": avg_val_ssim,
                    "val_clip": avg_val_clip,
                }
            )
        else:
            # Log train-only metrics
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}/{start_epoch + num_epochs}] - "
                f"Train D: {avg_d_loss:.4f}, Train G: {avg_g_loss:.4f} "
                f"(no validation this epoch) | Time: {elapsed:.2f}s\n"
            )

            metrics_logger.log(
                {
                    "epoch": epoch,
                    "train_d_loss": avg_d_loss,
                    "train_g_loss": avg_g_loss,
                }
            )

        # Optional: save checkpoint every epoch (user can thin these later)
        ckpt_path = os.path.join(save_dir, f"{model_name}_{ver}_epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "g_optimizer_state": g_optimizer.state_dict(),
            "d_optimizer_state": d_optimizer.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")


def _setup_ddp(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _ddp_train_worker(
    rank,
    world_size,
    num_epochs,
    batch_size,
    retrain,
    checkpoint_path,
    name=None,
    version=None,
):
    _configure_torch_perf()
    _setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Optional: helps pinpoint the exact op that causes inplace/version errors.
    if getattr(cfg, "DETECT_ANOMALY", False):
        torch.autograd.set_detect_anomaly(True)

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    train_dataset = train_loader.dataset
    if hasattr(train_dataset, "input_data"):
        feature_dim = train_dataset.input_data.shape[1]
    else:
        feature_dim = len(getattr(cfg, "FEATURE_COLUMNS", []))

    # Initialize models with SD-style embeddings
    generator = Generator(
        channels=cfg.CHANNELS,
        noise_dim=cfg.NOISE_DIM,
        num_features=feature_dim,  # Number of input features (9)
        embed_dim=512,  # SD embedding dimension
        initial_image=cfg.INITIAL_IMAGE,
    ).to(device)
    discriminator = Discriminator(
        channels=cfg.CHANNELS,
        num_features=feature_dim,  # Number of input features (9)
        embed_dim=512,  # SD embedding dimension
    ).to(device)

    # NOTE: SyncBatchNorm can introduce additional collectives during forward passes.
    # In this project it has shown a tendency to hang around validation/checkpoint
    # boundaries, so we keep standard BatchNorm under DDP.

    criterion = nn.BCEWithLogitsLoss().to(device)  # Changed for AMP compatibility
    l2_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.FIXED_G_LR, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.FIXED_D_LR, betas=(0.5, 0.999))

    # Setup AMP (Automatic Mixed Precision)
    use_amp = device.type == "cuda"
    g_scaler = GradScaler("cuda") if use_amp else None
    d_scaler = GradScaler("cuda") if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else nullcontext()

    start_epoch = 0
    if retrain and checkpoint_path:
        if os.path.exists(checkpoint_path):
            if rank == 0:
                print(f"[DDP] Loading checkpoint from {checkpoint_path}")
            start_epoch, _ = load_checkpoint(
                checkpoint_path,
                generator,
                discriminator,
                optimizer_g=g_optimizer,
                optimizer_d=d_optimizer,
            )
        elif rank == 0:
            print(f"[DDP] Checkpoint path {checkpoint_path} not found, starting from scratch")

    generator = DDP(generator, device_ids=[rank], find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=False)

    save_dir = "checkpoints"
    log_dir = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    model_name = name if name else "model"
    ver = version if version else "ddp"

    metrics_logger = None
    clip_model = None
    clip_preprocess = None
    
    if rank == 0:
        metrics_logger = MetricsLogger(log_dir, f"{model_name}_{ver}_training_log.csv")
        clip_model, clip_preprocess = load_clip_model(device)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        print(f"[GAN] Rank {rank} starting epoch {epoch}")
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        generator.train()
        discriminator.train()

        for input_image, input_feat, target_image, wrong_image in train_loader:
            batch_time = time.time()

            # Move data to device
            input_image = input_image.to(device, non_blocking=True) if cfg.INITIAL_IMAGE else None
            images = target_image.to(device, non_blocking=True)
            wrong_images = wrong_image.to(device, non_blocking=True)
            features = input_feat.to(device, non_blocking=True)  # Now called "features"
            batch_size_local = images.size(0)

            # --- Train Discriminator ---
            d_optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                noise = torch.randn(batch_size_local, cfg.NOISE_DIM, 1, 1, device=device)
                fake_images = generator(noise, features, input_image)

                real_out, _ = discriminator(images, features)
                d_loss_real = criterion(real_out, torch.full_like(real_out, 1.0, device=device))

                wrong_out, _ = discriminator(wrong_images, features)
                d_loss_wrong = criterion(wrong_out, torch.full_like(wrong_out, 0.0, device=device))

                fake_out, _ = discriminator(fake_images.detach(), features)
                d_loss_fake = criterion(fake_out, torch.full_like(fake_out, 0.0, device=device))

                d_loss = d_loss_real + d_loss_wrong + d_loss_fake
            
            if d_scaler is not None:
                d_scaler.scale(d_loss).backward()
                d_scaler.step(d_optimizer)
                d_scaler.update()
            else:
                d_loss.backward()
                d_optimizer.step()
            
            # Clear discriminator gradients and intermediate tensors
            del fake_images, real_out, wrong_out, fake_out
            del d_loss_real, d_loss_wrong, d_loss_fake

            # --- Train Generator ---
            g_optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                noise = torch.randn(batch_size_local, cfg.NOISE_DIM, 1, 1, device=device)
                fake_images = generator(noise, features, input_image)

                out_fake, act_fake = discriminator(fake_images, features)
                out_real, act_real = discriminator(images, features)

                g_bce = cfg.BCE_FACTOR * criterion(out_fake, torch.full_like(out_fake, 1.0, device=device))
                g_l1 = cfg.L1_FACTOR * l1_loss(fake_images, images)
                g_l2 = cfg.L2_FACTOR * l2_loss(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())

                g_loss = g_bce + g_l1 + g_l2
            
            if g_scaler is not None:
                g_scaler.scale(g_loss).backward()
                g_scaler.step(g_optimizer)
                g_scaler.update()
            else:
                g_loss.backward()
                g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1
            
            # Explicit memory cleanup after each batch
            del fake_images, out_fake, act_fake, out_real, act_real
            del g_bce, g_l1, g_l2, g_loss, d_loss

        # End of batch loop
        print(f"[GAN] Rank {rank} Epoch {epoch} - Completed all {len(train_loader)} batches in {time.time() - start_time:.2f}s")
        
        # All-reduce train losses across ranks
        loss_tensor = torch.tensor([epoch_d_loss, epoch_g_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_d_loss, total_g_loss, total_batches = loss_tensor.tolist()
        avg_d_loss = total_d_loss / max(total_batches, 1)
        avg_g_loss = total_g_loss / max(total_batches, 1)

        print(f"[GAN] Rank {rank} finished all_reduce for epoch {epoch}")
        
        # Keep all ranks in sync before validation
        if dist.is_initialized():
            print(f"[GAN] Rank {rank} entering barrier before validation")
            dist.barrier()
            print(f"[GAN] Rank {rank} passed barrier")

        val_metrics = None
        should_validate = (
            rank == 0
            and val_loader is not None
            and (cfg.VAL_EPOCH <= 1 or epoch % cfg.VAL_EPOCH == 0)
        )
        
        if should_validate:
            print(f"[GAN] Rank {rank} starting validation")
            # --- Validation metrics (Rank 0 only) ---
            try:
                generator.eval()
                discriminator.eval()

                val_l1_total = 0.0
                val_psnr_sum = 0.0
                val_ssim_sum = 0.0
                val_clip_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for val_batch_idx, (v_input_image, v_input_feat, v_target_image, _) in enumerate(val_loader):
                        # Move data to device
                        v_input_image = v_input_image.to(device) if cfg.INITIAL_IMAGE else None
                        v_images = v_target_image.to(device)
                        v_features = v_input_feat.to(device)
                        bs = v_images.size(0)

                        noise = torch.randn(bs, cfg.NOISE_DIM, 1, 1, device=device)
                        v_fake = generator(noise, v_features, v_input_image)

                        val_l1 = l1_loss(v_fake, v_images).item()
                        val_l1_total += val_l1 * bs

                        batch_psnr = calculate_psnr(v_fake, v_images)
                        batch_ssim = calculate_ssim(v_fake, v_images)
                        val_psnr_sum += batch_psnr * bs
                        val_ssim_sum += batch_ssim * bs

                        clip_sum_batch, _ = compute_clip_metrics_batch(v_fake, v_images, clip_model, clip_preprocess, device)
                        val_clip_sum += clip_sum_batch

                        if val_batch_idx == 0:
                            save_random_sample_pairs(
                                v_input_image,
                                v_fake,
                                v_images,
                                sample_dir,
                                epoch,
                                prefix="val_ddp",
                                num_samples=4,
                            )

                        val_count += bs
                        
                        # Clear memory after each validation batch
                        del v_fake, noise, v_images, v_features
                        if v_input_image is not None:
                            del v_input_image
                
                avg_val_l1 = val_l1_total / max(val_count, 1)
                avg_val_psnr = val_psnr_sum / max(val_count, 1)
                avg_val_ssim = val_ssim_sum / max(val_count, 1)
                avg_val_clip = val_clip_sum / max(val_count, 1)
                
                val_metrics = {
                    "val_l1": avg_val_l1,
                    "val_psnr": avg_val_psnr,
                    "val_ssim": avg_val_ssim,
                    "val_clip": avg_val_clip,
                }
                
            except Exception as e:
                print(f"[GAN] Validation failed at epoch {epoch}, batch {val_batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                val_metrics = None
            
            # Set models back to train mode
            generator.train()
            discriminator.train()
            
            # Force cleanup after validation completes - only on rank 0 which did validation
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"[GAN] Rank {rank} completed validation, memory cleared")
        
        # Keep all ranks in sync after validation
        if dist.is_initialized():
            print(f"[GAN] Rank {rank} entering barrier after validation")
            dist.barrier()
            print(f"[GAN] Rank {rank} passed barrier after validation")

        if rank == 0:
            elapsed = time.time() - start_time
            if val_metrics:
                print(
                    f"[GAN] Epoch [{epoch}/{start_epoch + num_epochs}] - "
                    f"Train D: {avg_d_loss:.4f}, Train G: {avg_g_loss:.4f}, "
                    f"Val L1: {val_metrics['val_l1']:.4f}, Val CLIP: {val_metrics['val_clip']:.4f} "
                    f"(PSNR: {val_metrics['val_psnr']:.2f}, SSIM: {val_metrics['val_ssim']:.4f}) | "
                    f"Time: {elapsed:.2f}s"
                )

                metrics_logger.log(
                    {
                        "epoch": epoch,
                        "train_d_loss": avg_d_loss,
                        "train_g_loss": avg_g_loss,
                        "val_l1": val_metrics['val_l1'],
                        "val_psnr": val_metrics['val_psnr'],
                        "val_ssim": val_metrics['val_ssim'],
                        "val_clip": val_metrics['val_clip'],
                    }
                )
            else:
                print(f"[GAN] Epoch [{epoch}/{start_epoch + num_epochs}] - "
                      f"Train D: {avg_d_loss:.4f}, Train G: {avg_g_loss:.4f} | "
                      f"Time: {elapsed:.2f}s")
                metrics_logger.log(
                    {
                        "epoch": epoch,
                        "train_d_loss": avg_d_loss,
                        "train_g_loss": avg_g_loss,
                    }
                )
        
        # --- Checkpoint sync (broadcast decision + bracket slow I/O) ---
        # IMPORTANT: all ranks must execute the same sequence of collectives.
        # Rank 0 decides whether a checkpoint will be written this epoch.
        if dist.is_initialized():
            do_save_tensor = torch.tensor(
                [1 if (rank == 0 and val_metrics is not None) else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(do_save_tensor, src=0)
            do_save = bool(do_save_tensor.item())
            print(f"[GAN] Rank {rank} checkpoint decision do_save={do_save}")

            # Sync point before any potential slow disk I/O
            print(f"[GAN] Rank {rank} entering barrier before checkpoint")
            dist.barrier()
            print(f"[GAN] Rank {rank} passed barrier before checkpoint")

            if do_save and rank == 0:
                print(f"[GAN] Rank {rank} saving checkpoint...")
                val_ckpt = os.path.join(save_dir, f"{model_name}_{ver}_epoch_{epoch}.pth")
                torch.save(
                    {
                        "generator_state_dict": generator.module.state_dict(),
                        "discriminator_state_dict": discriminator.module.state_dict(),
                        "g_optimizer_state_dict": g_optimizer.state_dict(),
                        "d_optimizer_state_dict": d_optimizer.state_dict(),
                        "epoch": epoch,
                        "val_l1": val_metrics['val_l1'],
                        "val_psnr": val_metrics['val_psnr'],
                        "val_ssim": val_metrics['val_ssim'],
                        "val_clip": val_metrics['val_clip'],
                    },
                    val_ckpt,
                )
                # Ensure CUDA work is complete before releasing other rank
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[GAN] Checkpoint saved to {val_ckpt}")

            # Always run the same collective sequence, even when skipping.
            print(f"[GAN] Rank {rank} entering barrier after checkpoint")
            dist.barrier()
            print(f"[GAN] Rank {rank} passed barrier after checkpoint")

            if not do_save:
                print(f"[GAN] Rank {rank} skipping checkpoint this epoch")

    # End of epoch loop - no barrier needed, DDP handles synchronization
    
    if rank == 0:
        final_ckpt = os.path.join("checkpoints", f"{model_name}_{ver}_final_ddp.pth")
        torch.save(
            {
                "generator_state_dict": generator.module.state_dict(),
                "discriminator_state_dict": discriminator.module.state_dict(),
                "epoch": epoch,
            },
            final_ckpt,
        )
        print(f"[DDP] Final model saved to {final_ckpt}")

    _cleanup_ddp()


def train_distributed(
    num_epochs,
    retrain,
    checkpoint_path=None,
    name=None,
    version=None,
):

    device_ids = getattr(cfg, "DEVICE_IDS", list(range(torch.cuda.device_count())))
    world_size = len(device_ids)

    if world_size == 0:
        raise RuntimeError("No devices configured in DEVICE_IDS for distributed training")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested world_size={world_size} from config, but only {torch.cuda.device_count()} CUDA devices available"
        )

    batch_size = cfg.BATCH_SIZE_PER_GPU

    mp.spawn(
        _ddp_train_worker,
        args=(world_size, num_epochs, batch_size, retrain, checkpoint_path, name, version),
        nprocs=world_size,
        join=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal_basic GAN")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    epochs = args.epochs
    retrain_flag = bool(args.retrain)
    checkpoint_path = args.checkpoint or None

    if retrain_flag and not checkpoint_path:
        parser.error("--checkpoint is required when --retrain=1")

    device_ids = getattr(cfg, "DEVICE_IDS", [0])

    # Multi-GPU DDP path
    if len(device_ids) > 1 and torch.cuda.is_available():
        print(
            f"Launching DDP training on devices {device_ids} for {epochs} epochs "
            f"(retrain={retrain_flag}, checkpoint={checkpoint_path})"
        )
        train_distributed(
            num_epochs=epochs,
            retrain=retrain_flag,
            checkpoint_path=checkpoint_path,
            name="multimodal_basic",
            version="ddp",
        )
        return

    # Single-GPU / CPU path
    device_index = device_ids[0] if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{device_index}" if device_index is not None else "cpu")

    # Build dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=False,
    )

    # Dataset / feature info for the user
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    feature_dim = train_dataset.input_data.shape[1]

    print("=" * 80)
    print("Dataset summary")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Batch size: {cfg.BATCH_SIZE_PER_GPU}")
    print(f"Initial image size: {cfg.IMG_WIDTH}x{cfg.IMG_HEIGHT}")
    print(f"Target image size: {cfg.TARGET_WIDTH}x{cfg.TARGET_HEIGHT}")
    print(f"Device: {device}")
    print(f"Using initial image: {cfg.INITIAL_IMAGE}")
    print(f"SD-style embeddings: feature_dim={feature_dim}, embed_dim=512")
    print("=" * 80)

    # Initialize models with SD-style embeddings
    generator = Generator(
        channels=cfg.CHANNELS,
        noise_dim=cfg.NOISE_DIM,
        num_features=feature_dim,  # Number of input features (9)
        embed_dim=512,  # SD embedding dimension
        initial_image=cfg.INITIAL_IMAGE,
    ).to(device)
    discriminator = Discriminator(
        channels=cfg.CHANNELS,
        num_features=feature_dim,  # Number of input features (9)
        embed_dim=512,  # SD embedding dimension
    ).to(device)

    print(
        f"Starting single-GPU training for {epochs} epochs on {device} "
        f"(retrain={retrain_flag}, checkpoint={checkpoint_path})"
    )

    train(
        generator,
        discriminator,
        train_loader,
        val_loader,
        num_epochs=epochs,
        device=device,
        reload=retrain_flag,
        checkpoint_path=checkpoint_path,
        name="multimodal_basic",
        version="single"
    )


if __name__ == "__main__":
    main()