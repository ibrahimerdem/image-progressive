import argparse
import time
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

    criterion = nn.BCELoss().to(device)
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

            # If the generator uses an image embedding, pass the image; otherwise None
            if generator.module.image_embedding if isinstance(generator, nn.DataParallel) else generator.image_embedding is not None:
                input_image = input_image.to(device)
            else:
                input_image = None

            images = target_image.to(device)
            wrong_images = wrong_image.to(device)
            embeddings = input_feat.to(device)
            batch_size = images.size(0)

            # --- Train Discriminator ---
            d_optimizer.zero_grad()

            noise = torch.randn(batch_size, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, embeddings, input_image)

            real_out, real_act = discriminator(images, embeddings)
            d_loss_real = criterion(real_out, torch.full_like(real_out, 1.0, device=device))

            wrong_out, _ = discriminator(wrong_images, embeddings)
            d_loss_wrong = criterion(wrong_out, torch.full_like(wrong_out, 0.0, device=device))

            fake_out, _ = discriminator(fake_images.detach(), embeddings)
            d_loss_fake = criterion(fake_out, torch.full_like(fake_out, 0.0, device=device))

            d_loss = d_loss_real + d_loss_wrong + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # --- Train Generator ---
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, embeddings, input_image)

            out_fake, act_fake = discriminator(fake_images, embeddings)
            out_real, act_real = discriminator(images, embeddings)

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
                    if generator.module.image_embedding if isinstance(generator, nn.DataParallel) else generator.image_embedding is not None:
                        v_input_image = v_input_image.to(device)
                    else:
                        v_input_image = None

                    v_images = v_target_image.to(device)
                    v_embeddings = v_input_feat.to(device)
                    bs = v_images.size(0)

                    noise = torch.randn(bs, 128, 1, 1, device=device)
                    v_fake = generator(noise, v_embeddings, v_input_image)

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
    # Add timeout to prevent infinite hangs
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=torch.distributed.timedelta(seconds=1800))
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
    _setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Reduce workers for DDP to prevent deadlocks (max 2 per GPU)
    num_workers = min(cfg.NUM_WORKERS, 2)
    
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
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

    generator = Generator(
        channels=cfg.CHANNELS,
        noise_dim=cfg.NOISE_DIM,
        embed_dim=cfg.EMBEDDING_OUT_DIM,
        num_features=feature_dim,
        initial_image=cfg.INITIAL_IMAGE,
    ).to(device)
    
    discriminator = Discriminator(
        channels=cfg.CHANNELS,
        embed_dim=cfg.EMBEDDING_OUT_DIM,
        num_features=feature_dim,
    ).to(device)

    # Convert BatchNorm to SyncBatchNorm for DDP
    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    criterion = nn.BCELoss().to(device)
    l2_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.FIXED_G_LR, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.FIXED_D_LR, betas=(0.5, 0.999))

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

    # Remove find_unused_parameters to prevent DDP deadlocks
    generator = DDP(generator, device_ids=[rank], find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[rank])

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
    if rank == 0:
        metrics_logger = MetricsLogger(log_dir, f"{model_name}_{ver}_training_log.csv")

    clip_model, clip_preprocess = load_clip_model(device)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
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

            # Non-blocking transfers to prevent synchronization issues
            input_image = input_image.to(device, non_blocking=True)
            images = target_image.to(device, non_blocking=True)
            wrong_images = wrong_image.to(device, non_blocking=True)
            embeddings = input_feat.to(device, non_blocking=True)
            batch_size_local = images.size(0)

            # --- Train Discriminator ---
            d_optimizer.zero_grad()

            noise = torch.randn(batch_size_local, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, embeddings, input_image)

            real_out, _ = discriminator(images, embeddings)
            d_loss_real = criterion(real_out, torch.full_like(real_out, 1.0, device=device))

            wrong_out, _ = discriminator(wrong_images, embeddings)
            d_loss_wrong = criterion(wrong_out, torch.full_like(wrong_out, 0.0, device=device))

            fake_out, _ = discriminator(fake_images.detach(), embeddings)
            d_loss_fake = criterion(fake_out, torch.full_like(fake_out, 0.0, device=device))

            d_loss = d_loss_real + d_loss_wrong + d_loss_fake
            d_loss.backward()
            # Add gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            d_optimizer.step()

            # --- Train Generator ---
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size_local, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, embeddings, input_image)

            out_fake, act_fake = discriminator(fake_images, embeddings)
            out_real, act_real = discriminator(images, embeddings)

            g_bce = cfg.BCE_FACTOR * criterion(out_fake, torch.full_like(out_fake, 1.0, device=device))
            g_l1 = cfg.L1_FACTOR * l1_loss(fake_images, images)
            g_l2 = cfg.L2_FACTOR * l2_loss(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())

            g_loss = g_bce + g_l1 + g_l2
            g_loss.backward()
            # Add gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

        # Synchronize all processes before aggregating metrics
        dist.barrier()
        
        # All-reduce train losses across ranks
        loss_tensor = torch.tensor([epoch_d_loss, epoch_g_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_d_loss, total_g_loss, total_batches = loss_tensor.tolist()
        avg_d_loss = total_d_loss / max(total_batches, 1)
        avg_g_loss = total_g_loss / max(total_batches, 1)

        if (val_loader is not None) and (cfg.VAL_EPOCH <= 1 or epoch % cfg.VAL_EPOCH == 0):
            # --- Validation metrics ---
            generator.eval()
            discriminator.eval()

            val_l1_total = 0.0
            val_psnr_sum = 0.0
            val_ssim_sum = 0.0
            val_clip_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for val_batch_idx, (v_input_image, v_input_feat, v_target_image, _) in enumerate(val_loader):
                    # Non-blocking transfers for validation
                    v_input_image = v_input_image.to(device, non_blocking=True)
                    v_images = v_target_image.to(device, non_blocking=True)
                    v_embeddings = v_input_feat.to(device, non_blocking=True)
                    bs = v_images.size(0)

                    noise = torch.randn(bs, cfg.NOISE_DIM, 1, 1, device=device)
                    v_fake = generator(noise, v_embeddings, v_input_image)

                    val_l1 = l1_loss(v_fake, v_images).item()
                    val_l1_total += val_l1 * bs

                    batch_psnr = calculate_psnr(v_fake, v_images)
                    batch_ssim = calculate_ssim(v_fake, v_images)
                    val_psnr_sum += batch_psnr * bs
                    val_ssim_sum += batch_ssim * bs

                    clip_sum_batch, _ = compute_clip_metrics_batch(v_fake, v_images, clip_model, clip_preprocess, device)
                    val_clip_sum += clip_sum_batch

                    if val_batch_idx == 0 and rank == 0:
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

            # Synchronize all processes before aggregating validation metrics
            dist.barrier()
            
            val_tensor = torch.tensor(
                [val_l1_total, val_psnr_sum, val_ssim_sum, val_clip_sum, val_count],
                device=device,
            )
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            (
                total_l1,
                total_psnr,
                total_ssim,
                total_clip,
                total_count,
            ) = val_tensor.tolist()

            avg_val_l1 = total_l1 / max(total_count, 1)
            avg_val_psnr = total_psnr / max(total_count, 1)
            avg_val_ssim = total_ssim / max(total_count, 1)
            avg_val_clip = total_clip / max(total_count, 1)

            if rank == 0:
                elapsed = time.time() - start_time
                print(
                    f"[DDP] Epoch [{epoch}/{num_epochs}] - "
                    f"Train D: {avg_d_loss:.4f}, Train G: {avg_g_loss:.4f}, "
                    f"Val L1: {avg_val_l1:.4f}, Val CLIP: {avg_val_clip:.4f} "
                    f"(PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}) | "
                    f"Time: {elapsed:.2f}s"
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

                # Save checkpoint after validation
                val_ckpt = os.path.join(save_dir, f"{model_name}_{ver}_epoch_{epoch}.pth")
                torch.save(
                    {
                        "generator_state_dict": generator.module.state_dict(),
                        "discriminator_state_dict": discriminator.module.state_dict(),
                        "g_optimizer_state_dict": g_optimizer.state_dict(),
                        "d_optimizer_state_dict": d_optimizer.state_dict(),
                        "epoch": epoch,
                        "val_l1": avg_val_l1,
                        "val_psnr": avg_val_psnr,
                        "val_ssim": avg_val_ssim,
                        "val_clip": avg_val_clip,
                    },
                    val_ckpt,
                )
                print(f"[DDP] Checkpoint saved to {val_ckpt}")
        elif rank == 0:
            elapsed = time.time() - start_time
            print(
                f"[DDP] Epoch [{epoch}/{num_epochs}] - "
                f"Train D: {avg_d_loss:.4f}, Train G: {avg_g_loss:.4f} "
                f"(no validation this epoch) | Time: {elapsed:.2f}s"
            )

            metrics_logger.log(
                {
                    "epoch": epoch,
                    "train_d_loss": avg_d_loss,
                    "train_g_loss": avg_g_loss,
                }
            )

    if rank == 0:
        final_ckpt = os.path.join("checkpoints", f"{model_name}_{ver}_final_ddp.pth")
        torch.save(
            {
                "generator_state_dict": generator.module.state_dict(),
                "discriminator_state_dict": discriminator.module.state_dict(),
                "epoch": num_epochs - 1,
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
    print("=" * 80)

    generator = Generator(
        channels=cfg.CHANNELS,
        noise_dim=cfg.NOISE_DIM,
        embed_dim=cfg.EMBEDDING_OUT_DIM,  # Use EMBEDDING_OUT_DIM as embed_dim
        num_features=feature_dim,
        initial_image=cfg.INITIAL_IMAGE,
    ).to(device)
    
    discriminator = Discriminator(
        channels=cfg.CHANNELS,
        embed_dim=cfg.EMBEDDING_OUT_DIM,  # Use EMBEDDING_OUT_DIM as embed_dim
        num_features=feature_dim,
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