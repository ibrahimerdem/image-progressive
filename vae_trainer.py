import argparse
import os
import csv
import io
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

from torchvision.transforms.functional import to_pil_image

import config as cfg
from utils.dataset import create_dataloaders
from utils.training import calculate_psnr, calculate_ssim
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def encode_with_kl(encoder_module, x, noise):

    for module in encoder_module:
        if getattr(module, 'stride', None) == (2, 2):
            x = F.pad(x, (0, 1, 0, 1))
        x = module(x)
    
    mean, log_variance = torch.chunk(x, 2, dim=1)
    log_variance = torch.clamp(log_variance, -30, 20)
    stdev = (log_variance.exp()).sqrt()
    latent = (mean + stdev * noise) * 0.18215
    # Correct KL: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())
    return latent, kl_loss


def train_worker(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"Initializing VAE models on rank {rank}...")

    vae_encoder = VAE_Encoder().to(device)
    vae_decoder = VAE_Decoder().to(device)

    vae_encoder = DDP(vae_encoder, device_ids=[rank])
    vae_decoder = DDP(vae_decoder, device_ids=[rank])

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    optimizer = torch.optim.AdamW(
        list(vae_encoder.parameters()) + list(vae_decoder.parameters()),
        lr=cfg.VAE_LR
    )
    scaler = GradScaler('cuda')

    # Setup CSV logger on rank 0
    log_path = "checkpoints/diffusion/logs/d_vae_training_log.csv"
    if rank == 0:
        os.makedirs("checkpoints/diffusion/logs", exist_ok=True)
        log_file = open(log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "train_loss", "train_l1", "train_kl",
                              "val_tgt_kb", "val_rec_kb", "val_tgt_std", "val_rec_std",
                              "val_psnr", "val_ssim"])
        log_file.flush()

    start_epoch = 0
    if args.retrain and args.checkpoint and os.path.exists(args.checkpoint):
        if rank == 0:
            print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        vae_encoder.module.load_state_dict(checkpoint['encoder'])
        vae_decoder.module.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    if rank == 0:
        print("Starting VAE training...")

    for epoch in range(start_epoch, args.epochs):
        vae_encoder.train()
        vae_decoder.train()
        train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for batch_idx, (target_img,) in enumerate(train_loader):
            target_img = target_img.to(device)
            batch_size = target_img.shape[0]

            optimizer.zero_grad()

            with autocast('cuda'):
                noise = torch.randn(
                    batch_size, 4,
                    cfg.TARGET_HEIGHT // 8, cfg.TARGET_WIDTH // 8,
                    device=device
                )
                # Fix 1: extract mean/logvar for correct KL
                latents, kl_loss = encode_with_kl(vae_encoder.module, target_img, noise)
                reconstructed = vae_decoder(latents)

                # Fix 2: L1 instead of MSE — preserves edges, avoids blur
                l1_loss = F.l1_loss(reconstructed, target_img)

                loss = l1_loss + cfg.VAE_KL_WEIGHT * kl_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_l1 += l1_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1

            if rank == 0 and batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}  L1: {l1_loss.item():.4f}  KL: {kl_loss.item():.4f}"
                )

        if rank == 0:
            avg_loss = epoch_loss / num_batches
            avg_l1   = epoch_l1   / num_batches
            avg_kl   = epoch_kl   / num_batches
            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Loss: {avg_loss:.4f}  L1: {avg_l1:.4f}  KL: {avg_kl:.6f}"
            )

        # Validation on rank 0 only
        val_tgt_kb = val_rec_kb = val_tgt_std = val_rec_std = 0.0
        val_psnr = val_ssim = 0.0
        if rank == 0 and (epoch + 1) % cfg.VAL_EPOCH == 0:
            

            vae_encoder.eval()
            vae_decoder.eval()
            val_batches = 0
            val_num_batches = 0

            with torch.no_grad():
                for (target_img,) in val_loader:
                    target_img = target_img.to(device)
                    B = target_img.size(0)

                    noise = torch.zeros(
                        B, 4,
                        cfg.TARGET_HEIGHT // 8, cfg.TARGET_WIDTH // 8,
                        device=device
                    )
                    latents, _ = encode_with_kl(vae_encoder.module, target_img, noise)
                    reconstructed = torch.clamp(vae_decoder(latents), -1.0, 1.0)

                    tgt_01 = (target_img.cpu() + 1) / 2
                    rec_01 = (reconstructed.cpu() + 1) / 2

                    for i in range(B):
                        tgt_arr = np.array(to_pil_image(tgt_01[i].clamp(0,1)))
                        rec_arr = np.array(to_pil_image(rec_01[i].clamp(0,1)))

                        buf_t = io.BytesIO()
                        buf_r = io.BytesIO()
                        to_pil_image(tgt_01[i].clamp(0,1)).save(buf_t, format='PNG')
                        to_pil_image(rec_01[i].clamp(0,1)).save(buf_r, format='PNG')

                        val_tgt_kb  += len(buf_t.getvalue()) / 1024
                        val_rec_kb  += len(buf_r.getvalue()) / 1024
                        val_tgt_std += float(tgt_arr.std())
                        val_rec_std += float(rec_arr.std())
                        val_batches += 1

                    val_psnr += calculate_psnr(reconstructed, target_img)
                    val_ssim += calculate_ssim(reconstructed, target_img)
                    val_num_batches += 1

                    if val_batches >= 32:
                        break

            val_tgt_kb  /= val_batches
            val_rec_kb  /= val_batches
            val_tgt_std /= val_batches
            val_rec_std /= val_batches
            val_psnr    /= max(val_num_batches, 1)
            val_ssim    /= max(val_num_batches, 1)

            print(f"[Val] Epoch {epoch+1}  "
                  f"target={val_tgt_kb:.0f}KB std={val_tgt_std:.1f} | "
                  f"recon={val_rec_kb:.0f}KB std={val_rec_std:.1f} | "
                  f"PSNR={val_psnr:.2f}dB SSIM={val_ssim:.4f}")

        if rank == 0:
            log_writer.writerow([
                epoch + 1, avg_loss, avg_l1, avg_kl,
                val_tgt_kb, val_rec_kb, val_tgt_std, val_rec_std,
                round(val_psnr, 4), round(val_ssim, 6),
            ])
            log_file.flush()

        if rank == 0 and (epoch + 1) % cfg.VAL_EPOCH == 0:
            os.makedirs("checkpoints/diffusion", exist_ok=True)
            ckpt_path = f"checkpoints/diffusion/d_vae_scale_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'encoder': vae_encoder.module.state_dict(),
                'decoder': vae_decoder.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        dist.barrier()

    if rank == 0:
        log_file.close()

    cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE encoder-decoder")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    print(f"Launching VAE training on devices {cfg.DEVICE_IDS}")
    mp.spawn(train_worker, args=(cfg.WORLD_SIZE, args), nprocs=cfg.WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
