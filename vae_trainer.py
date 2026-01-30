import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

import config as cfg
from utils.dataset import create_dataloaders
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def train_worker(rank: int, world_size: int, args) -> None:
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Initializing VAE models on rank {rank}...")
    
    # Create VAE encoder and decoder
    vae_encoder = VAE_Encoder().to(device)
    vae_decoder = VAE_Decoder().to(device)
    
    vae_encoder = DDP(vae_encoder, device_ids=[rank])
    vae_decoder = DDP(vae_decoder, device_ids=[rank])
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(vae_encoder.parameters()) + list(vae_decoder.parameters()),
        lr=cfg.VAE_LR
    )
    scaler = GradScaler('cuda')
    
    # Load checkpoint if resuming
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
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        vae_encoder.train()
        vae_decoder.train()
        
        train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        for batch_idx, (initial_img, _, target_img, _) in enumerate(train_loader):
            # Use only target images for VAE training
            target_img = target_img.to(device)
            batch_size = target_img.shape[0]
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # Encode to latent space
                noise = torch.randn(batch_size, 4, cfg.TARGET_HEIGHT // 8, cfg.TARGET_WIDTH // 8, device=device)
                latents = vae_encoder(target_img, noise)
                
                # Decode back to image space
                reconstructed = vae_decoder(latents)
                
                # Reconstruction loss (L1 or L2)
                recon_loss = F.mse_loss(reconstructed, target_img)
                
                # KL divergence loss (regularization)
                # Approximate KL loss based on latent statistics
                kl_loss = -0.5 * torch.mean(1 + torch.log(latents.var(dim=[2, 3]) + 1e-8) - latents.mean(dim=[2, 3]).pow(2) - latents.var(dim=[2, 3]))
                
                # Total loss
                loss = recon_loss + cfg.VAE_KL_WEIGHT * kl_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f} "
                      f"(Training on {batch_size} target images only)")
        
        # Log epoch average
        if rank == 0:
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches
            avg_kl = epoch_kl_loss / num_batches
            print(f"Epoch [{epoch+1}/{args.epochs}] completed. Avg Loss: {avg_loss:.4f}, "
                  f"Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}")
        
        # Validation
        if rank == 0 and (epoch + 1) % cfg.VAL_EPOCH == 0:
            print(f"Running validation at epoch {epoch+1}...")
            vae_encoder.eval()
            vae_decoder.eval()
            
            val_loss = 0.0
            val_recon_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, (initial_img, input_feat, target_img, _) in enumerate(val_loader):
                    # Use only target images for validation
                    target_img = target_img.to(device)
                    batch_size = target_img.shape[0]
                    
                    # Encode and decode
                    noise = torch.randn(batch_size, 4, cfg.TARGET_HEIGHT // 8, cfg.TARGET_WIDTH // 8, device=device)
                    latents = vae_encoder(target_img, noise)
                    reconstructed = vae_decoder(latents)
                    
                    # Reconstruction loss
                    recon_loss = F.mse_loss(reconstructed, target_img)
                    val_loss += recon_loss.item()
                    val_recon_loss += recon_loss.item()
                    val_batches += 1
                    
                    if batch_idx >= cfg.SD_VAL_STEPS:
                        break
            
            avg_val_loss = val_loss / val_batches
            print(f"Validation Loss at epoch {epoch+1}: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % cfg.VAL_EPOCH == 0:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch+1}.pth")
            
            torch.save({
                'epoch': epoch,
                'encoder': vae_encoder.module.state_dict(),
                'decoder': vae_decoder.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"VAE checkpoint saved to {checkpoint_path}")
    
    cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE encoder-decoder")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()
    
    print(f"Launching VAE training on devices {cfg.DEVICE_IDS}")
    
    world_size = cfg.WORLD_SIZE
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
