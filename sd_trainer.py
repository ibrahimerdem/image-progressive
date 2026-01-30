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
from torchvision.utils import save_image

import config as cfg
from torch.amp import autocast, GradScaler

from utils.dataset import create_dataloaders
from utils.training import calculate_psnr, calculate_ssim, load_clip_model, compute_clip_metrics_batch
from models.feature_encoder import FeatureProjector
from models.diffusion import Diffusion
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder
from models.ddpm import DDPMSampler

def setup_ddp(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=torch.distributed.timedelta(minutes=cfg.SD_DDP_TIMEOUT_MINUTES))
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def get_time_embedding(timestep):
    # Shape: (320,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def train_worker(rank: int, world_size: int, args) -> None:
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Initializing models on rank {rank}...")
    
    # Create feature encoder - projects input features to embedding space
    input_dim = len(cfg.FEATURE_COLUMNS)
    output_dim = cfg.SD_EMB_DIM
    feature_encoder = FeatureProjector(input_dim, output_dim).to(device)
    feature_encoder = DDP(feature_encoder, device_ids=[rank])
    
    # Create VAE encoder for encoding target images to latent space (frozen)
    vae_encoder = VAE_Encoder().to(device)
    
    # Create VAE decoder for generating images from latents (frozen)
    vae_decoder = VAE_Decoder().to(device)
    
    # Load pretrained VAE encoder and decoder
    if cfg.SD_INITIAL_ENCODER_CKPT and os.path.exists(cfg.SD_INITIAL_ENCODER_CKPT):
        if rank == 0:
            print(f"Loading pretrained VAE encoder from {cfg.SD_INITIAL_ENCODER_CKPT}...")
        vae_checkpoint = torch.load(cfg.SD_INITIAL_ENCODER_CKPT, map_location=device)
        vae_encoder.load_state_dict(vae_checkpoint['encoder'])
        vae_decoder.load_state_dict(vae_checkpoint['decoder'])
    else:
        if rank == 0:
            print("WARNING: No pretrained VAE encoder found. Train VAE first using vae_trainer.py")
    
    # Freeze VAE encoder and decoder
    for param in vae_encoder.parameters():
        param.requires_grad = False
    for param in vae_decoder.parameters():
        param.requires_grad = False
    vae_encoder.eval()
    vae_decoder.eval()
    
    # Create diffusion model (UNet)
    diffusion = Diffusion().to(device)
    diffusion = DDP(diffusion, device_ids=[rank])
    
    # Create DDPM noise scheduler
    generator = torch.Generator(device=device)
    noise_scheduler = DDPMSampler(generator, num_training_steps=cfg.SD_TIMESTEPS)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    
    # Optimizer - only feature_encoder and diffusion (VAE is frozen)
    optimizer = torch.optim.AdamW(
        list(feature_encoder.parameters()) + list(diffusion.parameters()),
        lr=cfg.SD_LR
    )
    scaler = GradScaler('cuda')
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if args.retrain and args.checkpoint and os.path.exists(args.checkpoint):
        if rank == 0:
            print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        feature_encoder.module.load_state_dict(checkpoint['feature_encoder'])
        diffusion.module.load_state_dict(checkpoint['diffusion'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")
    
    if rank == 0:
        print("Starting training...")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        feature_encoder.train()
        diffusion.train()
        vae_encoder.eval()  # VAE encoder is frozen
        
        train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (initial_img, input_feat, target_img, _) in enumerate(train_loader):
            initial_img = initial_img.to(device)
            input_feat = input_feat.to(device)
            target_img = target_img.to(device)
            
            batch_size = target_img.shape[0]
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # 1. Encode input features to conditioning embedding
                # input_feat: [B, feature_dim] -> encoded_features: [B, SD_EMB_DIM]
                encoded_features = feature_encoder(input_feat)
                
                # 2. Reshape to sequence format for cross-attention [B, 1, D]
                if len(encoded_features.shape) == 2:
                    context = encoded_features.unsqueeze(1)
                elif len(encoded_features.shape) == 3:
                    context = encoded_features[:, :1, :]
                else:
                    context = encoded_features.reshape(batch_size, 1, -1)
                
                # 3. Encode target image to latent space using VAE encoder (frozen)
                # target_img: [B, 3, H, W] -> latents: [B, 4, H/8, W/8]
                with torch.no_grad():
                    noise = torch.randn(batch_size, 4, cfg.TARGET_HEIGHT // 8, cfg.TARGET_WIDTH // 8, device=device)
                    latents = vae_encoder(target_img, noise)
                    # Scale latents (standard SD practice)
                    latents = latents * 0.18215
                
                # 4. Sample random timesteps for each image
                timesteps = torch.randint(0, cfg.SD_TIMESTEPS, (batch_size,), device=device).long()
                
                # 5. Add noise to latents according to timesteps
                noise = torch.randn_like(latents, device=device)
                noisy_latents = noise_scheduler.add_noise(latents, timesteps)
                
                # 6. Predict the noise using diffusion model
                # Get time embeddings
                time_embeddings = torch.stack([get_time_embedding(t.item())[0] for t in timesteps]).to(device)
                
                # Forward through diffusion model
                predicted_noise = diffusion(noisy_latents, context, time_embeddings)
                
                # 7. Compute loss (simple MSE between predicted and actual noise)
                loss = F.mse_loss(predicted_noise, noise)
                
                # 8. Optional: Add x0 prediction loss for better quality
                if cfg.SD_X0_LOSS_WEIGHT > 0:
                    # Predict x0 from noise prediction
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps.cpu()].to(device)
                    alpha_t = alpha_t.view(-1, 1, 1, 1)
                    sqrt_alpha_t = torch.sqrt(alpha_t)
                    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                    
                    pred_x0 = (noisy_latents - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
                    x0_loss = F.mse_loss(pred_x0, latents)
                    loss = loss + cfg.SD_X0_LOSS_WEIGHT * x0_loss
                
                # Check for NaN/Inf in inputs and outputs
                if torch.isnan(input_feat).any() or torch.isinf(input_feat).any():
                    if rank == 0:
                        print(f"Warning: NaN/Inf in input_feat at batch {batch_idx}, skipping...")
                    optimizer.zero_grad()
                    continue
                
                if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
                    if rank == 0:
                        print(f"Warning: NaN/Inf in predicted_noise at batch {batch_idx}, skipping...")
                    optimizer.zero_grad()
                    continue
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"Warning: NaN/Inf loss detected at batch {batch_idx}, loss={loss.item()}")
                    optimizer.zero_grad()
                    continue
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(feature_encoder.parameters()) + list(diffusion.parameters()),
                cfg.SD_GRAD_CLIP
            )
            
            # Check for NaN gradients with more info
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if rank == 0:
                    print(f"Warning: NaN/Inf gradients at batch {batch_idx}, grad_norm={grad_norm.item() if not torch.isnan(grad_norm) else 'nan'}")
                    print(f"  Loss was: {loss.item()}")
                    print(f"  Input range: [{input_feat.min().item():.4f}, {input_feat.max().item():.4f}]")
                optimizer.zero_grad()
                scaler.update()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if rank == 0 and batch_idx % cfg.SD_LOG_INTERVAL == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Log epoch average
        if rank == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}")
        
        # Validation and metrics calculation
        if rank == 0 and (epoch + 1) % cfg.VAL_EPOCH == 0:
            print(f"Running validation at epoch {epoch+1}...")
            feature_encoder.eval()
            diffusion.eval()
            
            val_loss = 0.0
            val_batches = 0
            sample_batch = None
            
            # Load CLIP model for similarity metrics
            clip_model, clip_preprocess = load_clip_model(device)
            
            # Metrics accumulators
            all_psnr = []
            all_ssim = []
            clip_sum = 0.0
            clip_count = 0
            
            # Clear GPU cache before generation
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                for batch_idx, (initial_img, input_feat, target_img, _) in enumerate(val_loader):
                    initial_img = initial_img.to(device)
                    input_feat = input_feat.to(device)
                    target_img = target_img.to(device)
                    
                    batch_size = target_img.shape[0]
                    
                    # Encode features
                    encoded_features = feature_encoder(input_feat)
                    context = encoded_features.unsqueeze(1)
                    
                    # Encode target to latents for loss calculation
                    noise = torch.randn(batch_size, 4, cfg.TARGET_HEIGHT // 8, cfg.TARGET_WIDTH // 8, device=device)
                    latents = vae_encoder(target_img, noise) * 0.18215
                    
                    # Sample timesteps and add noise
                    timesteps = torch.randint(0, cfg.SD_TIMESTEPS, (batch_size,), device=device).long()
                    noise_sample = torch.randn_like(latents, device=device)
                    noisy_latents = noise_scheduler.add_noise(latents, timesteps)
                    
                    # Predict noise for validation loss
                    time_embeddings = torch.stack([get_time_embedding(t.item())[0] for t in timesteps]).to(device)
                    predicted_noise = diffusion(noisy_latents, context, time_embeddings)
                    
                    # Compute loss
                    loss = F.mse_loss(predicted_noise, noise_sample)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Generate images for metrics (full denoising)
                    gen_latents = torch.randn(batch_size, 4, cfg.TARGET_HEIGHT // 8, cfg.TARGET_WIDTH // 8, device=device)
                    sampler = DDPMSampler(generator=torch.Generator(device=device))
                    sampler.set_inference_timesteps(cfg.SD_SAMPLE_STEPS)
                    
                    for i, timestep in enumerate(sampler.timesteps):
                        timestep_tensor = torch.tensor([timestep], device=device).long().repeat(batch_size)
                        time_emb = torch.stack([get_time_embedding(t.item())[0] for t in timestep_tensor]).to(device)
                        model_output = diffusion.module(gen_latents, context, time_emb)
                        gen_latents = sampler.step(timestep, gen_latents, model_output)
                    
                    # Decode latents to images
                    gen_latents = gen_latents / 0.18215
                    generated_img = vae_decoder(gen_latents)
                    
                    # Calculate metrics (images in [-1, 1] range)
                    psnr_val = calculate_psnr(generated_img, target_img)
                    ssim_val = calculate_ssim(generated_img, target_img)
                    clip_sim, clip_batch = compute_clip_metrics_batch(generated_img, target_img, clip_model, clip_preprocess, device)
                    
                    all_psnr.append(psnr_val)
                    all_ssim.append(ssim_val)
                    clip_sum += clip_sim
                    clip_count += clip_batch
                    
                    # Save first batch for visualization
                    if batch_idx == 0:
                        sample_batch = (initial_img, generated_img, target_img)
                    
                    if batch_idx >= cfg.SD_VAL_STEPS:
                        break
                    
                    # Clear cache periodically
                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
            
            # Calculate average metrics
            avg_val_loss = val_loss / val_batches
            avg_psnr = sum(all_psnr) / len(all_psnr)
            avg_ssim = sum(all_ssim) / len(all_ssim)
            avg_clip = clip_sum / clip_count if clip_count > 0 else 0.0
            
            print(f"Validation at epoch {epoch+1}:")
            print(f"  Loss: {avg_val_loss:.4f}")
            print(f"  PSNR: {avg_psnr:.2f}")
            print(f"  SSIM: {avg_ssim:.4f}")
            print(f"  CLIP Score: {avg_clip:.4f}")
            
            # Save sample images as triplets: initial-generated-target
            if sample_batch is not None:
                initial_img, generated_img, target_img = sample_batch
                
                # Normalize to [0, 1] for saving
                initial_img = (initial_img + 1) / 2
                generated_img = (generated_img + 1) / 2
                target_img = (target_img + 1) / 2
                
                # Resize initial image to match target resolution for visualization
                initial_img = F.interpolate(initial_img, size=(cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH), mode='bilinear', align_corners=False)
                
                # Create output directory
                sample_dir = os.path.join("checkpoints", "sd", "samples", f"epoch_{epoch+1}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save as triplet grid: [initial, generated, target] for each sample
                triplet = torch.cat([initial_img, generated_img, target_img], dim=0)
                save_image(triplet, os.path.join(sample_dir, "triplets.png"), nrow=batch_size, padding=2)
                
                print(f"Sample triplets saved to {sample_dir}")
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % cfg.VAL_EPOCH == 0:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"sd_epoch_{epoch+1}.pth")
            
            torch.save({
                'epoch': epoch,
                'feature_encoder': feature_encoder.module.state_dict(),
                'diffusion': diffusion.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stable diffusion conditioned on features and an optional initial image")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    retrain_flag = bool(args.retrain)
    checkpoint_path = args.checkpoint or None
    # Note: retrain=1 means RESUME training, retrain=0 means start fresh
    # Checkpoint path is optional - if not provided, auto-detects latest

    print(f"Launching stable diffusion DDP training on devices {cfg.DEVICE_IDS}")
    
    world_size = cfg.WORLD_SIZE
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
