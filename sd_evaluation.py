import argparse
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from models.stable_diffusion import (
    GaussianDiffusion,
    StableDiffusionConditioned,
    StableDiffusionPipeline,
)
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder
from utils.training import (
    calculate_avg_rgb_distance,
    calculate_psnr,
    calculate_ssim,
    load_clip_model,
    compute_clip_metrics_batch,
)
from utils.dataset import create_dataloaders
import config as cfg


def load_vae(checkpoint_path, device):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    
    print(f"Loading VAE from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load encoder and decoder
    vae_encoder = VAE_Encoder().to(device)
    vae_decoder = VAE_Decoder().to(device)
    
    if 'encoder' in checkpoint and 'decoder' in checkpoint:
        vae_encoder.load_state_dict(checkpoint['encoder'])
        vae_decoder.load_state_dict(checkpoint['decoder'])
    else:
        raise ValueError("VAE checkpoint must contain 'encoder' and 'decoder' keys")
    
    vae_encoder.eval()
    vae_decoder.eval()
    
    print("VAE loaded successfully")
    return vae_encoder, vae_decoder


def load_sd_model_from_checkpoint(checkpoint_path, vae_checkpoint_path, device):
    print(f"Loading SD checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load VAE
    vae_encoder, vae_decoder = load_vae(vae_checkpoint_path, device)
    
    # Freeze VAE
    for param in vae_encoder.parameters():
        param.requires_grad = False
    for param in vae_decoder.parameters():
        param.requires_grad = False
    
    # Create diffusion schedule (using default beta values)
    schedule = GaussianDiffusion(timesteps=cfg.SD_TIMESTEPS).to(device)
    
    # Create SD model (matching sd_trainer.py API)
    sd_model = StableDiffusionConditioned(
        latent_channels=4,
        emb_dim=cfg.SD_EMB_DIM,
        base_channels=cfg.SD_BASE_CHANNELS,
        use_initial_image=cfg.INITIAL_IMAGE,
    ).to(device)
    
    # Load model weights (use EMA if available)
    if 'ema_state_dict' in checkpoint:
        print("Loading EMA weights")
        sd_model.load_state_dict(checkpoint['ema_state_dict'])
    elif 'model_state_dict' in checkpoint:
        sd_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        sd_model.load_state_dict(checkpoint)
    
    sd_model.eval()
 
    pipeline = StableDiffusionPipeline(
        model=sd_model,
        schedule=schedule,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder
    )
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Successfully loaded SD model from epoch {epoch}")
    
    return pipeline, sd_model, schedule


def evaluate_test_set(
    pipeline,
    device,
    batch_size=8,
    num_workers=2,
    save_samples=True,
    num_inference_steps=cfg.SD_SAMPLE_STEPS,
):

    print("\n" + "="*60)
    print("EVALUATING SD MODEL ON TEST DATASET")
    print("="*60)

    _, _, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        distributed=False,
    )
    
    if test_loader is None:
        raise ValueError("Test dataloader is None. Check if test CSV exists.")
    
    print(f"Test set size: {len(test_loader.dataset)} samples")
    print(f"Inference steps: {num_inference_steps}")
    
    # Load CLIP model for metrics
    clip_model, clip_preprocess = load_clip_model(device)
    
    # Initialize metrics
    l1_loss = nn.L1Loss()
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_clip = 0.0
    total_rgb_dist = 0.0
    total_count = 0
    
    # Output directories
    output_dir = "outputs/sd"
    generated_dir = os.path.join(output_dir, "generated")
    target_dir_out = os.path.join(output_dir, "target")
    input_dir_out = os.path.join(output_dir, "input")
    if save_samples:
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(target_dir_out, exist_ok=True)
        os.makedirs(input_dir_out, exist_ok=True)
    
    pipeline.model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        
        for batch_idx, (input_image, input_feat, target_image, _) in enumerate(test_loader):
            input_image = input_image.to(device)
            target_image = target_image.to(device)
            input_feat = input_feat.to(device)
            
            batch_size_local = target_image.size(0)

            generated_images = pipeline.sample(
                features=input_feat,
                steps=num_inference_steps,
                save_intermediates=False,
                initial_images=input_image,
                temperature=cfg.SD_SAMPLE_TEMPERATURE,
            )

            l1 = l1_loss(generated_images, target_image).item()
            psnr = calculate_psnr(generated_images, target_image)
            ssim = calculate_ssim(generated_images, target_image)
            clip_score, _ = compute_clip_metrics_batch(
                generated_images, target_image, clip_model, clip_preprocess, device
            )
            rgb_dist = calculate_avg_rgb_distance(generated_images, target_image)
            
            total_l1 += l1 * batch_size_local
            total_psnr += psnr * batch_size_local
            total_ssim += ssim * batch_size_local
            total_clip += clip_score
            total_rgb_dist += rgb_dist * batch_size_local
            total_count += batch_size_local
            
            if save_samples:
                to_pil = transforms.ToPILImage()
                for i in range(batch_size_local):
                    global_idx = (batch_idx * batch_size) + i

                    gen_img   = torch.clamp((generated_images[i].cpu() + 1) / 2, 0, 1)
                    tgt_img   = torch.clamp((target_image[i].cpu()     + 1) / 2, 0, 1)
                    inp_img   = torch.clamp((input_image[i].cpu()      + 1) / 2, 0, 1)

                    to_pil(gen_img).save(os.path.join(generated_dir, f"{global_idx:05d}.png"))
                    to_pil(tgt_img).save(os.path.join(target_dir_out, f"{global_idx:05d}.png"))
                    to_pil(inp_img).save(os.path.join(input_dir_out,  f"{global_idx:05d}.png"))
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")
        
        elapsed_time = time.time() - start_time

    avg_l1 = total_l1 / total_count
    avg_psnr = total_psnr / total_count
    avg_ssim = total_ssim / total_count
    avg_clip = total_clip / total_count
    avg_rgb_dist = total_rgb_dist / total_count
    
    # Print results
    print("\n" + "="*60)
    print("SD MODEL TEST SET RESULTS")
    print("="*60)
    print(f"Total samples evaluated: {total_count}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Evaluation time: {elapsed_time:.2f}s")
    print(f"Time per sample: {elapsed_time/total_count:.2f}s")
    print(f"\nMetrics:")
    print(f"  L1 Loss:       {avg_l1:.4f}")
    print(f"  PSNR:          {avg_psnr:.2f} dB")
    print(f"  SSIM:          {avg_ssim:.4f}")
    print(f"  CLIP Score:    {avg_clip:.4f}")
    print(f"  RGB Distance:  {avg_rgb_dist:.4f}")
    print("="*60)
    
    if save_samples:
        print(f"\nGenerated images saved to: {generated_dir}/")
        print(f"Target images saved to:    {target_dir_out}/")
        print(f"Input images saved to:     {input_dir_out}/")
    
    # Save results to file
    results = {
        'total_samples': total_count,
        'inference_steps': num_inference_steps,
        'l1_loss': avg_l1,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'clip_score': avg_clip,
        'rgb_distance': avg_rgb_dist,
        'evaluation_time': elapsed_time,
        'time_per_sample': elapsed_time / total_count,
    }
    
    results_file = os.path.join(output_dir, "sd_test_results.txt")
    with open(results_file, 'w') as f:
        f.write("SD MODEL TEST SET EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Evaluation for Stable Diffusion Model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to SD model checkpoint'
    )
    parser.add_argument(
        '--vae_checkpoint',
        type=str,
        default=cfg.SD_VAE_CKPT,
        help='Path to VAE checkpoint (default: from config.SD_VAE_CKPT)'
    )
    parser.add_argument(
        '--no_save_samples',
        action='store_true',
        help='Do not save sample outputs'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device_str = f"cuda:{cfg.DEVICE_IDS[0]}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    print(f"Using device: {device}")
    print(f"Batch size:   {cfg.BATCH_SIZE_PER_GPU} (from config)")
    print(f"Inference steps: {cfg.SD_SAMPLE_STEPS} (from config)")
    
    # Load model
    pipeline, sd_model, schedule = load_sd_model_from_checkpoint(
        args.checkpoint,
        args.vae_checkpoint,
        device
    )
    
    # Evaluate on test set
    evaluate_test_set(
        pipeline,
        device,
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        save_samples=not args.no_save_samples,
        num_inference_steps=cfg.SD_SAMPLE_STEPS
    )


if __name__ == '__main__':
    main()
