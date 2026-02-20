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
        print("✓ Loading EMA weights (recommended for best quality)")
        sd_model.load_state_dict(checkpoint['ema_state_dict'])
    elif 'model_state_dict' in checkpoint:
        print("⚠ Loading regular model weights (EMA not found)")
        sd_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠ Loading weights from unknown format")
        sd_model.load_state_dict(checkpoint)
    
    sd_model.eval()
    
    # Put model in eval mode and disable dropout/batchnorm updates
    for module in sd_model.modules():
        if hasattr(module, 'training'):
            module.training = False
 
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
    save_all=True,
    num_inference_steps=200,  # Match training validation steps for better quality
):

    print("\n" + "="*60)
    print("EVALUATING SD MODEL ON TEST DATASET")
    print("="*60)
    print(f"Sampling Configuration:")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Total timesteps: {cfg.SD_TIMESTEPS}")
    print(f"  Step ratio: {cfg.SD_TIMESTEPS / num_inference_steps:.1f}x subsampling")
    print(f"  Image size: {cfg.TARGET_WIDTH}x{cfg.TARGET_HEIGHT}")
    print(f"  Initial image conditioning: {cfg.INITIAL_IMAGE}")
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
    total_count = 0
    
    # Storage for all images if save_all is True
    all_images = {'generated': [], 'target': [], 'input': []} if save_all else None
    
    # Output directory for samples
    output_dir = "outputs/sd"
    if save_samples or save_all:
        os.makedirs(output_dir, exist_ok=True)
        if save_all:
            os.makedirs(os.path.join(output_dir, "generated"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "target"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "input"), exist_ok=True)
    
    pipeline.model.eval()
    
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        start_time = time.time()
        
        for batch_idx, (input_image, input_feat, target_image, _) in enumerate(test_loader):
            input_image = input_image.to(device)
            target_image = target_image.to(device)
            input_feat = input_feat.to(device)
            
            batch_size_local = target_image.size(0)

            # Generate images using the same settings as training validation
            generated_images = pipeline.sample(
                features=input_feat,
                steps=num_inference_steps,
                save_intermediates=False,
                initial_images=input_image
            )

            l1 = l1_loss(generated_images, target_image).item()
            psnr = calculate_psnr(generated_images, target_image)
            ssim = calculate_ssim(generated_images, target_image)
            clip_score, _ = compute_clip_metrics_batch(
                generated_images, target_image, clip_model, clip_preprocess, device
            )
            
            total_l1 += l1 * batch_size_local
            total_psnr += psnr * batch_size_local
            total_ssim += ssim * batch_size_local
            total_clip += clip_score
            total_count += batch_size_local
            
            # Save all images if requested
            if save_all:
                to_pil = transforms.ToPILImage()
                for i in range(batch_size_local):
                    # Denormalize images (from [-1, 1] to [0, 1])
                    input_img = torch.clamp((input_image[i].cpu() + 1) / 2, 0, 1)
                    generated_img = torch.clamp((generated_images[i].cpu() + 1) / 2, 0, 1)
                    target_img = torch.clamp((target_image[i].cpu() + 1) / 2, 0, 1)
                    
                    # Convert to PIL
                    all_images['input'].append(to_pil(input_img))
                    all_images['generated'].append(to_pil(generated_img))
                    all_images['target'].append(to_pil(target_img))
            
            # Save samples from first batch
            elif save_samples and batch_idx == 0:
                num_samples = min(8, batch_size_local)
                to_pil = transforms.ToPILImage()
                for i in range(num_samples):
                    # Denormalize images (from [-1, 1] to [0, 1])
                    input_img = torch.clamp((input_image[i].cpu() + 1) / 2, 0, 1)
                    generated_img = torch.clamp((generated_images[i].cpu() + 1) / 2, 0, 1)
                    target_img = torch.clamp((target_image[i].cpu() + 1) / 2, 0, 1)
                    
                    # Convert to PIL and save
                    input_pil = to_pil(input_img)
                    generated_pil = to_pil(generated_img)
                    target_pil = to_pil(target_img)
                    
                    input_pil.save(os.path.join(output_dir, f"sample_{i}_input.png"))
                    generated_pil.save(os.path.join(output_dir, f"sample_{i}_generated.png"))
                    target_pil.save(os.path.join(output_dir, f"sample_{i}_target.png"))
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")
        
        elapsed_time = time.time() - start_time
    
    # Save all images to organized folders if requested
    if save_all and all_images:
        print(f"\nSaving {len(all_images['generated'])} images to organized folders...")
        for idx, (gen_img, target_img, input_img) in enumerate(zip(
            all_images['generated'], 
            all_images['target'], 
            all_images['input']
        )):
            gen_img.save(os.path.join(output_dir, "generated", f"sample_{idx:04d}.png"))
            target_img.save(os.path.join(output_dir, "target", f"sample_{idx:04d}.png"))
            input_img.save(os.path.join(output_dir, "input", f"sample_{idx:04d}.png"))
        
        print(f"Images saved to:")
        print(f"  Generated: {os.path.join(output_dir, 'generated')}/")
        print(f"  Target:    {os.path.join(output_dir, 'target')}/")
        print(f"  Input:     {os.path.join(output_dir, 'input')}/")

    avg_l1 = total_l1 / total_count
    avg_psnr = total_psnr / total_count
    avg_ssim = total_ssim / total_count
    avg_clip = total_clip / total_count
    
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
    print("="*60)
    
    if save_samples and not save_all:
        print(f"\nSample outputs saved to: {output_dir}/")
    
    # Save results to file
    results = {
        'total_samples': total_count,
        'inference_steps': num_inference_steps,
        'l1_loss': avg_l1,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'clip_score': avg_clip,
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
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (e.g., cuda:0, cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for test evaluation (smaller due to diffusion steps)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--inference_steps',
        type=int,
        default=200,  # Match SD_SAMPLE_STEPS from config for quality
        help='Number of diffusion steps for sampling (default: 200, matching training validation)'
    )
    parser.add_argument(
        '--no_save_samples',
        action='store_true',
        help='Do not save sample outputs'
    )
    parser.add_argument(
        '--no_save_all',
        action='store_true',
        help='Do not save all images to organized folders (only save first 8 samples instead)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_samples=not args.no_save_samples,
        save_all=not args.no_save_all,
        num_inference_steps=args.inference_steps
    )


if __name__ == '__main__':
    main()
