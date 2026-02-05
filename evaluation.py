#!/usr/bin/env python
"""
Test and Inference Script for Image Progressive GAN

This script provides two main functionalities:
1. Evaluate model performance on test dataset
2. Single-shot inference for individual samples
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from models.multimodal_basic import Generator
from utils.training import (
    calculate_psnr,
    calculate_ssim,
    load_clip_model,
    compute_clip_metrics_batch,
)
from utils.dataset import create_dataloaders
import config as cfg


def load_generator_from_checkpoint(checkpoint_path, device):
    """Load generator model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine feature dimension from config
    feature_dim = len(cfg.FEATURE_COLUMNS)
    
    # Create generator with built-in ImageEmbedding
    generator = Generator(
        channels=cfg.CHANNELS,
        noise_dim=cfg.NOISE_DIM,
        embed_dim=cfg.EMBEDDING_OUT_DIM,
        num_features=feature_dim,
        initial_image=cfg.INITIAL_IMAGE,
    ).to(device)
    
    # Load state dict
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"Successfully loaded generator from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_l1' in checkpoint:
        print(f"Checkpoint validation metrics:")
        print(f"  L1: {checkpoint['val_l1']:.4f}")
        print(f"  PSNR: {checkpoint.get('val_psnr', 0):.2f}")
        print(f"  SSIM: {checkpoint.get('val_ssim', 0):.4f}")
        print(f"  CLIP: {checkpoint.get('val_clip', 0):.4f}")
    
    return generator


def evaluate_test_set(generator, device, batch_size=8, num_workers=4, save_samples=True):
    """
    Evaluate model performance on test dataset.
    
    Args:
        generator: The trained generator model
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers
        save_samples: Whether to save sample outputs
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING ON TEST DATASET")
    print("="*60)
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        distributed=False,
    )
    
    if test_loader is None:
        raise ValueError("Test dataloader is None. Check if test CSV exists.")
    
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # Load CLIP model for metrics
    clip_model, clip_preprocess = load_clip_model(device)
    
    # Initialize metrics
    l1_loss = nn.L1Loss()
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_clip = 0.0
    total_count = 0
    
    # Output directory for samples
    output_dir = "outputs"
    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
    
    generator.eval()
    
    with torch.no_grad():
        start_time = time.time()
        
        for batch_idx, (input_image, input_feat, target_image, _) in enumerate(test_loader):
            input_image = input_image.to(device)
            target_image = target_image.to(device)
            input_feat = input_feat.to(device)
            
            batch_size_local = target_image.size(0)
            
            # Generate images
            noise = torch.randn(batch_size_local, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, input_feat, input_image)
            
            # Calculate metrics
            l1 = l1_loss(fake_images, target_image).item()
            psnr = calculate_psnr(fake_images, target_image)
            ssim = calculate_ssim(fake_images, target_image)
            clip_score, _ = compute_clip_metrics_batch(
                fake_images, target_image, clip_model, clip_preprocess, device
            )
            
            total_l1 += l1 * batch_size_local
            total_psnr += psnr * batch_size_local
            total_ssim += ssim * batch_size_local
            total_clip += clip_score
            total_count += batch_size_local
            
            # Save samples from first batch
            if save_samples and batch_idx == 0:
                num_samples = min(8, batch_size_local)
                for i in range(num_samples):
                    # Denormalize images (from [-1, 1] to [0, 1])
                    input_img = (input_image[i].cpu() + 1) / 2
                    fake_img = (fake_images[i].cpu() + 1) / 2
                    target_img = (target_image[i].cpu() + 1) / 2
                    
                    # Convert to PIL and save
                    to_pil = transforms.ToPILImage()
                    
                    input_pil = to_pil(input_img)
                    fake_pil = to_pil(fake_img)
                    target_pil = to_pil(target_img)
                    
                    input_pil.save(os.path.join(output_dir, f"sample_{i}_input.png"))
                    fake_pil.save(os.path.join(output_dir, f"sample_{i}_generated.png"))
                    target_pil.save(os.path.join(output_dir, f"sample_{i}_target.png"))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")
        
        elapsed_time = time.time() - start_time
    
    # Calculate average metrics
    avg_l1 = total_l1 / total_count
    avg_psnr = total_psnr / total_count
    avg_ssim = total_ssim / total_count
    avg_clip = total_clip / total_count
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Total samples evaluated: {total_count}")
    print(f"Evaluation time: {elapsed_time:.2f}s")
    print(f"\nMetrics:")
    print(f"  L1 Loss:       {avg_l1:.4f}")
    print(f"  PSNR:          {avg_psnr:.2f} dB")
    print(f"  SSIM:          {avg_ssim:.4f}")
    print(f"  CLIP Score:    {avg_clip:.4f}")
    print("="*60)
    
    if save_samples:
        print(f"\nSample outputs saved to: {output_dir}/")
    
    # Save results to file
    results = {
        'total_samples': total_count,
        'l1_loss': avg_l1,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'clip_score': avg_clip,
        'evaluation_time': elapsed_time,
    }
    
    results_file = os.path.join(output_dir, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to: {results_file}")
    
    return results


def normalize_features(features):
    """Normalize feature values based on config."""
    if not cfg.FEATURE_NORMALIZATION:
        return features
    
    feature_mins = np.array(cfg.FEATURE_MINS)
    feature_maxs = np.array(cfg.FEATURE_MAXS)
    
    normalized = 2 * (features - feature_mins) / (feature_maxs - feature_mins) - 1
    return normalized


def single_inference(generator, device, input_image_path, features, output_path="output.png", num_samples=1):
    """
    Perform single-shot inference with given input image and features.
    
    Args:
        generator: The trained generator model
        device: Device to run inference on
        input_image_path: Path to input image
        features: Dictionary or list of feature values
        output_path: Path to save generated image
        num_samples: Number of samples to generate (with different noise)
    
    Returns:
        Generated image tensor
    """
    print("\n" + "="*60)
    print("SINGLE-SHOT INFERENCE")
    print("="*60)
    
    generator.eval()
    
    # Load and preprocess input image
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    input_image = Image.open(input_image_path).convert('RGB')
    
    # Transform input image
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Process features
    if isinstance(features, dict):
        feature_values = [features[col] for col in cfg.FEATURE_COLUMNS]
    else:
        feature_values = features
    
    feature_array = np.array(feature_values, dtype=np.float32)
    
    # Normalize features
    feature_array = normalize_features(feature_array)
    
    feature_tensor = torch.from_numpy(feature_array).unsqueeze(0).to(device)
    
    print(f"Input image: {input_image_path}")
    print(f"Input image size: {input_image.size}")
    print(f"Features: {dict(zip(cfg.FEATURE_COLUMNS, feature_values))}")
    
    # Generate images
    generated_images = []
    
    with torch.no_grad():
        for i in range(num_samples):
            noise = torch.randn(1, cfg.NOISE_DIM, 1, 1, device=device)
            fake_image = generator(noise, feature_tensor, input_tensor)
            generated_images.append(fake_image)
    
    # Save outputs
    output_dir = os.path.dirname(output_path) or "."
    output_basename = os.path.splitext(os.path.basename(output_path))[0]
    output_ext = os.path.splitext(output_path)[1] or ".png"
    
    os.makedirs(output_dir, exist_ok=True)
    
    to_pil = transforms.ToPILImage()
    
    for i, gen_image in enumerate(generated_images):
        # Denormalize (from [-1, 1] to [0, 1])
        gen_image_norm = (gen_image.squeeze(0).cpu() + 1) / 2
        gen_pil = to_pil(gen_image_norm)
        
        if num_samples > 1:
            save_path = os.path.join(output_dir, f"{output_basename}_{i+1}{output_ext}")
        else:
            save_path = output_path
        
        gen_pil.save(save_path)
        print(f"Generated image saved to: {save_path}")
    
    # Also save input for comparison
    input_save_path = os.path.join(output_dir, f"{output_basename}_input.png")
    input_image.save(input_save_path)
    print(f"Input image saved to: {input_save_path}")
    
    print("="*60)
    
    return generated_images[0] if num_samples == 1 else generated_images


def main():
    parser = argparse.ArgumentParser(description="Test and Inference for Image Progressive GAN")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'inference'],
        required=True,
        help='Mode: test (evaluate on test set) or inference (single-shot generation)'
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
        default=8,
        help='Batch size for test evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )
    
    # Inference-specific arguments
    parser.add_argument(
        '--input_image',
        type=str,
        help='Path to input image for inference'
    )
    parser.add_argument(
        '--features',
        type=str,
        help='Comma-separated feature values for inference (e.g., "20,30,50,25,0.5,-10,2,3,5000")'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.png',
        help='Output path for generated image'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='Number of samples to generate with different noise'
    )
    parser.add_argument(
        '--no_save_samples',
        action='store_true',
        help='Do not save sample outputs during test evaluation'
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
    generator = load_generator_from_checkpoint(args.checkpoint, device)
    
    # Execute based on mode
    if args.mode == 'test':
        evaluate_test_set(
            generator,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_samples=not args.no_save_samples
        )
    
    elif args.mode == 'inference':
        if not args.input_image:
            parser.error("--input_image is required for inference mode")
        if not args.features:
            parser.error("--features is required for inference mode")
        
        # Parse features
        try:
            feature_values = [float(x.strip()) for x in args.features.split(',')]
            if len(feature_values) != len(cfg.FEATURE_COLUMNS):
                raise ValueError(
                    f"Expected {len(cfg.FEATURE_COLUMNS)} features, got {len(feature_values)}"
                )
        except Exception as e:
            parser.error(f"Error parsing features: {e}")
        
        single_inference(
            generator,
            device,
            args.input_image,
            feature_values,
            args.output,
            args.num_samples
        )


if __name__ == '__main__':
    main()
