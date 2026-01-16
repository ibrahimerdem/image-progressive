import os
import csv
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import clip
from PIL import Image


def save_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch,
    loss,
    filename,
    scheduler_g=None,
    scheduler_d=None,
):
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict()
    }

    if scheduler_g is not None:
        checkpoint['scheduler_g_state_dict'] = scheduler_g.state_dict()
    if scheduler_d is not None:
        checkpoint['scheduler_d_state_dict'] = scheduler_d.state_dict()

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(
    filename,
    generator,
    discriminator,
    optimizer_g=None,
    optimizer_d=None,
    scheduler_g=None,
    scheduler_d=None,
):
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint['model_state_dict'])

    if 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    else:
        print("Warning: discriminator state not found in checkpoint; using current weights.")

    if optimizer_g is not None:
        if "optimizer_g_state_dict" in checkpoint:
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    if optimizer_d is not None:
        if 'optimizer_d_state_dict' in checkpoint:
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    if scheduler_g is not None and 'scheduler_g_state_dict' in checkpoint:
        scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    if scheduler_d is not None and 'scheduler_d_state_dict' in checkpoint:
        scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)

    print(f"Checkpoint loaded: {filename} (epoch {epoch})")
    return epoch, loss

def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Denormalize image from [-1, 1] to [0, 1]"""
    return (image + 1) / 2

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images. Detach to avoid grad issues during validation."""
    img1 = denormalize_image(img1.detach()).cpu().numpy()
    img2 = denormalize_image(img2.detach()).cpu().numpy()
    
    # Convert from [B, C, H, W] to [B, H, W, C]
    img1 = np.transpose(img1, (0, 2, 3, 1))
    img2 = np.transpose(img2, (0, 2, 3, 1))
    
    psnr_values = []
    for i in range(img1.shape[0]):
        psnr_val = psnr(img1[i], img2[i], data_range=1.0)
        psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images. Detach to avoid grad issues during validation."""
    img1 = denormalize_image(img1.detach()).cpu().numpy()
    img2 = denormalize_image(img2.detach()).cpu().numpy()
    
    # Convert from [B, C, H, W] to [B, H, W, C]
    img1 = np.transpose(img1, (0, 2, 3, 1))
    img2 = np.transpose(img2, (0, 2, 3, 1))
    
    ssim_values = []
    for i in range(img1.shape[0]):
        ssim_val = ssim(img1[i], img2[i], data_range=1.0, channel_axis=2)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def visualize_results(
    initial_images,
    generated_images,
    target_images,
    num_samples=4,
    save_path=None
):
    num_samples = min(num_samples, initial_images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Detach and denormalize images
        initial = denormalize_image(initial_images[i].detach()).cpu().permute(1, 2, 0).numpy()
        generated = denormalize_image(generated_images[i].detach()).cpu().permute(1, 2, 0).numpy()
        target = denormalize_image(target_images[i].detach()).cpu().permute(1, 2, 0).numpy()
        
        # Clip to [0, 1]
        initial = np.clip(initial, 0, 1)
        generated = np.clip(generated, 0, 1)
        target = np.clip(target, 0, 1)
        
        # Plot
        axes[i, 0]. imshow(initial)
        axes[i, 0].set_title('Initial Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(generated)
        axes[i, 1].set_title('Generated Image')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target)
        axes[i, 2]. set_title('Target Image')
        axes[i, 2]. axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self. sum += val * n
        self.count += n
        self. avg = self.sum / self. count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self. best_loss is None: 
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self. min_delta:
            self. counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self. best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MetricsLogger:

    def __init__(self, log_dir: str, filename: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.header_written = os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0

    def log(self, metrics: dict):

        if not metrics:
            return

        # Ensure deterministic column order
        keys = list(metrics.keys())
        values = [metrics[k] for k in keys]

        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not self.header_written:
                writer.writerow(keys)
                self.header_written = True
            writer.writerow(values)


def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def _clip_prepare_image(image):
    img = image.detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = np.clip((img + 1) / 2 * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def get_clip_embeddings(image, model, preprocess, device):
    img = _clip_prepare_image(image)
    image_input = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features / image_features.norm(dim=-1, keepdim=True)


def calculate_clip_similarity(fake_image, target_image, model, preprocess, device):
    fake_features = get_clip_embeddings(fake_image, model, preprocess, device)
    target_features = get_clip_embeddings(target_image, model, preprocess, device)
    similarity = (fake_features @ target_features.T).item()
    return similarity


def compute_clip_metrics_batch(fake_images, target_images, model, preprocess, device):
    batch_size = fake_images.size(0)
    sim_sum = 0.0
    for i in range(batch_size):
        sim_sum += calculate_clip_similarity(fake_images[i], target_images[i], model, preprocess, device)
    return sim_sum, batch_size


def save_random_sample_pairs(
    initial_images,
    generated_images,
    target_images,
    sample_dir,
    epoch,
    prefix="val",
    num_samples=4,
):

    os.makedirs(sample_dir, exist_ok=True)

    batch_size = initial_images.size(0)
    num = min(num_samples, batch_size)

    indices = list(range(batch_size))
    random.shuffle(indices)
    indices = indices[:num]

    initials_subset = initial_images[indices]
    generated_subset = generated_images[indices]
    target_subset = target_images[indices]

    save_path = os.path.join(sample_dir, f"{prefix}_epoch_{epoch:04d}.png")
    visualize_results(initials_subset, generated_subset, target_subset, num_samples=num, save_path=save_path)


def save_diffusion_intermediates(intermediates, sample_dir, epoch, sample_idx=0):
    """Save intermediate diffusion steps as a grid."""
    os.makedirs(sample_dir, exist_ok=True)
    
    num_steps = len(intermediates)
    if num_steps == 0:
        return
    
    # Create a grid showing denoising progress
    fig, axes = plt.subplots(1, num_steps, figsize=(3 * num_steps, 3))
    if num_steps == 1:
        axes = [axes]
    
    for idx, (step, img_tensor) in enumerate(intermediates):
        # Take first image from batch
        img = denormalize_image(img_tensor[sample_idx].detach()).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'Step {step}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(sample_dir, f"intermediates_epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__": 
    print("Utils module loaded successfully!")