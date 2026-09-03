#!/usr/bin/env python
"""
Generate images for the entire test set using a trained cross-attention
Generator checkpoint, saving each output using the dataset's target filename.
"""

import argparse
import os

import torch
from torchvision import transforms

from models.multimodal_cross import Generator
from utils.dataset import CustomDataset
import config as cfg


def load_generator(checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_dim = len(cfg.FEATURE_COLUMNS)

    generator = Generator(
        channels=cfg.CHANNELS,
        noise_dim=cfg.NOISE_DIM,
        embed_dim=cfg.EMBEDDING_OUT_DIM,
        num_features=feature_dim,
        initial_image=cfg.INITIAL_IMAGE,
    ).to(device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    print(f"Loaded generator from epoch {checkpoint.get('epoch', 'unknown')}")
    return generator


def generate_test_set(generator, device, output_dir, batch_size=8, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    target_dir = os.path.join(output_dir, "targets")
    os.makedirs(target_dir, exist_ok=True)

    test_dataset = CustomDataset(split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    to_pil = transforms.ToPILImage()

    print(f"Test set size: {len(test_dataset)} samples")
    print(f"Saving generated images to: {output_dir}/")
    print(f"Saving 256x256 target images to: {target_dir}/")

    global_idx = 0
    generator.eval()
    with torch.no_grad():
        for batch_idx, (input_image, input_feat, target_image, _) in enumerate(test_loader):
            input_image = input_image.to(device)
            input_feat = input_feat.to(device)
            bs = input_feat.size(0)

            noise = torch.randn(bs, cfg.NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise, input_feat, input_image)

            for i in range(bs):
                target_name = test_dataset.target_paths[global_idx]

                fake_img = (fake_images[i].cpu() + 1) / 2  # [-1,1] -> [0,1]
                fake_img = fake_img.clamp(0, 1)
                to_pil(fake_img).save(os.path.join(output_dir, target_name))

                real_img = (target_image[i].cpu() + 1) / 2  # [-1,1] -> [0,1], already 256x256
                real_img = real_img.clamp(0, 1)
                to_pil(real_img).save(os.path.join(target_dir, target_name))

                global_idx += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"Processed {global_idx}/{len(test_dataset)} samples...")

    print(f"Done. {global_idx} images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate test-set images from a GAN checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/multimodal_basic_cross.pth")
    parser.add_argument("--output-dir", default="outputs/gan/cross")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_generator(args.checkpoint, device)
    generate_test_set(
        generator,
        device,
        args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
