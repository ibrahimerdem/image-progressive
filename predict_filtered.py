import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import config as cfg
from ld_evaluation import load_model_from_checkpoint


class FilteredDataset(Dataset):
    """Dataset built from a pre-filtered dataframe of data/all_features.csv rows."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.feature_cols = cfg.FEATURE_COLUMNS
        self.mins = np.array(cfg.FEATURE_MINS, dtype=np.float32)
        self.maxs = np.array(cfg.FEATURE_MAXS, dtype=np.float32)

        self.transform_initial = transforms.Compose([
            transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * cfg.CHANNELS, std=[0.5] * cfg.CHANNELS),
        ])
        self.transform_target = transforms.Compose([
            transforms.Resize((cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * cfg.CHANNELS, std=[0.5] * cfg.CHANNELS),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        features = row[self.feature_cols].values.astype(np.float32)
        scaled_feats = (features - self.mins) / (self.maxs - self.mins)
        scaled_feats = np.clip(scaled_feats, 0, 1)
        input_feat = torch.tensor(scaled_feats, dtype=torch.float32)

        initial_path = os.path.join(cfg.INITIAL_DIR, row['initial_filename'])
        target_path = os.path.join(cfg.TARGET_DIR, row['target_filename'])

        initial_img = self.transform_initial(Image.open(initial_path).convert("RGB"))
        target_img = self.transform_target(Image.open(target_path).convert("RGB"))

        name = f"type{int(row['type'])}_recipe{int(row['recipe'])}_{os.path.splitext(row['target_filename'])[0]}"

        return initial_img, input_feat, target_img, name


def filter_data(csv_path, types, max_recipe):
    df = pd.read_csv(csv_path)
    filtered = df[df['type'].isin(types) & (df['recipe'] <= max_recipe)]
    print(f"Filtered {len(filtered)}/{len(df)} rows (type in {types}, recipe <= {max_recipe})")
    return filtered


def predict(pipeline, device, dataset, output_dir, batch_size, num_workers, steps):
    generated_dir = os.path.join(output_dir, "generated")
    target_dir_out = os.path.join(output_dir, "target")
    input_dir_out = os.path.join(output_dir, "input")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(target_dir_out, exist_ok=True)
    os.makedirs(input_dir_out, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    to_pil = transforms.ToPILImage()
    pipeline.model.eval()
    total = 0

    with torch.no_grad():
        for batch_idx, (input_image, input_feat, target_image, names) in enumerate(loader):
            input_image = input_image.to(device)
            input_feat = input_feat.to(device)
            target_image = target_image.to(device)

            generated_images = pipeline.sample(
                features=input_feat,
                steps=steps,
                save_intermediates=False,
                initial_images=input_image,
                temperature=cfg.SAMPLE_TEMPERATURE,
                eta=cfg.SAMPLER_ETA,
            )

            for i, name in enumerate(names):
                gen_img = torch.clamp((generated_images[i].cpu() + 1) / 2, 0, 1)
                tgt_img = torch.clamp((target_image[i].cpu() + 1) / 2, 0, 1)
                inp_img = torch.clamp((input_image[i].cpu() + 1) / 2, 0, 1)

                to_pil(gen_img).save(os.path.join(generated_dir, f"{name}.png"))
                to_pil(tgt_img).save(os.path.join(target_dir_out, f"{name}.png"))
                to_pil(inp_img).save(os.path.join(input_dir_out, f"{name}.png"))

            total += len(names)
            print(f"Processed {total}/{len(dataset)} samples...")

    print(f"\nSaved {total} samples to {output_dir}/ (input/, target/, generated/)")


def main():
    parser = argparse.ArgumentParser(description="Predict on a filtered subset of all_features.csv")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to diffusion model checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default=cfg.VAE_CKPT, help='Path to VAE checkpoint')
    parser.add_argument('--csv', type=str, default=os.path.join(cfg.DATA_DIR, "all_features.csv"))
    parser.add_argument('--output_dir', type=str, default="outputs/diffusion/filtered")
    parser.add_argument('--types', type=int, nargs='+', default=[27, 34, 35, 47])
    parser.add_argument('--max_recipe', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE_PER_GPU)
    parser.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    parser.add_argument('--steps', type=int, default=cfg.SAMPLE_STEPS)
    args = parser.parse_args()

    device_str = f"cuda:{cfg.DEVICE_IDS[0]}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    df = filter_data(args.csv, args.types, args.max_recipe)
    if len(df) == 0:
        raise ValueError("No rows matched the given filter.")

    dataset = FilteredDataset(df)

    pipeline, _, _ = load_model_from_checkpoint(args.checkpoint, args.vae_checkpoint, device)

    predict(
        pipeline,
        device,
        dataset,
        args.output_dir,
        args.batch_size,
        args.num_workers,
        args.steps,
    )


if __name__ == '__main__':
    main()
