import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image

import config as cfg


class CustomDataset(Dataset):

    def __init__(self, split: str = "train"):
        assert split in {"train", "val", "test"}
        self.split = split

        self.img_width = cfg.IMG_WIDTH
        self.img_height = cfg.IMG_HEIGHT
        self.imgh_width = cfg.TARGET_WIDTH
        self.imgh_height = cfg.TARGET_HEIGHT
        self.channels = cfg.CHANNELS

        self.initial_dir = cfg.INITIAL_DIR
        self.target_dir = cfg.TARGET_DIR

        if split == "train":
            self.csv_path = cfg.TRAIN_CSV
        elif split == "val":
            self.csv_path = cfg.VAL_CSV
        else:
            self.csv_path = cfg.TEST_CSV

        self.input_data, self.initial_paths, self.target_paths, self.recipes = self._load_data()

        self.transform_initial = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels),
        ])

        self.transform_target = transforms.Compose([
            transforms.Resize((self.imgh_height, self.imgh_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels),
        ])

    def _load_data(self):
        df = pd.read_csv(self.csv_path)

        recipes = None
        if "recipe" in df.columns:
            recipes = df["recipe"].astype(np.int32).values

        type_indices = df["type"].astype(np.int32).values - 1
        replication_indices = df["replication"].astype(np.int32).values - 1 

        input_data = np.stack([type_indices, replication_indices], axis=1).astype(np.int64)

        initial_paths = df["initial_filename"].values
        target_paths = df["target_filename"].values

        return input_data, initial_paths, target_paths, recipes

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        type_idx = self.input_data[idx, 0]
        rep_idx = self.input_data[idx, 1]
        
        type_onehot = torch.zeros(cfg.NUM_TYPES, dtype=torch.float32)
        type_onehot[type_idx] = 1.0
        
        rep_onehot = torch.zeros(cfg.NUM_REPLICATIONS, dtype=torch.float32)
        rep_onehot[rep_idx] = 1.0

        input_feat = torch.cat([type_onehot, rep_onehot], dim=0)

        initial_path = os.path.join(self.initial_dir, self.initial_paths[idx])
        target_path = os.path.join(self.target_dir, self.target_paths[idx])

        initial_img = Image.open(initial_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        initial_img = self.transform_initial(initial_img)
        target_img = self.transform_target(target_img)

        if self.split in {"train", "val"}:
            num_samples = len(self.target_paths)

            if self.recipes is not None:
                current_recipe = self.recipes[idx]
                diff_recipe_indices = np.where(self.recipes != current_recipe)[0]

                if len(diff_recipe_indices) == 0:
                    if num_samples > 1:
                        wrong_idx = (idx + np.random.randint(1, num_samples)) % num_samples
                    else:
                        wrong_idx = idx
                else:
                    wrong_idx = int(np.random.choice(diff_recipe_indices))
            else:
                if num_samples > 1:
                    wrong_idx = np.random.randint(0, num_samples - 1)
                    if wrong_idx >= idx:
                        wrong_idx += 1
                else:
                    wrong_idx = idx

            wrong_target_path = os.path.join(self.target_dir, self.target_paths[wrong_idx])
            wrong_img = Image.open(wrong_target_path).convert("RGB")
            wrong_img = self.transform_target(wrong_img)

            return initial_img, input_feat, target_img, wrong_img

        return initial_img, input_feat, target_img, target_img


def create_dataloaders(
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """Build train/val/test loaders from the shared data/ layout."""

    train_dataset = CustomDataset(split="train")
    val_dataset = CustomDataset(split="val")
    test_dataset = CustomDataset(split="test")

    train_sampler = None
    val_sampler = None

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
