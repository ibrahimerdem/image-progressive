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

    def __init__(self, split="train"):
        assert split in {"train", "val", "test"}
        self.split = split

        self.imgh_width = cfg.TARGET_WIDTH
        self.imgh_height = cfg.TARGET_HEIGHT
        self.channels = cfg.CHANNELS

        self.target_dir = cfg.TARGET_DIR

        if split == "train":
            self.csv_path = cfg.TRAIN_CSV
        elif split == "val":
            self.csv_path = cfg.VAL_CSV
        else:
            self.csv_path = cfg.TEST_CSV
        
        self.target_paths = self._load_data()

        self.transform_target = transforms.Compose([
            transforms.Resize((self.imgh_height, self.imgh_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels),
        ])

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        
        target_paths = []
        
        for _, row in df.iterrows():

            target_paths.append(row['target_filename'])
        
        return target_paths

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        target_path = os.path.join(self.target_dir, self.target_paths[idx])
        target_img = Image.open(target_path).convert("RGB")
        target_img = self.transform_target(target_img)
        
        return target_img


def create_dataloaders(
    batch_size,
    num_workers=4,
    pin_memory=True,
    distributed=False,
    rank=0,
    world_size=1,
):
    train_dataset = CustomDataset(split="train")
    val_dataset = CustomDataset(split="val")

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

    return train_loader, val_loader