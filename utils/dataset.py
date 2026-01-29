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

    def __init__(self, split: str = "train", use_simple_features: bool = False):
        assert split in {"train", "val", "test"}
        self.split = split
        self.use_simple_features = use_simple_features

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

        self.feature_cols = cfg.FEATURE_COLUMNS
        
        self.input_data, self.initial_paths, self.target_paths = self._load_data()

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
        
        maxs = np.array(cfg.FEATURE_MAXS, dtype=np.float32)
        mins = np.array(cfg.FEATURE_MINS, dtype=np.float32)
        
        input_data = []
        initial_paths = []
        target_paths = []
        
        for _, row in df.iterrows():
            features = row[self.feature_cols].values.astype(np.float32)
            scaled_feats = (features - mins) / (maxs - mins) * 100
            scaled_feats = np.clip(scaled_feats, 0, 100)
            
            input_data.append(scaled_feats)
            initial_paths.append(row['initial_filename'])
            target_paths.append(row['target_filename'])
        
        input_data = np.array(input_data, dtype=np.float32)
        return input_data, initial_paths, target_paths

    def __len__(self):
        return len(self.initial_paths)

    def __getitem__(self, idx):
        input_feat = torch.tensor(self.input_data[idx], dtype=torch.float32)
        
        initial_path = os.path.join(self.initial_dir, self.initial_paths[idx])
        target_path = os.path.join(self.target_dir, self.target_paths[idx])

        initial_img = Image.open(initial_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        initial_img = self.transform_initial(initial_img)
        target_img = self.transform_target(target_img)

        num_samples = len(self.initial_paths)
        wrong_idx = np.random.randint(0, num_samples - 1)
        if wrong_idx >= idx:
            wrong_idx += 1
        
        wrong_path = os.path.join(self.target_dir, self.target_paths[wrong_idx])
        wrong_img = Image.open(wrong_path).convert("RGB")
        wrong_img = self.transform_target(wrong_img)
        
        return initial_img, input_feat, target_img, wrong_img


def create_dataloaders(
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    train_dataset = CustomDataset(split="train", use_simple_features=False)
    val_dataset = CustomDataset(split="val", use_simple_features=False)
    test_dataset = CustomDataset(split="test", use_simple_features=False)

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
