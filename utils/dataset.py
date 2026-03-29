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
        self.target_cols = cfg.TARGET_FEATURE_COLUMNS
        
        self.input_data, self.initial_paths, self.target_paths, self.target_data = self._load_data()

        self.transform_initial = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transform_target = transforms.Compose([
            transforms.Resize((self.imgh_height, self.imgh_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
                
        input_data = []
        initial_paths = []
        target_paths = []
        target_data = []

        self.feature_min_max = (cfg.FEATURE_MINS, cfg.FEATURE_MAXS)
        self.target_min_max = (cfg.TARGET_MINS, cfg.TARGET_MAXS)

        if self.feature_min_max:
            self.feature_mins = torch.tensor(self.feature_min_max[0], dtype=torch.float32)
            self.feature_maxs = torch.tensor(self.feature_min_max[1], dtype=torch.float32)

        if self.target_min_max:
            self.target_mins = torch.tensor(self.target_min_max[0], dtype=torch.float32)
            self.target_maxs = torch.tensor(self.target_min_max[1], dtype=torch.float32)
        
        for _, row in df.iterrows():
            features = row[self.feature_cols].values.astype(np.float32)
            scaled_feats = 2 * (features - self.feature_mins.numpy()) / (self.feature_maxs.numpy() - self.feature_mins.numpy()) - 1
            #scaled_feats = np.clip(scaled_feats, -1, 1)

            target = row[self.target_cols].values.astype(np.float32)
            scaled_target = 2 * (target - self.target_mins.numpy()) / (self.target_maxs.numpy() - self.target_mins.numpy()) - 1
            #scaled_target = np.clip(scaled_target, -1, 1)
            
            input_data.append(scaled_feats)
            initial_paths.append(row['initial_filename'])
            target_paths.append(row['target_filename'])
            target_data.append(scaled_target)
        
        input_data = np.array(input_data, dtype=np.float32)
        target_data = np.array(target_data, dtype=np.float32)
        return input_data, initial_paths, target_paths, target_data

    def __len__(self):
        return len(self.initial_paths)

    def __getitem__(self, idx):
        
        input_feat = torch.tensor(self.input_data[idx], dtype=torch.float32)
        target_feat = torch.tensor(self.target_data[idx], dtype=torch.float32)

        initial_path = os.path.join(self.initial_dir, self.initial_paths[idx])
        target_path = os.path.join(self.target_dir, self.target_paths[idx])

        initial_img = Image.open(initial_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        initial_img = self.transform_initial(initial_img)
        target_img = self.transform_target(target_img)

        return input_feat, initial_img, target_img, target_feat
    

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