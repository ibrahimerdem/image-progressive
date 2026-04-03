import numpy as np
import torch
import torch.nn as nn

import config as cfg


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Shortcut: match dims if channels or spatial size changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResNet19(nn.Module):

    def __init__(self, in_channels=3):
        super(ResNet19, self).__init__()

        # Stem: 1 conv layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(64,  64,  num_blocks=1, stride=1)  
        self.stage2 = self._make_stage(64,  128, num_blocks=1, stride=2)  
        self.stage3 = self._make_stage(128, 256, num_blocks=1, stride=2)  
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)  

        self.gap = nn.AdaptiveAvgPool2d(1)  # → (B, 512, 1, 1)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        blocks = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return torch.flatten(x, 1)  # (B, 512)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=64):
        super(TabularEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, out_dim)


class Basic_Triplet(nn.Module):
    def __init__(self):
        super(Basic_Triplet, self).__init__()

        self.input_dim    = len(cfg.FEATURE_COLUMNS)
        self.img_channels = cfg.CHANNELS
        self.output_dim   = 3

        # Image encoder: ResNet-19 from scratch
        self.image_encoder = ResNet19(in_channels=self.img_channels)
        # Output: (B, 512)

        # Tabular encoder
        self.tab_encoder = TabularEncoder(
            input_dim=self.input_dim,
            hidden_dim=128,
            out_dim=64,
        )
        # Output: (B, 64)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.output_dim),   # (B, 3)
        )

    def forward(self, condition, image):
        # 1. Extract image features via ResNet-19
        img_feat = self.image_encoder(image)       # (B, 512)

        # 2. Encode tabular features
        tab_feat = self.tab_encoder(condition)     # (B, 64)

        # 3. Fuse and regress
        fused = torch.cat([img_feat, tab_feat], dim=1)  # (B, 576)
        out   = self.fusion_mlp(fused)                  # (B, 3)

        return out