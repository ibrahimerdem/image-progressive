import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

import config as cfg


# Feature-vector size for supported encoder names
_ENCODER_OUT_DIM = {
    "resnet18":  512,
    "resnet34":  512,
    "resnet50":  2048,
    "resnet101": 2048,
}


def _build_image_encoder(name: str, freeze: bool) -> tuple[nn.Module, int]:
    """Return (backbone_without_fc, out_dim)."""
    weights_map = {
        "resnet18":  tv_models.ResNet18_Weights.DEFAULT,
        "resnet34":  tv_models.ResNet34_Weights.DEFAULT,
        "resnet50":  tv_models.ResNet50_Weights.DEFAULT,
        "resnet101": tv_models.ResNet101_Weights.DEFAULT,
    }
    if name not in weights_map:
        raise ValueError(f"Unsupported IMAGE_ENCODER '{name}'. Choose from: {list(weights_map)}")

    backbone = getattr(tv_models, name)(weights=weights_map[name])
    out_dim = _ENCODER_OUT_DIM[name]

    # Remove the classification head; keep avgpool so output is (B, out_dim)
    backbone.fc = nn.Identity()

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, out_dim


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
        self.output_dim   = 3

        # Pretrained image encoder (backbone only, FC removed)
        freeze = getattr(cfg, "FREEZE_BACKBONE", True)
        self.image_encoder, img_feat_dim = _build_image_encoder(cfg.IMAGE_ENCODER, freeze=freeze)

        # Tabular encoder
        self.tab_encoder = TabularEncoder(
            input_dim=self.input_dim,
            hidden_dim=128,
            out_dim=64,
        )
        # Output: (B, 64)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(img_feat_dim + 64, 256),
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