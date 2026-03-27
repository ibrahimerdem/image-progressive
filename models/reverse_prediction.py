import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import config as cfg


class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, embed_out_dim, target_shape):
        super(FeatureEmbedding, self).__init__()
        self.target_shape = target_shape
        self.feat_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_out_dim, int(np.prod(target_shape))),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.feat_embedding(x)
        batch_size = x.shape[0]
        return x.view(batch_size, *self.target_shape)
    

class Basic_Triplet(nn.Module):
    def __init__(self):
        super(Basic_Triplet, self).__init__()

        self.img_height = cfg.IMG_HEIGHT
        self.img_width = cfg.IMG_WIDTH
        self.img_channels = cfg.CHANNELS
        self.input_dim = len(cfg.FEATURE_COLUMNS)
        self.output_dim = 3

        self.embed_dim = 128
        self.embed_out_dim = 512

        self.cond_embedding = FeatureEmbedding(
            self.input_dim, self.embed_dim, self.embed_out_dim,
            (self.img_height, self.img_width, self.img_channels),
        )

        # Load a pre-trained ResNet50 model, using 'weights' argument
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if cfg.IMAGE_ENCODER=="resnet50" else None)

        # Modify the first convolutional layer of ResNet to accept 6 channels (image + conditional image)
        original_conv1 = resnet.conv1
        self.resnet_conv1 = nn.Conv2d(
            self.img_channels * 2, # Image channels + Condition embedding channels (3 + 3 = 6)
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # Copy original weights to the first 3 channels and initialize the new 3 channels
        self.resnet_conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        # Initialize new channels with small random values
        self.resnet_conv1.weight.data[:, 3:, :, :] = torch.randn(
            self.resnet_conv1.out_channels, self.img_channels, *original_conv1.kernel_size
        ) * 0.01

        # Reconstruct ResNet's feature extractor with the modified conv1
        self.resnet_features = nn.Sequential(
            self.resnet_conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        # The output of resnet_features will be a 2048-channel feature map

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # The input to fc1 will be 2048 (from ResNet GAP) + self.input_dim (from raw features)
        self.fc1 = nn.Linear(2048 + self.input_dim, 64)
        # Changed output dimension to 3 for triplet prediction
        self.fc2 = nn.Linear(64, self.output_dim)

    def forward(self, condition, image):
        # Create condition embedding (image-like)
        c1 = self.cond_embedding(condition)
        # PyTorch uses channels first format (B, C, H, W)
        c1 = c1.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        # Concatenate image and conditional embedding along the channel dimension
        fused_input = torch.cat([image, c1], dim=1) # Should be (B, 6, H, W)

        # Pass the fused input through the modified ResNet50
        resnet_output = self.resnet_features(fused_input)

        # Global pooling on ResNet features
        y = self.global_avg_pool(resnet_output)
        y = torch.flatten(y, 1) # Flatten to (batch_size, 2048)

        # Concatenate with original condition input (raw features) for the MLP head
        y = torch.cat([y, condition], dim=1)

        # MLP head
        y = self.fc1(y)
        y = self.fc2(y)

        return y