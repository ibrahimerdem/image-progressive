import os
import torch
import torch.nn as nn
import torch.nn.functional as F
     

class CustomEncoder(nn.Module):

    def __init__(self, input_channels=3, feature_dim=512):
        super(CustomEncoder, self).__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(input_channels, 64, 4, 2, 1)  # 128 -> 64
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 64 -> 32
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 32 -> 16
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # 16 -> 8
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)  # 8 -> 4
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Global and local feature extractors
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(1024, feature_dim)
        self.local_conv = nn.Conv2d(1024, feature_dim, 1)
        
    def forward(self, x):
        # Encoder forward pass
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)  # 64x64x64
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.2)  # 32x32x128
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.2)  # 16x16x256
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.2)  # 8x8x512
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)), 0.2)  # 4x4x1024
        
        # Global features
        global_feat = self.global_pool(x5).view(x5.size(0), -1)
        global_feat = self.global_fc(global_feat)
        
        # Local features
        local_feat = self.local_conv(x5)
        
        return {
            'global': global_feat,
            'local': local_feat,
            'skip_64': x1,
            'skip_32': x2,
            'skip_16': x3,
            'skip_8': x4
        }


def load_trained_encoder(encoder_path, device='cpu', feature_dim=512):

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found at: {encoder_path}")

    encoder = CustomEncoder(input_channels=3, feature_dim=feature_dim)

    try:
        checkpoint = torch.load(encoder_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'encoder_state_dict' in checkpoint:
                # If saved with additional metadata
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                print(f"Loaded encoder from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'loss' in checkpoint:
                    print(f"Training loss: {checkpoint['loss']:.6f}")
            elif 'model_state_dict' in checkpoint:
                # Generic model state dict format
                encoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume it's just the state dict
                encoder.load_state_dict(checkpoint)
        else:
            # Direct state dict
            encoder.load_state_dict(checkpoint)
        
        encoder.to(device)
        encoder.eval()
        
        print(f"Successfully loaded trained encoder from: {encoder_path}")
        return encoder
        
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder: {str(e)}")