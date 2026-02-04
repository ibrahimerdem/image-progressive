import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedding(nn.Module):
    """Projects continuous features to embedding space via MLP (from SD)."""
    def __init__(self, num_features: int = 9, embed_dim: int = 512):
        super().__init__()
        self.num_features = num_features
        self.projection = nn.Sequential(
            nn.Linear(num_features, num_features * 256),
            nn.SiLU(),
            nn.Linear(num_features * 256, num_features * embed_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, F = features.shape
        return self.projection(features)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 512, image_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            # 8x8 -> 4x4
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            # 4x4 -> 1x1 (global features)
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.projection = nn.Sequential(
            nn.Linear(512, 2048),
            nn.SiLU(),
            nn.Linear(2048, 9 * embed_dim),  # 4608 to match FeatureEmbedding
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 128, 128]
        features = self.encoder(images)  # [B, 512, 1, 1]
        features = features.flatten(1)    # [B, 512]
        embedding = self.projection(features)  # [B, 4608]
        return embedding

    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar

    def forward(self, x):
        B, C, H, W = x.size()
        # Avoid subtle in-place/version-counter issues under AMP/DDP by keeping
        # the residual on a separate tensor.
        residual = x.clone()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        key = self.key(x).view(B, -1, H * W)  # (B, C//8, HW)
        attention = torch.bmm(query, key)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        out = self.gamma * out + residual
        return out

class Generator(nn.Module):
    def __init__(self,
                 channels = 3,
                 noise_dim = 100,
                 num_features = 9,
                 embed_dim = 512,
                 initial_image=False):
        super(Generator, self).__init__()
        self.channels = channels
        self.noise_dim = noise_dim
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.initial_image = initial_image
        
        # Use SD-style feature embedding (outputs 9 * 512 = 4608 dims)
        self.feature_embedding = FeatureEmbedding(num_features=num_features, embed_dim=embed_dim)
        feature_emb_dim = num_features * embed_dim  # 4608
        
        # Optional: Use SD-style image embedding
        if initial_image:
            self.image_embedding = ImageEmbedding(in_channels=3, embed_dim=embed_dim, image_size=128)
            # When using image, concatenate: 4608 + 4608 = 9216
            combined_emb_dim = feature_emb_dim * 2
        else:
            self.image_embedding = None
            combined_emb_dim = feature_emb_dim

        # Start from 8x8 feature map so 6 upsampling stages reach 512x512
        # 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512
        self.fc = nn.Linear(self.noise_dim + combined_emb_dim, 1024 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.attn = SelfAttention(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)

        self.deconv6 = nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.tanh = nn.Tanh()

    def freeze_image_encoder(self):
        """Freeze ImageEmbedding parameters."""
        if self.image_embedding is not None:
            for param in self.image_embedding.parameters():
                param.requires_grad = False
            print("Image embedding frozen")

    def unfreeze_image_encoder(self):
        """Unfreeze ImageEmbedding parameters."""
        if self.image_embedding is not None:
            for param in self.image_embedding.parameters():
                param.requires_grad = True
            print("Image embedding unfrozen for fine-tuning")

    def forward(self, noise, features, initial_image=None):
        """
        Args:
            noise: [B, noise_dim] - random noise vector
            features: [B, num_features] - continuous features in [0, 1]
            initial_image: [B, 3, 128, 128] - optional initial image
        """
        # Embed features using SD-style embedding
        feature_emb = self.feature_embedding(features)  # [B, 4608]
        
        # Optionally concatenate initial image embedding
        if self.initial_image and initial_image is not None and self.image_embedding is not None:
            image_emb = self.image_embedding(initial_image)  # [B, 4608]
            combined_emb = torch.cat([feature_emb, image_emb], dim=1)  # [B, 9216]
        else:
            combined_emb = feature_emb  # [B, 4608]
        
        # Flatten noise and combine with embeddings
        noise_flat = noise.view(noise.shape[0], -1)
        combined_features = torch.cat([noise_flat, combined_emb], dim=1)
        z = self.fc(combined_features)

        z = z.view(z.shape[0], 1024, 8, 8)

        z = F.relu(self.bn1(self.deconv1(z)))  # 16x16x512
        z = F.relu(self.bn2(self.deconv2(z)))  # 32x32x256
        z = F.relu(self.bn3(self.deconv3(z)))  # 64x64x128
        z = F.relu(self.bn4(self.deconv4(z)))  # 128x128x64
        
        z = self.attn(z)  # Self-attention at 128x128
        
        z = F.relu(self.bn5(self.deconv5(z)))  # 256x256x32
        z = self.deconv6(z)  # 512x512x3
        
        out = self.tanh(z)
        return out

class Discriminator(nn.Module):
    def __init__(self,
                 channels=3,
                 num_features=9,
                 embed_dim=512):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Use SD-style feature embedding
        self.feature_embedding = FeatureEmbedding(num_features=num_features, embed_dim=embed_dim)
        feature_emb_dim = num_features * embed_dim  # 4608

        self.conv1 = nn.Conv2d(self.channels, 32, 4, 2, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.InstanceNorm2d(64, affine=True, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.InstanceNorm2d(128, affine=True, track_running_stats=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)

        self.attn = SelfAttention(128)

        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn4 = nn.InstanceNorm2d(256, affine=True, track_running_stats=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=False)

        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn5 = nn.InstanceNorm2d(512, affine=True, track_running_stats=False)
        self.relu5 = nn.LeakyReLU(0.2, inplace=False)

        self.output = nn.Conv2d(512 + feature_emb_dim, 1, 4, 1, 0, bias=False)
        # Removed sigmoid - using BCEWithLogitsLoss instead for AMP compatibility

    def forward(self, x, features):
        """
        Args:
            x: [B, 3, H, W] - input image
            features: [B, num_features] - continuous features in [0, 1]
        """
        x_out = self.relu1(self.conv1(x))
        x_out = self.relu2(self.bn2(self.conv2(x_out)))
        x_out = self.relu3(self.bn3(self.conv3(x_out)))
        x_out = self.attn(x_out)
        x_out = self.relu4(self.bn4(self.conv4(x_out)))
        x_out = self.relu5(self.bn5(self.conv5(x_out)))

        batch_size, _, height, width = x_out.size()

        # Embed features using SD-style embedding
        feature_emb = self.feature_embedding(features)  # [B, 4608]
        feature_emb = feature_emb.view(feature_emb.size(0), feature_emb.size(1), 1, 1)
        feature_emb = feature_emb.expand(-1, -1, height, width)
        combined = torch.cat([x_out, feature_emb], dim=1)

        out = self.output(combined)
        # Return logits directly (no sigmoid)

        return out.squeeze(), x_out
    