import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, n_head=1):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.attn = SelfAttention(n_head=n_head, d_model=in_channels)

    def forward(self, x):
        batch, channel, height, width = x.shape
        residual = x
        x = x.view(batch, channel, height * width).transpose(-1, -2)
        x = self.attn(x)
        x = x.transpose_(-1, -2)
        x = x.view(batch, channel, height, width)
        x = x + residual
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.gn2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.residual_conv(residual)


class FeatureEncoder(nn.Module):
    def __init__(self, feature_dim = 9, hidden_dim = 64, out_channels = 3):
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, out_channels, bias=True),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor):
        batch = features.size(0)
        encoded = self.net(features)
        encoded = encoded.view(batch, self.out_channels, 1, 1)
        return encoded
    

class Encoder(nn.Sequential):
    def __init__(self, in_channels, base_channels=64):
        super().__init__(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 2, base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 4, base_channels * 4),
            AttentionBlock(base_channels * 4),
            nn.GroupNorm(num_groups=32, num_channels=base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    def forward(self, x, noise):
        # noise shape: (batch, outchannels, width/8, height/8)
        for module in self:
            x = module(x)

        mu, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        x = mu + std * noise
        x = x * 0.18215
        
        return x, mu, logvar

class Decoder(nn.Sequential):
    def __init__(self, out_channels, base_channels=64):
        super().__init__(
            nn.Conv2d(4, base_channels * 4, kernel_size=3, padding=1),
            ResidualBlock(base_channels * 4, base_channels * 4),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            ResidualBlock(base_channels * 2, base_channels * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels, base_channels),
            nn.Upsample(scale_factor=2),
            nn.GroupNorm(num_groups=32, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        x = x / 0.18215
        for module in self:
            x = module(x)
        return x


class FeatureVAE(nn.Module):
    def __init__(
        self,
        feature_dim = 9,
        hidden_dim = 128,
        latent_channels = 4,
        image_channels = 3,
    ):
        super().__init__()
        self.feature_encoder = FeatureEncoder(feature_dim, hidden_dim, latent_channels)
        self.encoder = Encoder(image_channels)
        self.decoder = Decoder(image_channels)

    def forward(self, features, target_image):
        condition = self.feature_encoder(features).expand(-1, -1, 64, 64)
        latent, mu, logvar = self.encoder(target_image, condition)
        reconstruction = self.decoder(latent + condition)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "condition": condition,
            "mu": mu,
            "logvar": logvar,
        }
    
    def generate(self, features):
        """Generate images directly from features without target image."""
        batch_size = features.shape[0]
        condition = self.feature_encoder(features).expand(-1, -1, 64, 64)
        # Sample from standard normal for latent
        latent = torch.randn(batch_size, 4, 64, 64, device=features.device) * 0.18215
        generated = self.decoder(latent + condition)
        return generated
    
    