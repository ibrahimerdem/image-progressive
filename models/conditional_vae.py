import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention


class AttentionBlock(nn.Module):
    """Attention block with GroupNorm and residual connection."""
    def __init__(self, in_channels, n_head=1):
        super().__init__()
        # Adjust num_groups to be compatible with channel count
        num_groups = min(32, in_channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.attn = SelfAttention(n_head=n_head, d_model=in_channels)

    def forward(self, x):
        batch, channel, height, width = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(batch, channel, height * width).transpose(-1, -2)
        x = self.attn(x)
        x = x.transpose(-1, -2)
        x = x.view(batch, channel, height, width)
        x = x + residual
        return x


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Adjust num_groups to be compatible with channel counts
        num_groups = min(32, in_channels)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        num_groups_out = min(32, out_channels)
        self.gn2 = nn.GroupNorm(num_groups=num_groups_out, num_channels=out_channels)
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


class ConditionalVAE(nn.Module):
    """Condition a VAE on raw initial images and structured recipe features.
    
    Upgraded architecture with:
    - ResidualBlocks with GroupNorm
    - AttentionBlocks with proper self-attention
    - Larger base channels (128)
    - Better feature encoding with LayerNorm
    - SiLU activations
    """

    def __init__(
        self,
        meta_dim: int = 9,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        channels: int = 3,
        base_channels: int = 128,
        image_size: int = 512
    ):
        super().__init__()
        self.meta_dim = meta_dim
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_size = image_size
        
        # Feature encoder with LayerNorm for better training
        self.feature_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim, bias=True),
        )
        
        # Mu and logvar heads for VAE
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

        # Image encoder with residual blocks (removed attention for memory)
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16 (5 downsamples to 16x16)
        # Then pool to 4x4 for memory efficiency
        self.image_encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels, base_channels),
            AttentionBlock(base_channels),
            ResidualBlock(base_channels, base_channels),
            # 512 -> 256
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            # 256 -> 128
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            # 128 -> 64
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 2, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
            # 64 -> 32
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
            # 32 -> 16
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, base_channels * 4), num_channels=base_channels * 4),
            nn.SiLU(),
        )
        # Pool to consistent 4x4 size for memory efficiency
        self.image_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Image embedding dimension: base_channels * 4 * 4 * 4 = 256 * 16 = 4096
        image_embed_dim = base_channels * 4 * 4 * 4

        # Decoder input: latent_dim + image_embed_dim + meta_dim
        decoder_input_dim = latent_dim + image_embed_dim + meta_dim
        
        # Decoder projection
        self.decoder_fc = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, base_channels * 4 * 4 * 4),
        )

        # Decoder with residual blocks (removed attention for memory)
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 (7 upsamples)
        self.decoder = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4),
            # 4 -> 8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            # 8 -> 16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            ResidualBlock(base_channels * 4, base_channels * 4),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
            # 16 -> 32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            # 32 -> 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            # 64 -> 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            # 128 -> 256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
            # 256 -> 512
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            # Final output
            nn.GroupNorm(num_groups=min(32, base_channels), num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )


    def encode(self, meta_features: torch.Tensor):
        """Encode meta features to latent distribution parameters."""
        features = self.feature_encoder(meta_features)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        return mu, logvar

    def encode_image(self, initial_image: torch.Tensor):
        """Encode initial image to feature representation."""
        encoded = self.image_encoder(initial_image)
        encoded = self.image_pool(encoded)
        # Flatten to (batch, base_channels * 4 * 4 * 4)
        return encoded.view(encoded.size(0), -1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor, encoded_image: torch.Tensor, meta_features: torch.Tensor):
        """Decode latent + image + features to output image."""
        # Concatenate all conditions
        decoder_input = torch.cat([latent, encoded_image, meta_features], dim=1)
        x = self.decoder_fc(decoder_input)
        x = x.view(x.size(0), self.base_channels * 4, 4, 4)
        x = self.decoder(x)
        return x

    def forward(
        self,
        initial_image: torch.Tensor,
        meta_features: torch.Tensor,
    ):
        """Forward pass: encode, reparameterize, decode."""
        # Encode image and features
        encoded_image = self.encode_image(initial_image)
        mu, logvar = self.encode(meta_features)
        
        # Reparameterize
        latent = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decode(latent, encoded_image, meta_features)

        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
        }
    
    def generate(self, initial_image: torch.Tensor, meta_features: torch.Tensor, num_samples: int = 1):
        """Generate multiple samples from the same input."""
        encoded_image = self.encode_image(initial_image)
        mu, logvar = self.encode(meta_features)
        
        # Expand for multiple samples
        if num_samples > 1:
            encoded_image = encoded_image.repeat(num_samples, 1)
            mu = mu.repeat(num_samples, 1)
            logvar = logvar.repeat(num_samples, 1)
            meta_features = meta_features.repeat(num_samples, 1)
        
        # Sample from latent distribution
        latent = self.reparameterize(mu, logvar)
        
        # Decode
        generated = self.decode(latent, encoded_image, meta_features)
        return generated

