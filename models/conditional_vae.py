import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channel, height, width = x.shape
        proj_query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, height * width)
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        proj_value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channel, height, width)
        return self.gamma * out + x


class ConditionalVAE(nn.Module):
    """Condition a VAE on raw initial images and structured recipe features."""

    def __init__(
        self,
        encoded_dim: int = 512,
        meta_dim: int = 9,
        noise_dim: int = 128,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        channels: int = 3
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.meta_dim = meta_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.meta_latent = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu_head = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, latent_dim)
        self.noise_proj = nn.Linear(noise_dim, latent_dim)

        # Decoder now takes encoded_image (16384 = 1024*4*4) + meta_dim
        decoder_condition_dim = 1024 * 4 * 4 + meta_dim
        self.decoder_condition_mlp = nn.Sequential(
            nn.Linear(decoder_condition_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder_input_dim = latent_dim + hidden_dim
        self.decoder_fc = nn.Linear(self.decoder_input_dim, 1024 * 4 * 4)

        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.attn = SelfAttention(64)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.image_encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Always pool to 4x4 so the flattened size stays 1024*4*4 regardless of input resolution
        self.image_pool = nn.AdaptiveAvgPool2d((4, 4))

    def encode(self, meta_features: torch.Tensor):
        hidden = self.meta_latent(meta_features)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        return mu, logvar

    def encode_image(self, initial_image: torch.Tensor):
        encoded = self.image_encoder(initial_image)
        encoded = self.image_pool(encoded)
        # Flatten to (batch, 1024*4*4)
        return encoded.view(encoded.size(0), -1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, noise: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        proj_noise = self.noise_proj(noise)
        return mu + proj_noise * std

    def decode(self, latent: torch.Tensor, encoded_image: torch.Tensor, meta_features: torch.Tensor):
        condition = torch.cat([encoded_image, meta_features], dim=1)
        condition = self.decoder_condition_mlp(condition)
        decoder_input = torch.cat([latent, condition], dim=1)
        x = self.decoder_fc(decoder_input)
        x = x.view(x.size(0), 1024, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.attn(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        return x

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def forward(
        self,
        initial_image: torch.Tensor,
        meta_features: torch.Tensor,
        noise: torch.Tensor | None = None,
    ):  # noqa: D401
        if noise is None:
            noise = torch.randn(initial_image.size(0), self.noise_dim, device=initial_image.device)
        elif noise.size(1) != self.noise_dim:
            raise ValueError(f"Expected noise dim {self.noise_dim}, got {noise.size(1)}")

        encoded_image = self.encode_image(initial_image)
        mu, logvar = self.encode(meta_features)
        latent = self.reparameterize(mu, logvar, noise)
        reconstruction = self.decode(latent, encoded_image, meta_features)

        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "noise": noise,
        }
