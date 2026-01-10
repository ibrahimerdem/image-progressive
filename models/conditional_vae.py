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
    """Condition a VAE on encoded images and structured recipe features."""

    def __init__(
        self,
        encoded_dim: int = 512,
        meta_dim: int = 9,
        noise_dim: int = 128,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        channels: int = 3,
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

        decoder_condition_dim = encoded_dim + meta_dim
        self.decoder_condition_mlp = nn.Sequential(
            nn.Linear(decoder_condition_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        decoder_input_dim = latent_dim + hidden_dim
        self.decoder_fc = nn.Linear(decoder_input_dim, 1024 * 4 * 4)

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
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(16)
        self.deconv7 = nn.ConvTranspose2d(16, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.output_activation = nn.Tanh()

    def encode(self, meta_features: torch.Tensor):
        hidden = self.meta_latent(meta_features)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        return mu, logvar

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
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = self.attn(x)
        x = F.relu(self.bn5(self.deconv5(x)))
        x = F.relu(self.bn6(self.deconv6(x)))
        x = self.output_activation(self.deconv7(x))
        return x

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def forward(
        self,
        encoded_image: torch.Tensor,
        meta_features: torch.Tensor,
        noise: torch.Tensor | None = None,
    ):  # noqa: D401
        if noise is None:
            noise = torch.randn(encoded_image.size(0), self.noise_dim, device=encoded_image.device)
        elif noise.size(1) != self.noise_dim:
            raise ValueError(f"Expected noise dim {self.noise_dim}, got {noise.size(1)}")

        mu, logvar = self.encode(meta_features)
        latent = self.reparameterize(mu, logvar, noise)
        reconstruction = self.decode(latent, encoded_image, meta_features)
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "noise": noise,
        }
