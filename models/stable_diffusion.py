import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros(*embedding.shape[:-1], 1, device=timesteps.device)), dim=-1)
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = get_timestep_embedding(timesteps, self.dim)
        return self.mlp(emb)


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = max(output_dim, input_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.film = nn.Linear(cond_dim, out_channels * 2)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        film = self.film(cond).unsqueeze(-1).unsqueeze(-1)
        scale, shift = film.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        return self.act(h + self.residual(x))


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, cond_dim: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.inc = ResidualBlock(in_channels, base_channels, cond_dim)
        self.down1 = ResidualBlock(base_channels, base_channels * 2, cond_dim)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, cond_dim)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.up3 = ResidualBlock(base_channels * 8, base_channels * 2, cond_dim)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, cond_dim)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, cond_dim)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h1 = self.inc(x, cond)
        h2 = self.down1(self.pool(h1), cond)
        h3 = self.down2(self.pool(h2), cond)
        middle = self.mid(self.pool(h3), cond)
        u3 = self.up3(torch.cat([self.upsample(middle), h3], dim=1), cond)
        u2 = self.up2(torch.cat([self.upsample(u3), h2], dim=1), cond)
        u1 = self.up1(torch.cat([self.upsample(u2), h1], dim=1), cond)
        return self.out_conv(u1)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.timesteps = timesteps

    def _extract(self, arr, timesteps, shape):
        out = arr.gather(0, timesteps).view(-1, *([1] * (len(shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = torch.sqrt(self._extract(self.alphas_cumprod, t, x_start.shape))
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self._extract(self.alphas_cumprod, t, x_start.shape))
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_start(self, x_t, t, noise):
        sqrt_alpha_bar = torch.sqrt(self._extract(self.alphas_cumprod, t, x_t.shape))
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self._extract(self.alphas_cumprod, t, x_t.shape))
        return (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar

    def p_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        features: torch.Tensor,
        initial_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x_start.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        pred_noise = model(x_t, t, features)
        return F.mse_loss(pred_noise, noise)

    def sample(
        self,
        model: nn.Module,
        features: torch.Tensor,
        initial_image: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        steps = steps or self.timesteps
        shape = (features.size(0), cfg.CHANNELS, cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)
        img = torch.randn(shape, device=features.device)
        for i in reversed(range(steps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=img.device)
            epsilon = model(img, t, features, initial_image)
            beta_t = self._extract(self.betas, t, img.shape)
            alpha_bar_t = self._extract(self.alphas_cumprod, t, img.shape)
            
            # Get next timestep's alpha
            if step_idx < len(timestep_schedule) - 1:
                t_prev = timestep_schedule[step_idx + 1].item()
                alpha_bar_prev = self._extract(self.alphas_cumprod, 
                                               torch.tensor([t_prev], device=img.device), img.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            
            # Predict x0 (the clean image)
            pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * epsilon) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - eta**2 * (1 - alpha_bar_t / alpha_bar_prev) * (1 - alpha_bar_prev)) * epsilon
            
            # Compute x_{t-1}
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            # Add noise (DDPM stochastic term)
            if eta > 0 and step_idx < len(timestep_schedule) - 1:
                noise = torch.randn_like(img)
                sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                x_prev = x_prev + sigma_t * noise
            
            img = x_prev
            
            # Save intermediate at key steps
            if save_intermediates and step_idx % 10 == 0:
                intermediates.append((timestep.item(), torch.clamp(img.clone(), -1.0, 1.0)))
        
        final = torch.clamp(img, -1.0, 1.0)
        if save_intermediates:
            return final, intermediates
        return final


class StableDiffusionConditioned(nn.Module):
    def __init__(
        self,
        input_dim: int,
        use_initial: bool = True,
        cond_dim: Optional[int] = None,
        initial_encoder_ckpt: Optional[str] = None,
        freeze_initial_encoder: bool = False,
    ):
        super().__init__()
        if cond_dim is None:
            cond_dim = cfg.SD_EMB_DIM * 2
        self.use_initial = use_initial
        self.feature_projection = FeatureProjector(input_dim, cond_dim)
        self.time_embedding = TimeEmbedding(cond_dim)
        self.initial_encoder = InitialImageEncoder(cfg.CHANNELS, cond_dim)
        self.unet = ImprovedUNet(cfg.CHANNELS, base_channels=cfg.SD_BASE_CHANNELS, cond_dim=cond_dim)
        self.time_scale = nn.Parameter(torch.tensor(0.8))
        self.feature_scale = nn.Parameter(torch.tensor(1.0))
        self.initial_scale = nn.Parameter(torch.tensor(1.0))

        if initial_encoder_ckpt:
            self._load_initial_encoder(initial_encoder_ckpt)
        if freeze_initial_encoder:
            for p in self.initial_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        features: torch.Tensor,
        initial_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps) * self.time_scale
        feature_emb = self.feature_projection(features) * self.feature_scale
        cond = time_emb + feature_emb
        if self.use_initial and initial_image is not None:
            cond = cond + self.initial_encoder(initial_image) * self.initial_scale
        return self.unet(noisy_image, cond)


class StableDiffusionPipeline:
    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    def sample(self, features: torch.Tensor, initial_image: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        return self.schedule.sample(self.model, features, initial_image, steps)


class ModelEMA:
    """Exponential Moving Average of model parameters for stable training."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, source: nn.Module):
        """Update EMA parameters with source model parameters."""
        src = source.module if isinstance(source, nn.parallel.DistributedDataParallel) else source
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            for name, param in src.named_parameters():
                if name in ema_params and param.dtype.is_floating_point:
                    ema_params[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    def state_dict(self):
        """Return EMA model state dict."""
        return self.ema.state_dict()

    def to(self, device):
        """Move EMA model to device."""
        self.ema.to(device)
