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
        """
        Args:
            features: [B, input_dim] or [B, N, input_dim]
        Returns:
            [B, output_dim] or [B, N, output_dim]
        """
        return self.net(features)


class CrossAttention(nn.Module):
    """Cross-attention layer for conditioning on sequence features."""
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - image features (queries)
            context: [B, N, D] - conditioning features (keys, values)
        Returns:
            [B, C, H, W] - attended features
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Multi-head attention
        q = self.to_q(x_flat)  # [B, H*W, C]
        k = self.to_k(context)  # [B, N, C]
        v = self.to_v(context)  # [B, N, C]
        
        # Reshape for multi-head
        head_dim = C // self.heads
        q = q.view(B, H * W, self.heads, head_dim).permute(0, 2, 1, 3)  # [B, heads, H*W, head_dim]
        k = k.view(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        v = v.view(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, H*W, N]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, H*W, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)  # [B, H*W, C]
        out = self.to_out(out)
        
        # Reshape back to image
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, feature_dim: int, use_cross_attn: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Time conditioning via FiLM
        self.time_film = nn.Linear(time_dim, out_channels * 2)
        
        # Feature conditioning: FiLM or Cross-Attention
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttention(out_channels, feature_dim, heads=8)
            self.attn_norm = nn.GroupNorm(8, out_channels)
        else:
            # FiLM for single-vector features
            self.feature_film = nn.Linear(feature_dim, out_channels * 2)
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_dim]
            feature_emb: [B, feature_dim] or [B, N, feature_dim]
        """
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        
        # Apply time conditioning via FiLM
        time_film = self.time_film(time_emb).unsqueeze(-1).unsqueeze(-1)
        time_scale, time_shift = time_film.chunk(2, dim=1)
        time_scale = torch.clamp(time_scale, -3.0, 3.0)
        h = h * (1 + time_scale) + time_shift
        
        # Apply feature conditioning
        if self.use_cross_attn:
            # Cross-attention for sequence features [B, N, D]
            h = h + self.cross_attn(self.attn_norm(h), feature_emb)
        else:
            # FiLM for single-vector features [B, D]
            feature_film = self.feature_film(feature_emb).unsqueeze(-1).unsqueeze(-1)
            feature_scale, feature_shift = feature_film.chunk(2, dim=1)
            feature_scale = torch.clamp(feature_scale, -5.0, 5.0)
            h = h * (1 + feature_scale) + feature_shift
        
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


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        flat = normed.view(b, c, -1).permute(2, 0, 1)
        attn_out, _ = self.attn(flat, flat, flat)
        attn_out = attn_out.permute(1, 2, 0).view(b, c, h, w)
        return x + attn_out


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, feature_dim: int, attn: bool, use_cross_attn: bool = False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim, feature_dim, use_cross_attn=use_cross_attn)
        self.attn = AttentionBlock(out_channels, cfg.SD_ATTENTION_HEADS) if attn else None
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor):
        h = self.res(x, time_emb, feature_emb)
        if self.attn is not None:
            h = self.attn(h)
        return self.downsample(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, feature_dim: int, attn: bool, use_cross_attn: bool = False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim, feature_dim, use_cross_attn=use_cross_attn)
        self.attn = AttentionBlock(out_channels, cfg.SD_ATTENTION_HEADS) if attn else None
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor):
        if x.shape[-2:] != skip.shape[-2:]:
            x = self.upsample(x)
        h = torch.cat([x, skip], dim=1)
        h = self.res(h, time_emb, feature_emb)
        if self.attn is not None:
            h = self.attn(h)
        return h


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, time_dim: int, feature_dim: int, use_cross_attn: bool = False):
        super().__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.use_cross_attn = use_cross_attn
        
        # Initial conv
        self.inc = ResidualBlock(in_channels, base_channels, time_dim, feature_dim, use_cross_attn=use_cross_attn)
        
        # Downsampling with cross-attention in deeper layers
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim, feature_dim, attn=False, use_cross_attn=use_cross_attn)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim, feature_dim, attn=True, use_cross_attn=use_cross_attn)
        
        # Middle with cross-attention
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim, feature_dim, use_cross_attn=use_cross_attn)
        
        # Upsampling with cross-attention in deeper layers
        self.up3 = UpBlock(base_channels * 8, base_channels * 2, time_dim, feature_dim, attn=True, use_cross_attn=use_cross_attn)
        self.up2 = UpBlock(base_channels * 4, base_channels, time_dim, feature_dim, attn=False, use_cross_attn=use_cross_attn)
        self.up1 = UpBlock(base_channels * 2, base_channels, time_dim, feature_dim, attn=False, use_cross_attn=use_cross_attn)
        
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_dim]
            feature_emb: [B, feature_dim] or [B, N, feature_dim]
        """
        h1 = self.inc(x, time_emb, feature_emb)
        d2, skip1 = self.down1(h1, time_emb, feature_emb)
        d3, skip2 = self.down2(d2, time_emb, feature_emb)
        middle = self.mid(d3, time_emb, feature_emb)
        u3 = self.up3(middle, skip2, time_emb, feature_emb)
        u2 = self.up2(u3, skip1, time_emb, feature_emb)
        u1 = self.up1(u2, h1, time_emb, feature_emb)
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
        cfg_dropout: float = 0.1,  # Classifier-free guidance dropout
        use_x0_loss: bool = True,  # Add direct x0 prediction loss
        x0_loss_weight: float = 0.1,  # Weight for x0 loss
        feature_consistency_weight: float = 0.0,  # Feature consistency loss
    ) -> dict:
        """
        Returns dict with 'loss' and optional 'metrics' for logging
        """
        batch_size = x_start.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        
        # Classifier-free guidance: randomly drop conditioning
        # This forces the model to learn both conditional and unconditional denoising
        if cfg_dropout > 0 and self.training:
            mask = torch.rand(batch_size, device=features.device) > cfg_dropout
            # Expand mask to match feature dimensions: [B] -> [B, 1, ...] 
            while len(mask.shape) < len(features.shape):
                mask = mask.unsqueeze(-1)
            mask = mask.float()
            features = features * mask  # Zero out features for some samples
        
        pred_noise = model(x_t, t, features)
        
        # Primary loss: noise prediction
        noise_loss = F.mse_loss(pred_noise, noise)
        total_loss = noise_loss
        metrics = {'noise_loss': noise_loss.item()}
        
        # Auxiliary loss: predict x0 directly and compare to target
        if use_x0_loss:
            # Predict x0 from noisy image and predicted noise
            alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
            sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-8))
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
            
            # Direct comparison to target - this guides content generation!
            x0_loss = F.mse_loss(pred_x0, x_start)
            metrics['x0_loss'] = x0_loss.item()
            total_loss = total_loss + x0_loss_weight * x0_loss
            
            # Feature consistency: predicted x0 should have similar visual characteristics
            # Use simple perceptual consistency via normalized L1 in pixel space
            if feature_consistency_weight > 0:
                # Normalize to [0, 1] range for perceptual comparison
                pred_x0_norm = torch.clamp((pred_x0 + 1) / 2, 0, 1)
                x_start_norm = torch.clamp((x_start + 1) / 2, 0, 1)
                
                # Perceptual loss: L1 distance is more aligned with human perception than L2
                perceptual_loss = F.l1_loss(pred_x0_norm, x_start_norm)
                metrics['perceptual_loss'] = perceptual_loss.item()
                total_loss = total_loss + feature_consistency_weight * perceptual_loss
        
        return {'loss': total_loss, 'metrics': metrics}

    def sample(
        self,
        model: nn.Module,
        features: torch.Tensor,
        steps: Optional[int] = None,
        save_intermediates: bool = False,
        eta: float = 0.0,  # 0 = deterministic DDIM, 1 = full DDPM
    ):
        """
        DDIM sampling with proper timestep subsampling.
        eta=0: deterministic DDIM
        eta=1: stochastic DDPM
        """
        steps = steps or self.timesteps
        shape = (features.size(0), cfg.CHANNELS, cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)
        img = torch.randn(shape, device=features.device)
        intermediates = []
        
        # Create timestep schedule - evenly subsample if steps < timesteps
        if steps < self.timesteps:
            c = self.timesteps // steps
            timestep_schedule = torch.arange(0, self.timesteps, c, dtype=torch.long)
            timestep_schedule = torch.flip(timestep_schedule, [0])
        else:
            timestep_schedule = torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long)
        
        for step_idx, timestep in enumerate(timestep_schedule):
            t = torch.full((shape[0],), timestep, dtype=torch.long, device=img.device)
            
            # Predict noise
            epsilon = model(img, t, features)
            
            # Check for NaNs/Infs immediately after prediction
            if torch.isnan(epsilon).any() or torch.isinf(epsilon).any():
                print(f"[WARNING] NaN/Inf detected in epsilon at timestep {timestep}")
                epsilon = torch.nan_to_num(epsilon, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isnan(img).any() or torch.isinf(img).any():
                print(f"[WARNING] NaN/Inf detected in img at timestep {timestep}")
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Get alpha values
            alpha_bar_t = self._extract(self.alphas_cumprod, t, img.shape)
            
            # Get next timestep's alpha
            if step_idx < len(timestep_schedule) - 1:
                t_prev = timestep_schedule[step_idx + 1]
                alpha_bar_prev = self._extract(self.alphas_cumprod, 
                                               torch.full_like(t, t_prev), img.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            
            # Predict x0 (the clean image) using DDIM formula
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            # Clamp to prevent numerical instability
            sqrt_alpha_bar_t = torch.clamp(sqrt_alpha_bar_t, min=1e-8)
            
            pred_x0 = (img - sqrt_one_minus_alpha_bar_t * epsilon) / sqrt_alpha_bar_t
            # Don't clamp pred_x0 here - it destroys information during iterative denoising!
            
            # DDIM direction term
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev_minus_var = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - eta**2 * variance, min=0))
            dir_xt = sqrt_one_minus_alpha_bar_prev_minus_var * epsilon
            
            # Compute x_{t-1}
            x_prev = sqrt_alpha_bar_prev * pred_x0 + dir_xt
            
            # Add stochastic noise (controlled by eta)
            if eta > 0 and step_idx < len(timestep_schedule) - 1:
                noise = torch.randn_like(img)
                sigma_t = eta * torch.sqrt(variance)
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
        cond_dim: Optional[int] = None,
        use_cross_attn: bool = False,  # Enable cross-attention for sequence features
    ):
        super().__init__()
        if cond_dim is None:
            cond_dim = cfg.SD_EMB_DIM * 2
        
        time_dim = cond_dim
        feature_dim = cond_dim
        self.use_cross_attn = use_cross_attn
        
        self.feature_projection = FeatureProjector(input_dim, feature_dim)
        self.time_embedding = TimeEmbedding(time_dim)
        self.unet = ImprovedUNet(
            cfg.CHANNELS, 
            base_channels=cfg.SD_BASE_CHANNELS, 
            time_dim=time_dim, 
            feature_dim=feature_dim,
            use_cross_attn=use_cross_attn
        )
        
        # Learnable scales for conditioning
        # Feature scale MUST be stronger - features determine content!
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_scale = nn.Parameter(torch.tensor(3.0))  # Dominant over time

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_image: [B, C, H, W]
            timesteps: [B]
            features: [B, input_dim] or [B, N, input_dim] for sequence features
        Returns:
            predicted noise: [B, C, H, W]
        """
        time_emb = self.time_embedding(timesteps) * self.time_scale
        
        # Project features: handles both [B, D] and [B, N, D]
        feature_emb = self.feature_projection(features) * self.feature_scale
        
        # Pass to UNet - it will handle both single-vector and sequence features
        return self.unet(noisy_image, time_emb, feature_emb)


class StableDiffusionPipeline:
    def __init__(self, model: StableDiffusionConditioned, schedule: GaussianDiffusion):
        self.model = model
        self.schedule = schedule

    def sample(self, features: torch.Tensor, steps: Optional[int] = None, save_intermediates: bool = False):
        return self.schedule.sample(self.model, features, steps, save_intermediates=save_intermediates)


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
