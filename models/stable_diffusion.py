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


class FeatureEmbedding(nn.Module):
    def __init__(self, num_features: int = 9, num_bins: int = 101, embed_dim: int = 512):
        super().__init__()
        self.num_features = num_features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_bins, embed_dim) for _ in range(num_features)
        ])
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, F = features.shape
        features_int = torch.round(features).long().clamp(0, 100)
        embs = [self.embeddings[i](features_int[:, i]) for i in range(F)]
        return torch.cat(embs, dim=1)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, chunk_size: int = 1024):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        self.chunk_size = chunk_size
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        head_dim = C // self.heads
        q = q.view(B, H * W, self.heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)
        num_queries = q.shape[2]
        out_chunks = []
        for i in range(0, num_queries, self.chunk_size):
            end = min(i + self.chunk_size, num_queries)
            q_chunk = q[:, :, i:end, :].contiguous()
            attn_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            out_chunk = torch.matmul(attn_chunk, v)
            out_chunks.append(out_chunk)
            del attn_chunk
        out = torch.cat(out_chunks, dim=2)
        del out_chunks
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
        out = self.to_out(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, feature_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.time_film = nn.Linear(time_dim, out_channels * 2)
        self.cross_attn = CrossAttention(out_channels, feature_dim, heads=8, chunk_size=1024)
        self.attn_norm = nn.GroupNorm(8, out_channels)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def _forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        time_film = self.time_film(time_emb).unsqueeze(-1).unsqueeze(-1)
        time_scale, time_shift = time_film.chunk(2, dim=1)
        time_scale = torch.clamp(time_scale, -3.0, 3.0)
        h = h * (1 + time_scale) + time_shift
        h = h + self.cross_attn(self.attn_norm(h), feature_emb)
        return self.act(h + self.residual(x))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor) -> torch.Tensor:
        return torch.utils.checkpoint.checkpoint(self._forward, x, time_emb, feature_emb, use_reentrant=False)


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
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, feature_dim: int, attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim, feature_dim)
        self.attn = AttentionBlock(out_channels, cfg.SD_ATTENTION_HEADS) if attn else None
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor):
        h = self.res(x, time_emb, feature_emb)
        if self.attn is not None:
            h = self.attn(h)
        return self.downsample(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, feature_dim: int, attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim, feature_dim)
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
    def __init__(self, in_channels: int, base_channels: int, time_dim: int, feature_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.inc = ResidualBlock(in_channels, base_channels, time_dim, feature_dim)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim, feature_dim, attn=False)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim, feature_dim, attn=True)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim, feature_dim)
        self.up3 = UpBlock(base_channels * 8, base_channels * 2, time_dim, feature_dim, attn=True)
        self.up2 = UpBlock(base_channels * 4, base_channels, time_dim, feature_dim, attn=False)
        self.up1 = UpBlock(base_channels * 2, base_channels, time_dim, feature_dim, attn=False)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor) -> torch.Tensor:
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
    ) -> dict:
        batch_size = x_start.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        pred_noise = model(x_t, t, features)
        noise_loss = F.mse_loss(pred_noise, noise)
        return {'loss': noise_loss, 'metrics': {'noise_loss': noise_loss.item()}}

    def sample(
        self,
        model: nn.Module,
        features: torch.Tensor,
        steps: Optional[int] = None,
        save_intermediates: bool = False,
        eta: float = 0.0,
    ):
        steps = steps or self.timesteps
        shape = (features.size(0), cfg.CHANNELS, cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)
        img = torch.randn(shape, device=features.device)
        intermediates = []
        if steps < self.timesteps:
            c = self.timesteps // steps
            timestep_schedule = torch.arange(0, self.timesteps, c, dtype=torch.long)
            timestep_schedule = torch.flip(timestep_schedule, [0])
        else:
            timestep_schedule = torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long)
        for step_idx, timestep in enumerate(timestep_schedule):
            t = torch.full((shape[0],), timestep, dtype=torch.long, device=img.device)
            with torch.no_grad() if not model.training else torch.enable_grad():
                epsilon = model(img, t, features)
            alpha_bar_t = self._extract(self.alphas_cumprod, t, img.shape)
            if step_idx < len(timestep_schedule) - 1:
                t_prev = timestep_schedule[step_idx + 1]
                alpha_bar_prev = self._extract(self.alphas_cumprod, 
                                               torch.full_like(t, t_prev), img.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            sqrt_alpha_bar_t = torch.clamp(sqrt_alpha_bar_t, min=1e-8)
            pred_x0 = (img - sqrt_one_minus_alpha_bar_t * epsilon) / sqrt_alpha_bar_t
            pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            variance = torch.clamp(variance, min=0.0, max=1.0)
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev_minus_var = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - eta**2 * variance, min=0))
            dir_xt = sqrt_one_minus_alpha_bar_prev_minus_var * epsilon
            x_prev = sqrt_alpha_bar_prev * pred_x0 + dir_xt
            img = x_prev
            if save_intermediates and step_idx % 10 == 0:
                intermediates.append((timestep.item(), torch.clamp(img.clone(), -1.0, 1.0)))
        
        final = torch.clamp(img, -1.0, 1.0)
        if save_intermediates:
            return final, intermediates
        return final


class StableDiffusionConditioned(nn.Module):
    def __init__(self):
        super().__init__()
        cond_dim = cfg.SD_EMB_DIM * 2
        time_dim = cond_dim
        feature_dim = 9 * 512
        self.feature_projection = FeatureEmbedding(num_features=9, num_bins=101, embed_dim=512)
        self.time_embedding = TimeEmbedding(time_dim)
        self.unet = ImprovedUNet(
            cfg.CHANNELS, 
            base_channels=cfg.SD_BASE_CHANNELS, 
            time_dim=time_dim, 
            feature_dim=feature_dim,
        )
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_scale = nn.Parameter(torch.tensor(3.0))

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps) * self.time_scale
        feature_emb = self.feature_projection(features).unsqueeze(1) * self.feature_scale
        output = self.unet(noisy_image, time_emb, feature_emb)
        return output


class StableDiffusionPipeline:
    def __init__(self, model: StableDiffusionConditioned, schedule: GaussianDiffusion):
        self.model = model
        self.schedule = schedule

    def sample(self, features: torch.Tensor, steps: Optional[int] = None, save_intermediates: bool = False):
        return self.schedule.sample(self.model, features, steps, save_intermediates=save_intermediates)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, source: nn.Module):
        src = source.module if isinstance(source, nn.parallel.DistributedDataParallel) else source
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            for name, param in src.named_parameters():
                if name in ema_params and param.dtype.is_floating_point:
                    ema_params[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

    def to(self, device):
        self.ema.to(device)
