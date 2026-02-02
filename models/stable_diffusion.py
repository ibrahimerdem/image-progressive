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
        return self.projection(features)  # [B, num_features * embed_dim]


class ImageEmbedding(nn.Module):
    """Encodes initial image to embedding space for conditioning."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 512, image_size: int = 128):
        super().__init__()
        # Convolutional encoder to extract image features
        # Input: [B, 3, 128, 128]
        # Output: [B, embed_dim * 9] (to match feature embedding dimension)
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
        
        # Project to match feature embedding dimension (9 * 512 = 4608)
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
        vae_encoder=None,
        initial_images: Optional[torch.Tensor] = None,
    ) -> dict:
        # If VAE encoder provided, encode images to latent space
        if vae_encoder is not None:
            with torch.no_grad():
                # VAE encoder expects noise for reparameterization
                noise_for_vae = torch.randn(
                    x_start.size(0), 4, 
                    x_start.size(2) // 8, x_start.size(3) // 8,
                    device=x_start.device
                )
                # VAE encoder already scales by 0.18215 internally
                x_start_latent = vae_encoder(x_start, noise_for_vae)
        else:
            x_start_latent = x_start
        
        batch_size = x_start_latent.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start_latent.device)
        noise = torch.randn_like(x_start_latent)
        x_t = self.q_sample(x_start_latent, t, noise)
        pred_noise = model(x_t, t, features, initial_images)
        noise_loss = F.mse_loss(pred_noise, noise)
        return {'loss': noise_loss, 'metrics': {'noise_loss': noise_loss.item()}}

    def sample(
        self,
        model: nn.Module,
        features: torch.Tensor,
        steps: Optional[int] = None,
        save_intermediates: bool = False,
        eta: float = 0.0,
        latent_shape: tuple = None,  # (B, C, H, W) for latent space
        initial_images: Optional[torch.Tensor] = None,
    ):
        steps = steps or self.timesteps
        if latent_shape is None:
            # Default to RGB image space (backward compatibility)
            shape = (features.size(0), cfg.CHANNELS, cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)
        else:
            shape = latent_shape
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
                epsilon = model(img, t, features, initial_images)
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
            # Clamp predicted x0: use wider range for latents, tight range for images
            if latent_shape is not None:
                # Latent space: use much wider range (VAE latents can be large)
                pred_x0 = torch.clamp(pred_x0, -20.0, 20.0)
            else:
                # Image space: standard pixel range
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            variance = torch.clamp(variance, min=0.0, max=1.0)
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev_minus_var = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - eta**2 * variance, min=0))
            dir_xt = sqrt_one_minus_alpha_bar_prev_minus_var * epsilon
            x_prev = sqrt_alpha_bar_prev * pred_x0 + dir_xt
            img = x_prev
            if save_intermediates and step_idx % 10 == 0:
                # Save intermediates without clamping (decoder will handle it)
                intermediates.append((timestep.item(), img.clone()))
        
        # Final output: don't clamp latents, let decoder handle it
        final = img if latent_shape is not None else torch.clamp(img, -1.0, 1.0)
        if save_intermediates:
            return final, intermediates
        return final


class StableDiffusionConditioned(nn.Module):
    def __init__(self, latent_channels=4, emb_dim=512, base_channels=64, use_initial_image=False):
        super().__init__()
        cond_dim = emb_dim * 2
        time_dim = cond_dim
        feature_dim = 9 * 512
        self.use_initial_image = use_initial_image
        
        self.feature_projection = FeatureEmbedding(num_features=9, embed_dim=512)
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Optional: Image embedding for initial image conditioning
        if use_initial_image:
            self.image_projection = ImageEmbedding(in_channels=3, embed_dim=512, image_size=128)
            # When using image, concatenate with features: 4608 + 4608 = 9216
            feature_dim = 9 * 512 * 2
        
        # UNet works in latent space (4 channels) not RGB space (3 channels)
        self.unet = ImprovedUNet(
            latent_channels,  # 4 channels for latent space
            base_channels=base_channels, 
            time_dim=time_dim, 
            feature_dim=feature_dim,
        )
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_scale = nn.Parameter(torch.tensor(3.0))
        if use_initial_image:
            self.image_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        features: torch.Tensor,
        initial_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps) * self.time_scale
        feature_emb = self.feature_projection(features) * self.feature_scale  # [B, 4608]
        
        # Optionally concatenate initial image embedding
        if self.use_initial_image and initial_images is not None:
            image_emb = self.image_projection(initial_images) * self.image_scale  # [B, 4608]
            # Concatenate: [B, 4608] + [B, 4608] = [B, 9216]
            feature_emb = torch.cat([feature_emb, image_emb], dim=1)
        
        feature_emb = feature_emb.unsqueeze(1)  # [B, 1, feature_dim]
        output = self.unet(noisy_latent, time_emb, feature_emb)
        return output


class StableDiffusionPipeline:
    def __init__(self, model: StableDiffusionConditioned, schedule: GaussianDiffusion, 
                 vae_encoder=None, vae_decoder=None):
        self.model = model
        self.schedule = schedule
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

    def sample(self, features: torch.Tensor, steps: Optional[int] = None, save_intermediates: bool = False,
               initial_images: Optional[torch.Tensor] = None):
        # Sample in latent space if VAE is provided
        if self.vae_encoder is not None and self.vae_decoder is not None:
            # Latent space: 4 channels, H/8, W/8
            batch_size = features.size(0)
            latent_h = cfg.TARGET_HEIGHT // 8
            latent_w = cfg.TARGET_WIDTH // 8
            latent_shape = (batch_size, 4, latent_h, latent_w)
            
            # Generate latents
            result = self.schedule.sample(
                self.model, features, steps, 
                save_intermediates=save_intermediates,
                latent_shape=latent_shape,
                initial_images=initial_images
            )
            
            if save_intermediates and isinstance(result, tuple):
                latents, intermediates = result
                # Decode final latents to images
                images = self.vae_decoder(latents)
                # Clamp decoded images to reasonable range for visualization
                images = torch.clamp(images, -1.0, 1.0)
                # Also decode intermediates
                decoded_intermediates = []
                for t, latent in intermediates:
                    img = self.vae_decoder(latent)
                    img = torch.clamp(img, -1.0, 1.0)
                    decoded_intermediates.append((t, img))
                return images, decoded_intermediates
            else:
                latents = result
                # Decode latents to images
                images = self.vae_decoder(latents)
                # Clamp decoded images to reasonable range for visualization
                images = torch.clamp(images, -1.0, 1.0)
                return images
        else:
            # Original behavior: sample directly in image space
            return self.schedule.sample(self.model, features, steps, 
                                       save_intermediates=save_intermediates,
                                       initial_images=initial_images)


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