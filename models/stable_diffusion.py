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
        self.embed_dim = embed_dim
        hidden = max(num_features * 256, embed_dim * 4)
        self.projection = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_features * embed_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, F = features.shape
        return self.projection(features)  # [B, num_features * embed_dim]


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 512, image_size: int = 128):
        super().__init__()
        # Input: [B, 3, 128, 128]
        # Output: [B, 9 * embed_dim]  (matches FeatureEmbedding output dim)
        self.encoder = nn.Sequential(
            # 128×128 → 64×64
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # 64×64 → 32×32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            # 32×32 → 16×16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            # 16×16 → 8×8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            # 8×8 → 4×4
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            # 4×4 → 1×1
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Sequential(
            nn.Linear(512, 2048),
            nn.SiLU(),
            nn.Linear(2048, 9 * embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        features = self.encoder(images)   # [B, 512, 1, 1]
        features = features.flatten(1)    # [B, 512]
        return self.projection(features)  # [B, 9 * embed_dim]


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
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, context_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.time_film = nn.Linear(time_dim, out_channels * 2)
        # context_dim = emb_dim (per-token dimension, not the full flat dim)
        self.cross_attn = CrossAttention(out_channels, context_dim, heads=8, chunk_size=1024)
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
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, context_dim: int, attn: bool):
        super().__init__()
        self.res1 = ResidualBlock(in_channels,  out_channels, time_dim, context_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim, context_dim)
        self.attn = AttentionBlock(out_channels, cfg.SD_ATTENTION_HEADS) if attn else None
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor):
        h = self.res1(x, time_emb, context)
        h = self.res2(h, time_emb, context)
        if self.attn is not None:
            h = self.attn(h)
        return self.downsample(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, context_dim: int, attn: bool):
        super().__init__()
        self.res1 = ResidualBlock(in_channels,  out_channels, time_dim, context_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim, context_dim)
        self.attn = AttentionBlock(out_channels, cfg.SD_ATTENTION_HEADS) if attn else None
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor):
        if x.shape[-2:] != skip.shape[-2:]:
            x = self.upsample(x)
        h = torch.cat([x, skip], dim=1)
        h = self.res1(h, time_emb, context)
        h = self.res2(h, time_emb, context)
        if self.attn is not None:
            h = self.attn(h)
        return h


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, time_dim: int, context_dim: int):
        super().__init__()
        C  = base_channels          # 192
        C2 = base_channels * 2      # 384
        C4 = base_channels * 4      # 768
        self.time_dim    = time_dim
        self.context_dim = context_dim

        # ---- encoder ----
        # 64×64
        self.inc   = ResidualBlock(in_channels, C, time_dim, context_dim)
        # 64×64 → 32×32  (self-attn added at this resolution)
        self.down1 = DownBlock(C,  C2, time_dim, context_dim, attn=True)
        # 32×32 → 16×16
        self.down2 = DownBlock(C2, C4, time_dim, context_dim, attn=True)

        # ---- bottleneck: ResBlock → SelfAttn → ResBlock ----
        self.mid1 = ResidualBlock(C4, C4, time_dim, context_dim)
        self.mid_attn = AttentionBlock(C4, cfg.SD_ATTENTION_HEADS)
        self.mid2 = ResidualBlock(C4, C4, time_dim, context_dim)

        # ---- decoder ----
        # 16×16 → 32×32
        self.up3 = UpBlock(C4 + C4, C2, time_dim, context_dim, attn=True)
        # 32×32 → 64×64
        self.up2 = UpBlock(C2 + C2, C,  time_dim, context_dim, attn=True)
        # 64×64 (concat with inc output)
        self.up1 = UpBlock(C  + C,  C,  time_dim, context_dim, attn=False)

        self.out_conv = nn.Conv2d(C, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, 64, 64]  (latent space at 512/8)
        h0 = self.inc(x, time_emb, context)                      # [B, C,  64, 64]
        d1, skip1 = self.down1(h0, time_emb, context)            # [B, C2, 32, 32]
        d2, skip2 = self.down2(d1, time_emb, context)            # [B, C4, 16, 16]

        m = self.mid1(d2, time_emb, context)
        m = self.mid_attn(m)
        m = self.mid2(m, time_emb, context)

        u3 = self.up3(m,  skip2, time_emb, context)              # [B, C2, 32, 32]
        u2 = self.up2(u3, skip1, time_emb, context)              # [B, C,  64, 64]
        u1 = self.up1(u2, h0,    time_emb, context)              # [B, C,  64, 64]
        return self.out_conv(u1)                                  # [B, 4,  64, 64]


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-16 relu2_2 and relu3_3 feature layers.

    Input images must be in [-1, 1]. Internally converts to ImageNet space.
    The VGG backbone is always frozen.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        # relu2_2 = index 9,  relu3_3 = index 16
        self.slice1 = nn.Sequential(*list(vgg.children())[:10])   # up to relu2_2
        self.slice2 = nn.Sequential(*list(vgg.children())[10:17]) # relu2_2 → relu3_3
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in [-1, 1] → ImageNet normalised
        x = (x + 1.0) * 0.5            # [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._preprocess(pred)
        t = self._preprocess(target)
        f1_p = self.slice1(p);  f1_t = self.slice1(t)
        f2_p = self.slice2(f1_p); f2_t = self.slice2(f1_t)
        return F.mse_loss(f1_p, f1_t) + F.mse_loss(f2_p, f2_t)


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
        # Lazy: built on first p_loss call if SD_PERCEPTUAL_WEIGHT > 0
        self._perceptual: Optional[VGGPerceptualLoss] = None

    def _get_perceptual(self, device) -> Optional["VGGPerceptualLoss"]:
        if cfg.SD_PERCEPTUAL_WEIGHT <= 0:
            return None
        if self._perceptual is None:
            self._perceptual = VGGPerceptualLoss().to(device)
        return self._perceptual

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
        model,
        x_start,
        features,
        vae_encoder=None,
        vae_decoder=None,
        initial_images=None,
    ):
        device = x_start.device

        if vae_encoder is not None:
            with torch.no_grad():
                noise_for_vae = torch.randn(
                    x_start.size(0), 4,
                    x_start.size(2) // 8, x_start.size(3) // 8,
                    device=device,
                )
                x_start_latent = vae_encoder(x_start, noise_for_vae)
        else:
            x_start_latent = x_start

        batch_size = x_start_latent.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x_start_latent)
        x_t = self.q_sample(x_start_latent, t, noise)
        pred_noise = model(x_t, t, features, initial_images)
        noise_loss = F.mse_loss(pred_noise, noise)

        perceptual_loss = torch.tensor(0.0, device=device)
        perceptual_fn = self._get_perceptual(device)
        if perceptual_fn is not None and vae_decoder is not None:
            # Reconstruct pred_x0 from (x_t, pred_noise)
            sqrt_alpha_bar = torch.sqrt(self._extract(self.alphas_cumprod, t, x_t.shape))
            sqrt_one_minus = torch.sqrt(1.0 - self._extract(self.alphas_cumprod, t, x_t.shape))
            pred_x0_latent = (x_t - sqrt_one_minus * pred_noise.detach()) / sqrt_alpha_bar.clamp(min=1e-8)

            with torch.no_grad():
                pred_img  = vae_decoder(pred_x0_latent)   # [B, 3, H, W] in [-1, 1]
                pred_img  = torch.clamp(pred_img, -1.0, 1.0)
                # x_start is the original pixel image passed before VAE encoding
                tgt_img   = x_start.detach()

            perceptual_loss = perceptual_fn(pred_img, tgt_img)

        total_loss = noise_loss + cfg.SD_PERCEPTUAL_WEIGHT * perceptual_loss
        return {
            'loss': total_loss,
            'metrics': {
                'noise_loss':      noise_loss.item(),
                'perceptual_loss': perceptual_loss.item(),
            },
        }

    def sample(
        self,
        model,
        features,
        steps=None,
        save_intermediates=False,
        latent_shape=None,  # (B, C, H, W) for latent space
        initial_images=None,
    ):
        """DDPM ancestral sampler (Algorithm 2 in Ho et al. 2020).

        The reverse step is:
            x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alphabar_t) * eps_theta)
                      + sqrt(beta_tilde_t) * z,   z ~ N(0,I)  for t > 0

        where the posterior variance is:
            beta_tilde_t = (1 - alphabar_{t-1}) / (1 - alphabar_t) * beta_t

        When steps < timesteps a uniform sub-sequence is used (strided DDPM).
        In that case beta_tilde_t is computed from the *sub-sequence* alphas so
        the variance correctly reflects the larger effective step size.
        """
        steps = steps or self.timesteps
        if latent_shape is None:
            shape = (features.size(0), cfg.CHANNELS, cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)
        else:
            shape = latent_shape

        img = torch.randn(shape, device=features.device)
        intermediates = []

        # Build descending timestep schedule (T-1 → 0)
        if steps < self.timesteps:
            # Uniform stride: pick `steps` evenly spaced indices spanning [0, T-1].
            # torch.linspace guarantees endpoints are exactly 0 and T-1.
            indices = torch.linspace(0, self.timesteps - 1, steps).round().long().clamp(0, self.timesteps - 1)
            indices = torch.unique(indices)                    # remove duplicates from rounding
            timestep_schedule = torch.flip(indices, [0])      # descending: T-1 → 0
        else:
            timestep_schedule = torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long)

        for step_idx, timestep in enumerate(timestep_schedule):
            t = torch.full((shape[0],), timestep, dtype=torch.long, device=img.device)

            with torch.no_grad():
                epsilon = model(img, t, features, initial_images)

            # --- alpha_bar at current and previous timestep in the schedule ---
            alpha_bar_t = self._extract(self.alphas_cumprod, t, img.shape)

            is_last = (step_idx == len(timestep_schedule) - 1)
            if not is_last:
                t_prev = timestep_schedule[step_idx + 1]
                alpha_bar_prev = self._extract(
                    self.alphas_cumprod,
                    torch.full_like(t, t_prev),
                    img.shape,
                )
            else:
                # t == 0: no noise added, use alpha_bar=1 (clean signal)
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            # --- effective single-step alpha for this schedule stride ---
            # When steps < timesteps, each "step" covers multiple real timesteps.
            # The correct effective alpha_t for the stride is:
            #   alpha_eff = alpha_bar_t / alpha_bar_prev
            # This reduces to the true alpha_t when stride == 1.
            alpha_eff  = alpha_bar_t / alpha_bar_prev.clamp(min=1e-8)
            beta_eff   = 1.0 - alpha_eff

            # posterior variance: beta_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_eff
            beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8) * beta_eff
            beta_tilde = beta_tilde.clamp(min=0.0)

            # --- DDPM posterior mean ---
            # mu_theta = 1/sqrt(alpha_eff) * (x_t - beta_eff/sqrt(1-alpha_bar_t) * eps)
            coeff = beta_eff / (1.0 - alpha_bar_t).clamp(min=1e-8).sqrt()
            mean  = (img - coeff * epsilon) / alpha_eff.clamp(min=1e-8).sqrt()

            # --- stochastic term (zero at final step t==0) ---
            if not is_last:
                noise = torch.randn_like(img)
                img   = mean + beta_tilde.sqrt() * noise
            else:
                img   = mean

            if save_intermediates and step_idx % 10 == 0:
                intermediates.append((timestep.item(), img.clone()))

        # Latents: let the decoder handle range; images: clamp to [-1, 1]
        final = img if latent_shape is not None else torch.clamp(img, -1.0, 1.0)
        if save_intermediates:
            return final, intermediates
        return final


class StableDiffusionConditioned(nn.Module):
    def __init__(self, latent_channels=4, emb_dim=512, base_channels=64, use_initial_image=False):
        super().__init__()
        num_features = len(cfg.FEATURE_COLUMNS)   # 9
        time_dim     = emb_dim * 2                # 1536
        self.emb_dim          = emb_dim
        self.num_features     = num_features
        self.use_initial_image = use_initial_image

        self.feature_projection = FeatureEmbedding(num_features=num_features, embed_dim=emb_dim)
        self.time_embedding     = TimeEmbedding(time_dim)

        # Image conditioning: output also [B, num_features * emb_dim] → same token layout
        if use_initial_image:
            self.image_projection = ImageEmbedding(embed_dim=emb_dim)

        # Cross-attention context_dim = emb_dim (per-token)
        # Feature tokens: num_features when no image, num_features*2 when image is concatenated
        self.unet = ImprovedUNet(
            latent_channels,
            base_channels=base_channels,
            time_dim=time_dim,
            context_dim=emb_dim,        # ← per-token dimension
        )
        self.time_scale    = nn.Parameter(torch.tensor(1.0))
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
        B = noisy_latent.size(0)
        time_emb    = self.time_embedding(timesteps) * self.time_scale          # [B, time_dim]

        # Feature embedding → [B, num_features * emb_dim] → [B, num_features, emb_dim]
        feat_flat   = self.feature_projection(features) * self.feature_scale    # [B, N*emb_dim]
        context     = feat_flat.view(B, self.num_features, self.emb_dim)        # [B, N, emb_dim]

        if self.use_initial_image and initial_images is not None:
            # Image embedding → [B, num_features * emb_dim] → [B, num_features, emb_dim]
            img_flat    = self.image_projection(initial_images) * self.image_scale  # [B, N*emb_dim]
            img_tokens  = img_flat.view(B, self.num_features, self.emb_dim)         # [B, N, emb_dim]
            # Concatenate along token dimension → [B, 2N, emb_dim]
            context = torch.cat([context, img_tokens], dim=1)

        # context: [B, N_tokens, emb_dim]  — cross-attn sees N_tokens rich key-value pairs
        output = self.unet(noisy_latent, time_emb, context)
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