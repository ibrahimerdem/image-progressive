"""
Adds classifier-free guidance (CFG) support to the LatentDiffusionConditioned
model from the previous document, per the design discussed in this
conversation:

- A learned null-context token sequence (`null_feature_tokens`, and
  `null_image_tokens` if `use_initial_image=True`) replaces the real
  context for a random subset of training samples (`p_uncond`), so the
  model learns to denoise both conditionally and unconditionally.
- Token *count* stays identical between the conditional and unconditional
  passes (only content changes) so the joint self+cross softmax inside
  CrossAttention normalizes over the same number of keys in both cases --
  this was the compatibility concern raised earlier with
  ConditionAwareSelfAttention-style joint-softmax attention.
- `GaussianDiffusion.sample` gains a `guidance_scale` argument. With
  `guidance_scale == 1.0` CFG is effectively off (a single conditional
  forward pass, mathematically identical to the un-guided case) -- this is
  exactly the "CFG on/off" switch used by `compare_cfg_on_off.py`.

Everything not touched by CFG (TimeEmbedding, FeatureEmbedding,
ImageEmbedding, ResidualBlock, AttentionBlock, DownBlock, UpBlock,
ImprovedUNet, CrossAttention) is unchanged from the previous document and
reproduced here only so this file is self-contained.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


def get_timestep_embedding(timesteps, dim):
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
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, timesteps):
        return self.mlp(get_timestep_embedding(timesteps, self.dim))


class FeatureEmbedding(nn.Module):
    def __init__(self, num_features=9, embed_dim=512):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        hidden = max(num_features * 256, embed_dim * 4)
        self.projection = nn.Sequential(
            nn.Linear(num_features, hidden), nn.SiLU(), nn.Linear(hidden, num_features * embed_dim)
        )

    def forward(self, features):
        return self.projection(features)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, image_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.GroupNorm(8, 512), nn.SiLU(),
            nn.Conv2d(512, 512, 4, 2, 1), nn.GroupNorm(8, 512), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Linear(512, 2048), nn.SiLU(), nn.Linear(2048, 9 * embed_dim))

    def forward(self, images):
        features = self.encoder(images).flatten(1)
        return self.projection(features)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, chunk_size=1024):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        self.chunk_size = chunk_size
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k_self = nn.Linear(query_dim, query_dim, bias=False)
        self.to_v_self = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k_ctx = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v_ctx = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        q = self.to_q(x_flat)
        k = torch.cat([self.to_k_self(x_flat), self.to_k_ctx(context)], dim=1)
        v = torch.cat([self.to_v_self(x_flat), self.to_v_ctx(context)], dim=1)
        head_dim = C // self.heads
        q = q.view(B, H * W, self.heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)
        out_chunks = []
        for i in range(0, q.shape[2], self.chunk_size):
            end = min(i + self.chunk_size, q.shape[2])
            q_chunk = q[:, :, i:end, :].contiguous()
            attn_chunk = F.softmax(torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale, dim=-1)
            out_chunks.append(torch.matmul(attn_chunk, v))
        out = torch.cat(out_chunks, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
        out = self.to_out(out)
        return out.permute(0, 2, 1).view(B, C, H, W)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, context_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.time_film = nn.Linear(time_dim, out_channels * 2)
        self.cross_attn = CrossAttention(out_channels, context_dim, heads=8, chunk_size=1024)
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def _forward(self, x, time_emb, feature_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        time_scale, time_shift = self.time_film(time_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        time_scale = torch.clamp(time_scale, -3.0, 3.0)
        h = h * (1 + time_scale) + time_shift
        h = h + self.cross_attn(self.attn_norm(h), feature_emb)
        return self.act(h + self.residual(x))

    def forward(self, x, time_emb, feature_emb):
        return torch.utils.checkpoint.checkpoint(self._forward, x, time_emb, feature_emb, use_reentrant=False)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=False)

    def forward(self, x):
        b, c, h, w = x.shape
        flat = self.norm(x).view(b, c, -1).permute(2, 0, 1)
        attn_out, _ = self.attn(flat, flat, flat)
        return x + attn_out.permute(1, 2, 0).view(b, c, h, w)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, context_dim, attn):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim, context_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim, context_dim)
        self.attn = AttentionBlock(out_channels, cfg.ATTENTION_HEADS) if attn else None
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, time_emb, context):
        h = self.res1(x, time_emb, context)
        h = self.res2(h, time_emb, context)
        if self.attn is not None:
            h = self.attn(h)
        return self.downsample(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, context_dim, attn):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim, context_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim, context_dim)
        self.attn = AttentionBlock(out_channels, cfg.ATTENTION_HEADS) if attn else None
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, skip, time_emb, context):
        if x.shape[-2:] != skip.shape[-2:]:
            x = self.upsample(x)
        h = torch.cat([x, skip], dim=1)
        h = self.res1(h, time_emb, context)
        h = self.res2(h, time_emb, context)
        if self.attn is not None:
            h = self.attn(h)
        return h


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels, base_channels, time_dim, context_dim):
        super().__init__()
        C, C2, C4 = base_channels, base_channels * 2, base_channels * 4
        self.inc = ResidualBlock(in_channels, C, time_dim, context_dim)
        self.down1 = DownBlock(C, C2, time_dim, context_dim, attn=True)
        self.down2 = DownBlock(C2, C4, time_dim, context_dim, attn=True)
        self.mid1 = ResidualBlock(C4, C4, time_dim, context_dim)
        self.mid_attn = AttentionBlock(C4, cfg.ATTENTION_HEADS)
        self.mid2 = ResidualBlock(C4, C4, time_dim, context_dim)
        self.up3 = UpBlock(C4 + C4, C2, time_dim, context_dim, attn=True)
        self.up2 = UpBlock(C2 + C2, C, time_dim, context_dim, attn=True)
        self.up1 = UpBlock(C + C, C, time_dim, context_dim, attn=False)
        self.out_conv = nn.Conv2d(C, in_channels, kernel_size=1)

    def forward(self, x, time_emb, context):
        h0 = self.inc(x, time_emb, context)
        d1, skip1 = self.down1(h0, time_emb, context)
        d2, skip2 = self.down2(d1, time_emb, context)
        m = self.mid2(self.mid_attn(self.mid1(d2, time_emb, context)), time_emb, context)
        u3 = self.up3(m, skip2, time_emb, context)
        u2 = self.up2(u3, skip1, time_emb, context)
        u1 = self.up1(u2, h0, time_emb, context)
        return self.out_conv(u1)


class LatentDiffusionConditioned(nn.Module):
    """CFG-enabled version. New vs. the previous document:
    `p_uncond`, `null_feature_tokens`, `null_image_tokens`, and the
    `force_uncond` forward argument used at sampling time.
    """
    def __init__(self, latent_channels=4, emb_dim=512, base_channels=64,
                 use_initial_image=False, p_uncond=0.1):
        super().__init__()
        num_features = len(cfg.FEATURE_COLUMNS)
        time_dim = emb_dim * 2
        self.emb_dim = emb_dim
        self.num_features = num_features
        self.use_initial_image = use_initial_image
        self.p_uncond = p_uncond

        self.feature_projection = FeatureEmbedding(num_features=num_features, embed_dim=emb_dim)
        self.time_embedding = TimeEmbedding(time_dim)
        if use_initial_image:
            self.image_projection = ImageEmbedding(embed_dim=emb_dim)

        self.unet = ImprovedUNet(latent_channels, base_channels=base_channels,
                                  time_dim=time_dim, context_dim=emb_dim)
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_scale = nn.Parameter(torch.tensor(3.0))
        if use_initial_image:
            self.image_scale = nn.Parameter(torch.tensor(1.0))

        # CFG: learned "empty condition" tokens, same shape as the real
        # per-source token blocks so the joint self+cross softmax always
        # sees the same number of keys, conditional or not.
        self.null_feature_tokens = nn.Parameter(torch.randn(1, num_features, emb_dim) * 0.02)
        if use_initial_image:
            self.null_image_tokens = nn.Parameter(torch.randn(1, num_features, emb_dim) * 0.02)

    def forward(self, noisy_latent, timesteps, features, initial_images=None, force_uncond=None):
        """
        force_uncond:
          None       -> training-time random dropout (uses self.p_uncond, only if self.training)
          True/False -> Python bool, forces every sample in the batch to
                         use the null / real context (used by CFG sampling)
          BoolTensor[B] -> per-sample override (rarely needed, provided for flexibility)
        """
        B = noisy_latent.size(0)
        time_emb = self.time_embedding(timesteps) * self.time_scale

        feat_flat = self.feature_projection(features) * self.feature_scale
        context = feat_flat.view(B, self.num_features, self.emb_dim)
        null_tokens = self.null_feature_tokens.expand(B, -1, -1)

        if self.use_initial_image and initial_images is not None:
            img_flat = self.image_projection(initial_images) * self.image_scale
            img_tokens = img_flat.view(B, self.num_features, self.emb_dim)
            context = torch.cat([context, img_tokens], dim=1)
            null_tokens = torch.cat([null_tokens, self.null_image_tokens.expand(B, -1, -1)], dim=1)

        if force_uncond is None:
            if self.training and self.p_uncond > 0:
                drop = (torch.rand(B, device=context.device) < self.p_uncond).view(B, 1, 1)
                context = torch.where(drop, null_tokens, context)
        elif isinstance(force_uncond, bool):
            context = null_tokens if force_uncond else context
        else:
            mask = force_uncond.to(context.device).view(B, 1, 1)
            context = torch.where(mask, null_tokens, context)

        return self.unet(noisy_latent, time_emb, context)


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
        return arr.gather(0, timesteps).view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = torch.sqrt(self._extract(self.alphas_cumprod, t, x_start.shape))
        sqrt_omab = torch.sqrt(1.0 - self._extract(self.alphas_cumprod, t, x_start.shape))
        return sqrt_ab * x_start + sqrt_omab * noise

    def predict_start(self, x_t, t, noise):
        sqrt_ab = torch.sqrt(self._extract(self.alphas_cumprod, t, x_t.shape))
        sqrt_omab = torch.sqrt(1.0 - self._extract(self.alphas_cumprod, t, x_t.shape))
        return (x_t - sqrt_omab * noise) / sqrt_ab

    def p_loss(self, model, x_start, features, vae_encoder=None, initial_images=None,
               vae_decoder=None, rgb_loss_weight=0.0, rgb_loss_max_timestep_ratio=0.7):
        device = x_start.device
        if vae_encoder is not None:
            with torch.no_grad():
                noise_for_vae = torch.randn(x_start.size(0), 4, x_start.size(2) // 8,
                                             x_start.size(3) // 8, device=device)
                x_start_latent = vae_encoder(x_start, noise_for_vae)
        else:
            x_start_latent = x_start

        batch_size = x_start_latent.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x_start_latent)
        x_t = self.q_sample(x_start_latent, t, noise)
        # CFG dropout (if any) happens inside model.forward via self.training/p_uncond
        pred_noise = model(x_t, t, features, initial_images)
        noise_loss = F.mse_loss(pred_noise, noise)

        rgb_loss = torch.zeros((), device=device)
        if vae_decoder is not None and rgb_loss_weight > 0:
            max_t = int(rgb_loss_max_timestep_ratio * self.timesteps)
            gate_mask = t < max_t
            if gate_mask.any():
                x0_pred = self.predict_start(x_t[gate_mask], t[gate_mask], pred_noise[gate_mask])
                decoded_rgb = vae_decoder(x0_pred)
                rgb_loss = F.l1_loss(decoded_rgb, x_start[gate_mask])

        total_loss = noise_loss + rgb_loss_weight * rgb_loss
        return {"loss": total_loss, "metrics": {"noise_loss": noise_loss.item(), "rgb_loss": rgb_loss.item()}}

    def sample(self, model, features, steps=None, save_intermediates=False, latent_shape=None,
               initial_images=None, temperature=1.0, eta=0.0, guidance_scale=1.0):
        """
        guidance_scale == 1.0  -> CFG OFF: a single conditional forward pass
                                   per step (mathematically identical to no
                                   guidance -- this is the "CFG off" branch
                                   used for the ablation).
        guidance_scale  > 1.0  -> CFG ON: two forward passes per step
                                   (conditional + unconditional), combined
                                   as eps = eps_uncond + scale*(eps_cond - eps_uncond).
        """
        steps = steps or self.timesteps
        shape = latent_shape if latent_shape is not None else (
            features.size(0), cfg.CHANNELS, cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH)

        img = torch.randn(shape, device=features.device)
        intermediates = []

        if steps < self.timesteps:
            step_ratio = torch.linspace(0, 1, steps) ** 2
            indices = (step_ratio * (self.timesteps - 1)).round().long().clamp(0, self.timesteps - 1)
            indices = torch.unique(indices)
            timestep_schedule = torch.flip(indices, [0])
        else:
            timestep_schedule = torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long)

        use_cfg = guidance_scale is not None and guidance_scale != 1.0

        for step_idx, timestep in enumerate(timestep_schedule):
            t = torch.full((shape[0],), timestep, dtype=torch.long, device=img.device)

            with torch.no_grad():
                if use_cfg:
                    eps_cond = model(img, t, features, initial_images, force_uncond=False)
                    eps_uncond = model(img, t, features, initial_images, force_uncond=True)
                    epsilon = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    epsilon = model(img, t, features, initial_images)

            alpha_bar_t = self._extract(self.alphas_cumprod, t, img.shape)
            is_last = (step_idx == len(timestep_schedule) - 1)
            if not is_last:
                t_prev = timestep_schedule[step_idx + 1]
                alpha_bar_prev = self._extract(self.alphas_cumprod, torch.full_like(t, t_prev), img.shape)
            else:
                alpha_bar_prev = self._extract(self.alphas_cumprod, torch.zeros_like(t), img.shape)

            sqrt_ab_t = alpha_bar_t.sqrt().clamp(min=1e-8)
            sqrt_omab_t = (1.0 - alpha_bar_t).clamp(min=1e-8).sqrt()
            x0_pred = (img - sqrt_omab_t * epsilon) / sqrt_ab_t

            if timestep > 200:
                x0_pred = torch.clamp(x0_pred, -2.0, 2.0)

            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)
                            * (1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))).clamp(min=0).sqrt()
            direction = (1 - alpha_bar_prev - sigma ** 2).clamp(min=1e-8).sqrt() * epsilon
            img = alpha_bar_prev.sqrt() * x0_pred + direction

            if not is_last and eta > 0:
                img = img + sigma * torch.randn_like(img) * temperature

            if save_intermediates and step_idx % 10 == 0:
                intermediates.append((timestep.item(), img.clone()))

        final = img if latent_shape is not None else torch.clamp(img, -1.0, 1.0)
        return (final, intermediates) if save_intermediates else final