import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.attention import SelfAttention, CrossAttention

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
        # Convert timesteps to embedding (handles int->float conversion)
        emb = get_timestep_embedding(timesteps, self.dim)
        
        # Ensure dtype matches the model weights (important for AMP/mixed precision)
        emb = emb.to(dtype=next(self.mlp.parameters()).dtype)
        
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




class ImprovedUNet(nn.Module):
    """UNet using the new UNET_ResidualBlock and UNET_AttentionBlock components"""
    def __init__(self, in_channels: int, base_channels: int, time_dim: int, feature_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        
        # Adaptive time embedding to convert time_dim to 1280 (standard for UNET blocks)
        self.time_adapter = nn.Linear(time_dim, 1280) if time_dim != 1280 else nn.Identity()
        
        # Initial convolution
        self.inc = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder path with SwitchSequential
        self.down1 = SwitchSequential(
            UNET_ResidualBlock(base_channels, base_channels * 2, 1280),
            nn.AvgPool2d(2)
        )
        
        self.down2 = SwitchSequential(
            UNET_ResidualBlock(base_channels * 2, base_channels * 4, 1280),
            UNET_AttentionBlock(8, base_channels * 4 // 8, feature_dim),
            nn.AvgPool2d(2)
        )
        
        # Bottleneck
        self.mid = SwitchSequential(
            UNET_ResidualBlock(base_channels * 4, base_channels * 4, 1280),
            UNET_AttentionBlock(8, base_channels * 4 // 8, feature_dim),
            UNET_ResidualBlock(base_channels * 4, base_channels * 4, 1280)
        )
        
        # Decoder path with SwitchSequential  
        self.up3 = SwitchSequential(
            UNET_ResidualBlock(base_channels * 8, base_channels * 2, 1280),
            UNET_AttentionBlock(8, base_channels * 2 // 8, feature_dim),
            Upsample(base_channels * 2)
        )
        
        self.up2 = SwitchSequential(
            UNET_ResidualBlock(base_channels * 4, base_channels, 1280),
            UNET_AttentionBlock(8, base_channels // 8, feature_dim),
            Upsample(base_channels)
        )
        
        self.up1 = SwitchSequential(
            UNET_ResidualBlock(base_channels * 2, base_channels, 1280)
        )
        
        # Output convolution
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, feature_emb: torch.Tensor) -> torch.Tensor:
        # Adapt time embedding to 1280 dimensions
        time_emb_adapted = self.time_adapter(time_emb)  # [B, 1280]
        
        # Initial convolution
        h1 = self.inc(x)
        
        # Encoder
        d2 = self.down1(h1, feature_emb, time_emb_adapted)
        d3 = self.down2(d2, feature_emb, time_emb_adapted)
        
        # Bottleneck
        middle = self.mid(d3, feature_emb, time_emb_adapted)
        
        # Decoder with skip connections
        u3 = torch.cat([middle, d3], dim=1)
        u3 = self.up3(u3, feature_emb, time_emb_adapted)
        
        u2 = torch.cat([u3, d2], dim=1)
        u2 = self.up2(u2, feature_emb, time_emb_adapted)
        
        u1 = torch.cat([u2, h1], dim=1)
        u1 = self.up1(u1, feature_emb, time_emb_adapted)
        
        return self.out_conv(u1)


class TimeEmbeddingUNET(nn.Module):
    """Time embedding for UNET blocks - expects pre-computed embeddings"""
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (B, 320) - already embedded

        # (B, 320) -> (B, 1280)
        x = self.linear_1(x)
        
        # (B, 1280) -> (B, 1280)
        x = F.silu(x) 
        
        # (B, 1280) -> (B, 1280)
        x = self.linear_2(x)

        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # Reduced channel counts: 320->256, 640->512, 1280->768
        # Remove deepest layer (Height/64) to save memory
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 256, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 256, kernel_size=3, padding=1)),
            
            # (Batch_Size, 256, Height / 8, Width / 8) -> (Batch_Size, 256, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(256, 256), UNET_AttentionBlock(8, 32)),
            
            # (Batch_Size, 256, Height / 8, Width / 8) -> (Batch_Size, 256, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 256, Height / 16, Width / 16) -> (Batch_Size, 512, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(256, 512), UNET_AttentionBlock(8, 64)),
            
            # (Batch_Size, 512, Height / 16, Width / 16) -> (Batch_Size, 512, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 512, Height / 32, Width / 32) -> (Batch_Size, 768, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(512, 768), UNET_AttentionBlock(8, 96)),
            
            # (Batch_Size, 768, Height / 32, Width / 32) -> (Batch_Size, 768, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(768, 768)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 768, Height / 32, Width / 32) -> (Batch_Size, 768, Height / 32, Width / 32)
            UNET_ResidualBlock(768, 768), 
            
            # (Batch_Size, 768, Height / 32, Width / 32) -> (Batch_Size, 768, Height / 32, Width / 32)
            UNET_AttentionBlock(8, 96), 
            
            # (Batch_Size, 768, Height / 32, Width / 32) -> (Batch_Size, 768, Height / 32, Width / 32)
            UNET_ResidualBlock(768, 768), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 1536, Height / 32, Width / 32) -> (Batch_Size, 768, Height / 32, Width / 32)
            # Skip from encoder[6]: 768 channels
            SwitchSequential(UNET_ResidualBlock(1536, 768)),
            
            # (Batch_Size, 1536, Height / 32, Width / 32) -> (Batch_Size, 768, Height / 32, Width / 32)
            # Skip from encoder[5]: 768 channels
            SwitchSequential(UNET_ResidualBlock(1536, 768), UNET_AttentionBlock(8, 96)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 512, Height / 32, Width / 32) -> (Batch_Size, 512, Height / 16, Width / 16)
            # Skip from encoder[4]: 512 channels, then upsample
            SwitchSequential(UNET_ResidualBlock(1280, 512), Upsample(512)),
            
            # (Batch_Size, 1024, Height / 16, Width / 16) -> (Batch_Size, 512, Height / 16, Width / 16)
            # Skip from encoder[3]: 512 channels
            SwitchSequential(UNET_ResidualBlock(1024, 512), UNET_AttentionBlock(8, 64)),
            
            # (Batch_Size, 768, Height / 16, Width / 16) -> (Batch_Size, 256, Height / 16, Width / 16) -> (Batch_Size, 256, Height / 8, Width / 8)
            # Skip from encoder[2]: 256 channels, then upsample
            SwitchSequential(UNET_ResidualBlock(768, 256), Upsample(256)),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 256, Height / 8, Width / 8)
            # Skip from encoder[1]: 256 channels
            SwitchSequential(UNET_ResidualBlock(512, 256), UNET_AttentionBlock(8, 32)),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 256, Height / 8, Width / 8)
            # Skip from encoder[0]: 256 channels
            SwitchSequential(UNET_ResidualBlock(512, 256), UNET_AttentionBlock(8, 32)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x


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
        self.use_initial_image = use_initial_image
        
        # UNET expects context with d_context=768 dimension
        # So we project features to 768-dim space
        self.feature_projection = FeatureEmbedding(num_features=9, embed_dim=768)
        
        # TimeEmbedding outputs time_dim dimensions, but UNET expects 1280
        self.time_embedding = TimeEmbedding(320)  # 320 -> 1280 via TimeEmbeddingUNET
        self.time_adapter = TimeEmbeddingUNET(320)  # Converts 320 -> 1280
        
        # Optional: Image embedding for initial image conditioning
        if use_initial_image:
            self.image_projection = ImageEmbedding(in_channels=3, embed_dim=768, image_size=128)
        
        # Standard UNET architecture
        self.unet = UNET()
        self.output_layer = UNET_OutputLayer(256, latent_channels)  # Updated to match smaller UNET
        
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_scale = nn.Parameter(torch.tensor(5.0))  # Increased from 3.0 for stronger feature conditioning
        if use_initial_image:
            self.image_scale = nn.Parameter(torch.tensor(2.0))  # Increased from 1.0 for stronger image conditioning

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        features: torch.Tensor,
        initial_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Generate time embedding: timesteps -> (B, 320) -> (B, 1280)
        time_emb = self.time_embedding(timesteps)  # [B, 320]
        time_emb = self.time_adapter(time_emb) * self.time_scale  # [B, 1280]
        
        # Generate feature context embedding: [B, 9] -> [B, 9*768=6912]
        feature_emb = self.feature_projection(features) * self.feature_scale  # [B, 6912]
        
        # Optionally concatenate initial image embedding
        if self.use_initial_image and initial_images is not None:
            image_emb = self.image_projection(initial_images) * self.image_scale  # [B, 6912]
            # Concatenate: [B, 6912] + [B, 6912] = [B, 13824]
            feature_emb = torch.cat([feature_emb, image_emb], dim=1)
        
        # Reshape feature_emb to (B, Seq_Len, Dim) format expected by UNET
        # Split into sequence format: [B, feature_dim] -> [B, 9, 768] or [B, 18, 768]
        B = feature_emb.shape[0]
        seq_len = feature_emb.shape[1] // 768
        context = feature_emb.view(B, seq_len, 768)  # [B, 9 or 18, 768]
        
        # UNET forward: (latent, context, time)
        output = self.unet(noisy_latent, context, time_emb)
        output = self.output_layer(output)
        
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