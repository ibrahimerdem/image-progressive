import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        t = self.linear1(t)
        t = F.silu(t)
        t = self.linear2(t)
        return t
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(n_time, out_channels)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x, t):
        residual = x
        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv1(x)
        t = F.silu(t)
        t = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t
        x = self.gn2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.residual_conv(residual)
    

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x
    

class AttentionBlock(nn.Module):
    def __init__(self, n_head, n_emb, d_model):
        super().__init__()
        channels = n_head * n_emb
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.ln1 = nn.LayerNorm(channels)
        self.attn1 = SelfAttention(n_head=n_head, d_model=d_model, in_bias=False)
        self.ln2 = nn.LayerNorm(channels)
        self.attn2 = CrossAttention(n_head=n_head, d_model=d_model, in_bias=False)
        self.ln3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, channels * 4)
        self.linear2 = nn.Linear(channels * 4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, cond=None):
        
        residual1 = x
        x = self.gn(x)
        x = self.conv1(x)
        batch, channel, height, width = x.shape
        x = x.view(batch, channel, height * width).transpose(-1, -2)
        residual2 = x
        x = self.attn1(self.ln1(x))
        x = x + residual2
        residual2 = x
        x = self.attn2(self.ln2(x), cond)
        x = x + residual2
        residual2 = x
        x, g = self.linear1(self.ln3(x)).chunk(2, dim=-1)
        F.gelu(g)
        x = x * g
        x = self.linear2(x)
        x = x + residual2
        x = x.transpose(-1, -2).view(batch, channel, height, width)
        x = self.conv2(x)
        x = x + residual1
        return x

class SwitchSequential(nn.Sequential):

    def forward(self, x, cond, time_emb):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, cond)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x
    

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])
        self.bottleneck = nn.ModuleList([
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160), ResidualBlock(1280, 1280)),
        ])
        self.decoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ])


class UNET_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.gn(x)
        x = F.silu(x)
        x = self.conv1(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(dim=320)
        self.unet = UNET()
        self.final = UNET_out(320, 4)


    def forward(self, latent, cond, time_step):
        
        time_emb = self.time_embedding(time_step)
        output = self.unet(latent, cond, time_emb)
        output = self.final(output)
        return output