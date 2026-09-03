import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeatureEmbedding(nn.Module):
    def __init__(self, num_features=9, embed_dim=512):
        super().__init__()
        self.num_features = num_features
        self.projection = nn.Sequential(
            nn.Linear(num_features, num_features * 256),
            nn.SiLU(),
            nn.Linear(num_features * 256, num_features * embed_dim),
        )
    
    def forward(self, features: torch.Tensor):
        B, F = features.shape
        return self.projection(features)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, num_features=9, image_size=128, pretrained=True):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Load pre-trained ResNet18 and remove final avgpool + fc layers
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Remove avgpool and fc, keep conv layers: output [B, 512, H/32, W/32]
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global average pooling to get [B, 512, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Project to match FeatureEmbedding output dimension
        self.projection = nn.Sequential(
            nn.Linear(512, 2048),
            nn.SiLU(),
            nn.Linear(2048, num_features * embed_dim),
        )
    
    def forward(self, images):
        # images: [B, 3, 128, 128]
        features = self.encoder(images)  # [B, 512, 4, 4]
        features = self.pool(features)   # [B, 512, 1, 1]
        features = features.flatten(1)   # [B, 512]
        embedding = self.projection(features)  # [B, num_features * embed_dim]
        return embedding
    
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(SelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Conv2d(in_channels, in_channels, 1)
        self.to_k = nn.Conv2d(in_channels, in_channels, 1)
        self.to_v = nn.Conv2d(in_channels, in_channels, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar

    def forward(self, x):
        B, C, H, W = x.size()
        HW = H * W

        q = self.to_q(x).view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # [B, h, HW, hd]
        k = self.to_k(x).view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # [B, h, HW, hd]
        v = self.to_v(x).view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # [B, h, HW, hd]

        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, h, HW, HW]
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, v)  # [B, h, HW, hd]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.to_out(out)
        out = self.gamma * out + x
        return out


class CrossAttention(nn.Module):
    """Cross-attention between a spatial feature map and a sequence of
    conditioning tokens (e.g. per-feature embeddings from FeatureEmbedding /
    ImageEmbedding). The spatial map provides the queries, the conditioning
    tokens provide keys/values, letting the GAN attend to the recipe /
    initial-image conditioning at specific spatial locations.
    """

    def __init__(self, query_channels, context_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        assert query_channels % num_heads == 0, "query_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = query_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Conv2d(query_channels, query_channels, 1)
        self.to_k = nn.Linear(context_dim, query_channels)
        self.to_v = nn.Linear(context_dim, query_channels)
        self.to_out = nn.Conv2d(query_channels, query_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar

    def forward(self, x, context):
        # x: [B, C, H, W]  context: [B, N, context_dim]
        B, C, H, W = x.size()
        N = context.size(1)

        q = self.to_q(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, HW, hd]
        k = self.to_k(context).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, N, hd]
        v = self.to_v(context).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, N, hd]

        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, heads, HW, N]
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, v)  # [B, heads, HW, hd]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.to_out(out)
        out = self.gamma * out + x
        return out


class ConditionAwareSelfAttention(nn.Module):
    """Condition-aware self-attention.

    Unlike ordinary cross-attention (where queries come from the spatial
    map and keys/values come *only* from the conditioning tokens), here the
    key/value sequence is the concatenation of the spatial map's own
    self-attention projections AND the conditioning tokens' projections.
    Each spatial location attends, in a single joint softmax, to every other
    spatial location (self) as well as the recipe / initial-image
    conditioning (aware of condition) - so it behaves like self-attention
    that has been made aware of the external conditioning, rather than a
    plain self-attention -> cross-attention pipeline of two separate ops.
    """
    def __init__(self, in_channels, context_dim, num_heads=8):
        super(ConditionAwareSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Conv2d(in_channels, in_channels, 1)

        # Self (spatial) key/value projections
        self.to_k_self = nn.Conv2d(in_channels, in_channels, 1)
        self.to_v_self = nn.Conv2d(in_channels, in_channels, 1)

        # Cross (conditioning) key/value projections
        self.to_k_cross = nn.Linear(context_dim, in_channels)
        self.to_v_cross = nn.Linear(context_dim, in_channels)

        self.to_out = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar

    def forward(self, x, context):
        # x: [B, C, H, W]  context: [B, N, context_dim]
        B, C, H, W = x.size()
        HW = H * W
        N = context.size(1)

        q = self.to_q(x).view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # [B, h, HW, hd]

        k_self = self.to_k_self(x).view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # [B, h, HW, hd]
        v_self = self.to_v_self(x).view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # [B, h, HW, hd]

        k_cross = self.to_k_cross(context).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, hd]
        v_cross = self.to_v_cross(context).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, hd]

        # Joint key/value sequence: spatial (self) tokens + conditioning (cross) tokens
        k = torch.cat([k_self, k_cross], dim=2)  # [B, h, HW+N, hd]
        v = torch.cat([v_self, v_cross], dim=2)  # [B, h, HW+N, hd]

        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, h, HW, HW+N]
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, v)  # [B, h, HW, hd]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.to_out(out)
        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    def __init__(self,
                 channels = 3,
                 noise_dim = 100,
                 embed_dim = 512,
                 num_features = 9,
                 initial_image=False):
        super(Generator, self).__init__()
        self.channels = channels
        self.noise_dim = noise_dim
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.initial_image = initial_image

        # Feature embedding: outputs [B, num_features * embed_dim] = [B, 4608]
        self.feature_embedding = FeatureEmbedding(num_features=num_features, embed_dim=embed_dim)

        feature_emb_dim = num_features * embed_dim
        
        if initial_image:
            # Image embedding: encodes 128x128 input image to [B, num_features * embed_dim]
            self.image_embedding = ImageEmbedding(in_channels=3, embed_dim=embed_dim, 
                                                  num_features=num_features, image_size=128)
            # Concatenate feature embedding + image embedding
            combined_emb_dim = feature_emb_dim * 2
        else:
            self.image_embedding = None
            combined_emb_dim = feature_emb_dim
        
        # FC layer: noise (100) + combined embeddings (9216 or 4608) -> 1024*4*4
        self.fc = nn.Linear(self.noise_dim + combined_emb_dim, 1024 * 4 * 4)

        # Decoder path: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        # deconv1: 4x4x1024 -> 8x8x512
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        # deconv2: 8x8x512 -> 16x16x256
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        # deconv3: 16x16x256 -> 32x32x128
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Condition-aware self-attention at 32x32 resolution: queries are the
        # spatial feature map, keys/values are jointly built from the spatial
        # map itself (self) and the recipe/image conditioning tokens (aware
        # of condition), attended over in a single joint softmax
        self.attn = ConditionAwareSelfAttention(128, context_dim=embed_dim, num_heads=8)

        # deconv4: 32x32x128 -> 64x64x64
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        # deconv5: 64x64x64 -> 128x128x32
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)

        # deconv6: 128x128x32 -> 256x256x16
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(16)

        # final_conv: 256x256x16 -> 256x256x3 (no further upsampling, final output)
        self.final_conv = nn.Conv2d(16, self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.tanh = nn.Tanh()

    def forward(self, noise, features, initial_image=None):
        # Embed features: [B, num_features] -> [B, num_features * embed_dim]
        feature_emb = self.feature_embedding(features)

        if self.initial_image and initial_image is not None and self.image_embedding is not None:
            # Embed initial image: [B, 3, 128, 128] -> [B, num_features * embed_dim]
            image_emb = self.image_embedding(initial_image)
            # Concatenate: [B, num_features * embed_dim] + [B, num_features * embed_dim]
            combined_emb = torch.cat([feature_emb, image_emb], dim=1)
        else:
            # Use only feature embedding
            combined_emb = feature_emb

        noise_flat = noise.view(noise.shape[0], -1)  # [B, noise_dim]
        combined_features = torch.cat([noise_flat, combined_emb], dim=1)  # [B, noise_dim + emb_dim]

        # Conditioning tokens for cross-attention: reshape the flat embedding
        # into a sequence of per-feature tokens of size embed_dim
        context = combined_emb.view(combined_emb.shape[0], -1, self.embed_dim)  # [B, N, embed_dim]

        z = self.fc(combined_features)
        z = z.view(z.shape[0], 1024, 4, 4)  # [B, 1024, 4, 4]

        # Decoder path with upsampling
        z = F.relu(self.bn1(self.deconv1(z)))  # [B, 512, 8, 8]
        z = F.relu(self.bn2(self.deconv2(z)))  # [B, 256, 16, 16]
        z = F.relu(self.bn3(self.deconv3(z)))  # [B, 128, 32, 32]

        # Condition-aware self-attention: attends jointly to spatial
        # features and conditioning tokens in a single softmax
        z = self.attn(z, context)  # [B, 128, 32, 32]

        z = F.relu(self.bn4(self.deconv4(z)))  # [B, 64, 64, 64]     
        z = F.relu(self.bn5(self.deconv5(z)))  # [B, 32, 128, 128]
        z = F.relu(self.bn6(self.deconv6(z)))  # [B, 16, 256, 256]
        z = self.tanh(self.final_conv(z))  # [B, 3, 256, 256]

        return z
    

class Discriminator(nn.Module):
    def __init__(self,
                 channels=3,
                 embed_dim=512,
                 num_features=9):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.num_features = num_features

        # Feature embedding for conditioning
        self.feature_embedding = FeatureEmbedding(num_features=num_features, embed_dim=embed_dim)
        feature_emb_dim = num_features * embed_dim

        # Discriminator convolution layers for 256x256 input
        # Input: [B, 3, 256, 256]
        self.conv1 = nn.Conv2d(self.channels, 64, 4, 2, 1)  # -> [B, 64, 128, 128]
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # -> [B, 128, 64, 64]
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # -> [B, 256, 32, 32]
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)

        # Condition-aware self-attention: queries are the spatial feature
        # map, keys/values are jointly built from the spatial map itself
        # (self) and the recipe conditioning tokens (aware of condition)
        self.attn = ConditionAwareSelfAttention(256, context_dim=embed_dim, num_heads=8)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # -> [B, 512, 16, 16]
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=False)

        # At 16x16 resolution: 512 image features + 4608 text features per spatial location
        self.output = nn.Conv2d(512 + feature_emb_dim, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, features):
        feature_emb = self.feature_embedding(features)  # [B, num_features * embed_dim]
        context = feature_emb.view(feature_emb.size(0), -1, self.embed_dim)  # [B, N, embed_dim]

        x_out = self.relu1(self.conv1(x))                # [B, 64, 128, 128]
        x_out = self.relu2(self.bn2(self.conv2(x_out)))  # [B, 128, 64, 64]
        x_out = self.relu3(self.bn3(self.conv3(x_out)))  # [B, 256, 32, 32]

        x_out = self.attn(x_out, context)         # [B, 256, 32, 32]

        x_out = self.relu4(self.bn4(self.conv4(x_out)))  # [B, 512, 16, 16]

        _, _, height, width = x_out.size()

        feature_emb_map = feature_emb.view(feature_emb.size(0), feature_emb.size(1), 1, 1)
        feature_emb_map = feature_emb_map.expand(-1, -1, height, width)

        combined = torch.cat([x_out, feature_emb_map], dim=1)  # [B, 512 + feature_emb_dim, 16, 16]

        out = self.output(combined)  # [B, 1, 13, 13]
        out = self.sigmoid(out)      # [B, 1, 13, 13]

        return out.squeeze(), x_out
    