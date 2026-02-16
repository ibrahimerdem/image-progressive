import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config as cfg


class FeatureEmbedding(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Read feature configuration from config
        self.feature_columns = getattr(cfg, 'FEATURE_COLUMNS', [])
        self.num_features = len(self.feature_columns)
        
        # MLP to project all normalized features to single embedding vector
        # Input: [B, num_features] -> Output: [B, embed_dim]
        self.projection = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
        )
    
    def forward(self, features):
        # features: [B, num_features] - normalized continuous values
        # Project to single embedding vector: [B, num_features] -> [B, embed_dim]
        # e.g., [B, 9] -> [B, 512]
        embedded = self.projection(features)
        
        # Add sequence dimension for compatibility: [B, embed_dim] -> [B, 1, embed_dim]
        return embedded.unsqueeze(1)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, image_size=128, pretrained=True):
        super().__init__()
        self.embed_dim = embed_dim

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
    
        self.projection = nn.Conv2d(512, embed_dim, kernel_size=1)
    
    def forward(self, images):
        # images: [B, 3, 128, 128]
        features = self.encoder(images)  # [B, 512, 4, 4]
        embedding = self.projection(features)  # [B, embed_dim, 4, 4]
        
        # Reshape to sequence: [B, embed_dim, 4, 4] -> [B, embed_dim, 16] -> [B, 16, embed_dim]
        B, C, H, W = embedding.shape
        embedding = embedding.view(B, C, H * W).permute(0, 2, 1)  # [B, 16, embed_dim]
        
        return embedding  # [B, 16, embed_dim] - 16 image patch tokens
    
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        key = self.key(x).view(B, -1, H * W)  # (B, C//8, HW)
        attention = torch.bmm(query, key)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    def __init__(self,
                 channels = 3,
                 noise_dim = 100,
                 embed_dim = 512,
                 initial_image=True):
        super(Generator, self).__init__()
        self.channels = channels
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.initial_image = initial_image

        self.feature_embedding = FeatureEmbedding(embed_dim=embed_dim)
        
        # Image embedding: encodes 128x128 input image to [B, 16, embed_dim]
        self.image_embedding = ImageEmbedding(in_channels=3, embed_dim=embed_dim, image_size=128)
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.num_noise_tokens = 16
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, self.num_noise_tokens * embed_dim),
            nn.ReLU(),
        )

        self.to_spatial = nn.Sequential(
            nn.Linear(embed_dim, 1024 * 4 * 4),
            nn.ReLU(),
        )

        # Decoder path: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512
        # deconv1: 4x4x1024 -> 8x8x512
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        # deconv2: 8x8x512 -> 16x16x256
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        # deconv3: 16x16x256 -> 32x32x128
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Self-attention at 32x32 resolution (moved here to save memory)
        self.attn = SelfAttention(128)

        # deconv4: 32x32x128 -> 64x64x64
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        # deconv5: 64x64x64 -> 128x128x32
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)

        # deconv6: 128x128x32 -> 256x256x16
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(16)

        # deconv7: 256x256x16 -> 512x512x3 (final output)
        self.deconv7 = nn.ConvTranspose2d(16, self.channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.tanh = nn.Tanh()

    def forward(self, noise, features, initial_image):
        B = noise.shape[0]
        
        feature_emb = self.feature_embedding(features) 
        image_emb = self.image_embedding(initial_image)  
        noise_flat = noise.view(B, -1)  # [B, noise_dim]
        noise_tokens = self.noise_proj(noise_flat).view(B, self.num_noise_tokens, self.embed_dim)
        
        # Concatenate all sequences: [B, 16+1+16, embed_dim] = [B, 33, 512]
        combined_seq = torch.cat([image_emb, feature_emb, noise_tokens], dim=1)

        attn_out, _ = self.attention(combined_seq, combined_seq, combined_seq)
        attn_out = self.norm(attn_out + combined_seq)
        
        # Mean pool over sequence dimension: [B, 512]
        pooled = attn_out.mean(dim=1)
        
        # Project to spatial feature map: [B, 1024*4*4]
        spatial = self.to_spatial(pooled).view(B, 1024, 4, 4)  # [B, 1024, 4, 4]

        # Decoder path with upsampling
        z = F.relu(self.bn1(self.deconv1(spatial)))  # [B, 512, 8, 8]
        z = F.relu(self.bn2(self.deconv2(z)))  # [B, 256, 16, 16]
        z = F.relu(self.bn3(self.deconv3(z)))  # [B, 128, 32, 32]

        z = self.attn(z)  # [B, 128, 32, 32]
        
        z = F.relu(self.bn4(self.deconv4(z)))  # [B, 64, 64, 64]     
        z = F.relu(self.bn5(self.deconv5(z)))  # [B, 32, 128, 128]
        z = F.relu(self.bn6(self.deconv6(z)))  # [B, 16, 256, 256]
        z = self.tanh(self.deconv7(z))  # [B, 3, 512, 512]

        return z
    

class Discriminator(nn.Module):
    def __init__(self,
                 channels=3,
                 embed_dim=512):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim

        # Feature embedding: reads cfg.FEATURE_COLUMNS and cfg.DISCRETE_NUMS
        self.feature_embedding = FeatureEmbedding(embed_dim=embed_dim)
        
        # Project sequence to spatial: [B, 53, embed_dim] -> mean pool -> [B, embed_dim] -> [B, embed_dim, 16, 16]
        self.feature_to_spatial = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 16 * 16),
            nn.ReLU(),
        )

        # Discriminator convolution layers for 512x512 input
        # Input: [B, 3, 512, 512]
        self.conv1 = nn.Conv2d(self.channels, 32, 4, 2, 1)  # -> [B, 32, 256, 256]
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # -> [B, 64, 128, 128]
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # -> [B, 128, 64, 64]
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)

        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)  # -> [B, 256, 32, 32]
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.LeakyReLU(0.2, inplace=False)

        self.attn = SelfAttention(256)

        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)  # -> [B, 512, 16, 16]
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.2, inplace=False)

        # At 16x16 resolution: 512 image features + embed_dim features per spatial location
        self.output = nn.Conv2d(512 + embed_dim, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, features):
        B = x.shape[0]
        
        # Encode features to vector: [B, 9] -> [B, 1, embed_dim]
        feature_seq = self.feature_embedding(features)
        
        # Squeeze sequence dimension: [B, 1, embed_dim] -> [B, embed_dim]
        feature_pooled = feature_seq.squeeze(1)
        
        # Project to spatial: [B, embed_dim] -> [B, embed_dim, 16, 16]
        feature_spatial = self.feature_to_spatial(feature_pooled).view(B, self.embed_dim, 16, 16)
        
        # Process image
        x_out = self.relu1(self.conv1(x))        # [B, 32, 256, 256]
        x_out = self.relu2(self.bn2(self.conv2(x_out)))  # [B, 64, 128, 128]
        x_out = self.relu3(self.bn3(self.conv3(x_out)))  # [B, 128, 64, 64]
        x_out = self.relu4(self.bn4(self.conv4(x_out)))  # [B, 256, 32, 32]

        x_out = self.attn(x_out)                 # [B, 256, 32, 32]
        
        x_out = self.relu5(self.bn5(self.conv5(x_out)))  # [B, 512, 16, 16]

        # Concatenate image features with feature spatial representation
        combined = torch.cat([x_out, feature_spatial], dim=1)  # [B, 512 + embed_dim, 16, 16]

        out = self.output(combined)  # [B, 1, 13, 13]
        out = self.sigmoid(out)      # [B, 1, 13, 13]

        return out.squeeze(), x_out
    