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

        z = self.fc(combined_features)
        z = z.view(z.shape[0], 1024, 4, 4)  # [B, 1024, 4, 4]

        # Decoder path with upsampling
        z = F.relu(self.bn1(self.deconv1(z)))  # [B, 512, 8, 8]
        z = F.relu(self.bn2(self.deconv2(z)))  # [B, 256, 16, 16]
        z = F.relu(self.bn3(self.deconv3(z)))  # [B, 128, 32, 32]
        
        # Apply self-attention at 32x32 resolution (saves memory)
        z = self.attn(z)  # [B, 128, 32, 32]
        
        z = F.relu(self.bn4(self.deconv4(z)))  # [B, 64, 64, 64]     
        z = F.relu(self.bn5(self.deconv5(z)))  # [B, 32, 128, 128]
        z = F.relu(self.bn6(self.deconv6(z)))  # [B, 16, 256, 256]
        z = self.tanh(self.deconv7(z))  # [B, 3, 512, 512]

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

        # At 16x16 resolution: 512 image features + 4608 text features per spatial location
        self.output = nn.Conv2d(512 + feature_emb_dim, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, features):
        x_out = self.relu1(self.conv1(x))        # [B, 32, 256, 256]
        x_out = self.relu2(self.bn2(self.conv2(x_out)))  # [B, 64, 128, 128]
        x_out = self.relu3(self.bn3(self.conv3(x_out)))  # [B, 128, 64, 64]
        x_out = self.relu4(self.bn4(self.conv4(x_out)))  # [B, 256, 32, 32]

        x_out = self.attn(x_out)                 # [B, 256, 32, 32]
        
        x_out = self.relu5(self.bn5(self.conv5(x_out)))  # [B, 512, 16, 16]

        _, _, height, width = x_out.size()

        feature_emb = self.feature_embedding(features)  # [B, num_features * embed_dim]
        feature_emb = feature_emb.view(feature_emb.size(0), feature_emb.size(1), 1, 1)
        feature_emb = feature_emb.expand(-1, -1, height, width)

        combined = torch.cat([x_out, feature_emb], dim=1)  # [B, 512 + feature_emb_dim, 16, 16]

        out = self.output(combined)  # [B, 1, 13, 13]
        out = self.sigmoid(out)      # [B, 1, 13, 13]

        return out.squeeze(), x_out
    