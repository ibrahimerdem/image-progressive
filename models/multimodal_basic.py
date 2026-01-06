import torch
import torch.nn as nn
import torch.nn.functional as F

    
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
                 embed_dim = 128,
                 embed_out_dim = 256,
                 input_dim = 10,
                 initial_image=False,
                 image_encoder=None,
                 freeze_encoder=True,):
        super(Generator, self).__init__()
        self.channels = channels
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.embed_out_dim = embed_out_dim
        self.input_dim = input_dim
        self.image_encoder = image_encoder

        self.text_embedding = nn.Sequential(
            nn.Linear(self.input_dim , self.embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, self.embed_out_dim),
            nn.BatchNorm1d(self.embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc_no_image = nn.Linear(self.noise_dim + self.embed_out_dim, 1024 * 4 * 4)
        self.fc_with_image = nn.Linear(self.noise_dim + self.embed_out_dim + 512, 1024 * 4 * 4)

        if initial_image:
            if image_encoder is not None:
                self.image_encoder = image_encoder
            if freeze_encoder:
                self.freeze_image_encoder()

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.attn = SelfAttention(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)

        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(16)

        self.deconv7 = nn.ConvTranspose2d(16, self.channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.tanh = nn.Tanh()

    def freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen")

    def unfreeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        print("Image encoder unfrozen for fine-tuning")

    def forward(self, noise, text, initial_image=None):
      
        text_emb = self.text_embedding(text)
        noise_flat = noise.view(noise.shape[0], -1)
        
        if initial_image is not None:
           
            image_features = self.image_encoder(initial_image)
            combined_features = torch.cat([noise_flat, text_emb, image_features['global']], dim=1)
            z = self.fc_with_image(combined_features)
        else:
            combined_features = torch.cat([noise_flat, text_emb], dim=1)
            z = self.fc_no_image(combined_features)
            image_features = None

        z = z.view(z.shape[0], 1024, 4, 4)

        z = F.relu(self.bn1(self.deconv1(z)))  # 8x8x512
        z = F.relu(self.bn2(self.deconv2(z)))  # 16x16x256
        z = F.relu(self.bn3(self.deconv3(z)))  # 32x32x128      
        z = F.relu(self.bn4(self.deconv4(z)))  # 64x64x64      
        z = self.attn(z)
        z = F.relu(self.bn5(self.deconv5(z)))  # 128x128x32
        z = F.relu(self.bn6(self.deconv6(z)))  # 256x256x16
        z = self.tanh(self.deconv7(z))  # 512x512x3

        return z

class Discriminator(nn.Module):
    def __init__(self,
                 channels=3,
                 embed_dim=128,
                 embed_out_dim=256,
                 input_dim=10):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.embed_out_dim = embed_out_dim
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(self.channels, 32, 4, 2, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.attn = SelfAttention(128)

        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.text_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.embed_out_dim),
            nn.BatchNorm1d(self.embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output = nn.Conv2d(512 + self.embed_out_dim, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text):
        
        x_out = self.relu1(self.conv1(x))
        x_out = self.relu2(self.bn2(self.conv2(x_out)))
        x_out = self.relu3(self.bn3(self.conv3(x_out)))
        x_out = self.attn(x_out)
        x_out = self.relu4(self.bn4(self.conv4(x_out)))
        x_out = self.relu5(self.bn5(self.conv5(x_out)))

        batch_size, _, height, width = x_out.size()

        text_emb = self.text_embedding(text)
        text_emb = text_emb.view(text_emb.size(0), text_emb.size(1), 1, 1)
        text_emb = text_emb.expand(-1, -1, height, width)
        combined = torch.cat([x_out, text_emb], dim=1)

        out = self.output(combined)
        out = self.sigmoid(out)

        #x_out.view(batch_size, -1)
        return out.squeeze(), x_out
    