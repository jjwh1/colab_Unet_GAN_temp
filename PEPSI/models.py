# PyTorch version of PEPSI model's encoder and decoder based on the provided TensorFlow code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, dilation=1, padding_mode='reflect'):
        super().__init__()
        pad = dilation * (k_size // 2)
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=stride,
                              padding=0, dilation=dilation)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return self.activation(x)

class ContextualBlockTFStyle(nn.Module):
    def __init__(self, in_channels=256, ksize=3, stride=1, lam=0.1):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.lam = lam
        self.in_channels = in_channels
        self.elu = nn.ELU(inplace=True)

    def forward(self, bg_in, fg_in, mask):
        B, C, H, W = bg_in.shape
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        mask_r = mask.expand(-1, C, -1, -1)
        bg = bg_in * (1 - mask_r)
        unfold = lambda x: F.unfold(x, kernel_size=self.ksize, stride=self.stride, padding=self.ksize // 2)
        fold = lambda x: F.fold(x, output_size=(H, W), kernel_size=self.ksize, stride=self.stride, padding=self.ksize // 2)
        bg_patches = unfold(bg).permute(0, 2, 1)
        fg_patches = unfold(fg_in).permute(0, 2, 1)
        bg_sq = (bg_patches ** 2).sum(dim=2, keepdim=True)
        fg_sq = (fg_patches ** 2).sum(dim=2, keepdim=True)
        sim = torch.bmm(fg_patches, bg_patches.transpose(1, 2))
        dists = bg_sq.transpose(1, 2) + fg_sq - 2 * sim
        dists_mean = dists.mean(dim=2, keepdim=True)
        dists_std = dists.std(dim=2, keepdim=True) + 1e-6
        dists_norm = (dists - dists_mean) / dists_std
        attn = F.softmax(-self.lam * torch.tanh(dists_norm), dim=-1)
        out = torch.bmm(attn, bg_patches)
        out = out.permute(0, 2, 1)
        out = fold(out)
        acl = bg + out * mask_r
        return acl


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(4, 32, 5, 1),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 64, 3, 1),
            ConvBlock(64, 128, 3, 2),
            ConvBlock(128, 128, 3, 1),
            ConvBlock(128, 256, 3, 2),
            ConvBlock(256, 256, 3, 1, dilation=2),
            ConvBlock(256, 256, 3, 1, dilation=4),
            ConvBlock(256, 256, 3, 1, dilation=8),
            ConvBlock(256, 256, 3, 1, dilation=16),
        )

    def forward(self, x):
        return self.layers(x)

class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, 3, 1)
        self.conv2 = ConvBlock(out_ch, out_ch, 3, 1)
        self.upsample = upsample

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvUpsampleBlock(256, 128, upsample=True)
        self.block2 = ConvUpsampleBlock(128, 64, upsample=True)
        self.block3 = ConvUpsampleBlock(64, 32, upsample=True)
        self.block4 = ConvUpsampleBlock(32, 16, upsample=False)
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final_conv(x)
        return torch.clamp(x, -1.0, 1.0)


class PepsiNet(nn.Module):
    def __init__(self, use_parallel_decoder=True):
        super().__init__()
        self.encoder = Encoder()
        self.cam = ContextualBlockTFStyle(in_channels=256, ksize=3, stride=1, lam=0.1)
        self.decoder = Decoder()
        self.use_parallel_decoder = use_parallel_decoder
        if self.use_parallel_decoder:
            self.coarse_decoder = self.decoder  # share the same weights

    def forward(self, image, mask):
        # image: [B, 3, H, W], mask: [B, 1, H, W]
        x = torch.cat([image, mask], dim=1)  # [B, 4, H, W]
        features = self.encoder(x)
        attended = self.cam(features, features, mask)
        output = self.decoder(attended)

        if self.use_parallel_decoder:
            coarse = self.coarse_decoder(features)  # same decoder instance
            return output, coarse
        else:
            return output



class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2))
        self.conv4 = spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2))
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2))
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        out = self.final_conv(x)
        out = torch.sum(out, dim=[1, 2, 3], keepdim=False)
        return out