import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class RateAdaptiveDilatedConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=(2, 4, 8, 1)):
        super().__init__()
        self.dilations = dilations
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02)
        self.gamma = nn.Parameter(torch.ones(len(dilations), out_channels, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(len(dilations), out_channels, in_channels, 1, 1))
        self.pointwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
            for _ in dilations
        ])
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = x
        for i, d in enumerate(self.dilations):
            residual = out
            gamma_d = self.gamma[i]
            beta_d = self.beta[i]
            mod_weight = gamma_d * self.weight + beta_d
            pad = d * (self.weight.shape[-1] // 2)
            out = F.conv2d(out, mod_weight, padding=pad, dilation=d)
            out = self.elu(out)
            out = self.pointwise_convs[i](out)
            out = out + residual
            out = self.elu(out)
        return out

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
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(4, 32, kernel_size=5, stride=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
        )
        self.dpus = RateAdaptiveDilatedConvStack(256, 256)

    def forward(self, x):
        x = self.initial(x)
        x = self.dpus(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.ELU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ELU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ELU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def forward(self, x):
        return torch.clamp(self.decode(x), -1.0, 1.0)

class DietPepsiPlusNet(nn.Module):
    def __init__(self, use_parallel_decoder=True):
        super().__init__()
        self.encoder = Encoder()
        self.cam = ContextualBlockTFStyle(in_channels=256, ksize=3, stride=1, lam=0.1)
        self.decoder = Decoder()
        self.use_parallel_decoder = use_parallel_decoder
        if self.use_parallel_decoder:
            self.coarse_decoder = self.decoder  # share the same weights

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        features = self.encoder(x)
        attended = self.cam(features, features, mask)
        final = self.decoder(attended)

        if self.use_parallel_decoder:
            coarse = self.coarse_decoder(features)  # same decoder instance
            return final, coarse
        else:
            return final

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
