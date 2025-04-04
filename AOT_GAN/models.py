
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetGenerator(nn.Module):
    def __init__(self):  # 1046
        super(UNetGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        # AOTBlock을 8개 연속 사용하도록 수정
        self.middle = nn.Sequential(*[AOTBlock(256) for _ in range(8)])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

    def forward(self, x):
        # x : 이미지랑 mask concat 된 것
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


# class AOTBlock(nn.Module):
#     def __init__(self, dim, rates):
#         super(AOTBlock, self).__init__()
#         self.rates = rates
#         for i, rate in enumerate(rates):
#             self.__setattr__(
#                 "block{}".format(str(i).zfill(2)),
#                 nn.Sequential(
#                     nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
#                 ),
#             )
#         self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
#         self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
#
#     def forward(self, x):
#         out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
#         out = torch.cat(out, 1)
#         out = self.fuse(out)
#         mask = my_layer_norm(self.gate(x))
#         mask = torch.sigmoid(mask)
#         return x * (1 - mask) + out * mask


class AOTBlock(nn.Module):
    def __init__(self, dim):
        super(AOTBlock, self).__init__()
        self.rates = [1, 2, 4, 8]  # 고정된 dilation rates
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(rate),
                nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                nn.ReLU(True)
            ) for rate in self.rates
        ])

        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [block(x) for block in self.blocks]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask



def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


# ----- discriminator -----
class Discriminator(nn.Module):
    def __init__(self,inc=3):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        feat = self.conv(x)
        return feat