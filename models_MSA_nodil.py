import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MultiKernelConvBlock, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=self.get_same_padding(k)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for k in kernel_sizes
        ])

    def get_same_padding(self, kernel_size):
        """ 동일한 Spatial Size를 유지하기 위한 Padding 자동 계산 """
        return (kernel_size - 1) // 2

    def forward(self, x):
        return [branch(x) for branch in self.branches]  # 각 커널 적용 후 리스트 반환


class PixelAttention(nn.Module):
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.pixel_attention(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // 16, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        ca_weights = self.global_avg_pool(x).view(b, c)
        ca_weights = self.relu(self.fc1(ca_weights))
        ca_weights = self.sigmoid(self.fc2(ca_weights)).view(b, c, 1, 1)
        return x * ca_weights


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.kernel_sizes = [3, 5, 7]
        num_branches = len(self.kernel_sizes)

        self.multi_kernel_conv = MultiKernelConvBlock(in_channels, in_channels, self.kernel_sizes)
        self.pixel_attention = nn.ModuleList([PixelAttention(in_channels) for _ in range(num_branches)])
        self.channel_attention = ChannelAttention(in_channels * num_branches)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = self.multi_kernel_conv(x)
        features = [self.pixel_attention[i](features[i]) for i in range(len(features))]
        x = torch.cat(features, dim=1)
        x = self.channel_attention(x)
        return self.final_conv(x)



class UNetGenerator(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(UNetGenerator, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = AttentionBlock(128, 256)  # ⬅ AttentionBlock(128 → 256)
        self.enc4 = AttentionBlock(256, 512)  # ⬅ AttentionBlock(256 → 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final layer
        self.final_layer = nn.Conv2d(64, output_channels, kernel_size=1)  # kernel_size = 3, padding = 1 해보기

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # max_pool2d : stride 기본값 = kernel_size
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # AttentionBlock (128 → 256) & Spatial Size 감소
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # AttentionBlock (256 → 512) & Spatial Size 감소

        dec3 = self.upconv3(enc4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final layer
        return torch.sigmoid(self.final_layer(dec1))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            # 이거 224x224 기준이면 (batch_size, 1, 11, 11) 로 최종 shape 나옴. 일반적인(이상적인) Discriminator는 스칼라 하나만 뱉는거니 (batch_size,1,1,1)이 맞긴한듯? 이것도 틀린건 아닌듯함. 다른 부분도 보겠다는 거니.....
            # ㄴ그럼 이게 patch gan과 같은 거??? ㅇㅇ
            nn.Sigmoid()
            # Discriminator의 출력이 로짓(logit) 값(예: [-∞, ∞] 범위)이기 때문에, 이를 확률 값으로 변환하기 위해 sigmoid()를 사용
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)  # view까지 한 결과 (Batch * 11 * 11, 1) -> 최종 shape : (Batch * 11 * 11,)

