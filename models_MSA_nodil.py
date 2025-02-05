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
