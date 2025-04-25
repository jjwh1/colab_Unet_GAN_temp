
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_module import *


class GatedGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, latent_channels=32):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 5, 1, 2), # 유지
            GatedConv2d(latent_channels, latent_channels * 2, 3, 2, 1), # 반
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1), # 유지
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 2, 1), # 반
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1), # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1), # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1,  2, dilation=2
                        ),  # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2
                        ),  # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2
                        ),  # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2
                        ),  # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1), # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1), # 유지
            PCAttention(latent_channels * 4),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1),
            PCAttention(latent_channels * 2),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1),
            GatedConv2d(latent_channels, latent_channels // 2, 3, 1, 1),
            GatedConv2d(latent_channels // 2, out_channels, 3, 1, 1),
            nn.Tanh()  # 원래 gated conv 코드
            # nn.Sigmoid()   # -> 이거 쓰고 싶으면 dataset.py에서 정규화 -1 ~ 1 범위로 다시 해야함 -> 근데 마스크는 애매???
        )

        self.refine_conv = nn.Sequential(  # TT-Unet_GAN_D_100x100 그림에서 위 branch dilated conv 부분까지
            GatedConv2d(in_channels, latent_channels, 5, 1, 2), # 유지
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1), # 절반
            GatedConv2d(latent_channels, latent_channels * 2, 3, 1, 1), # 유지
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1), # 유지
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 2, 1), # 절반
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1), # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2), # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2), # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2), # 유지
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2) # 유지
        )
        self.refine_atten_1 = nn.Sequential(  # TT-GAN에서 아래 branch
            GatedConv2d(in_channels, latent_channels, 5, 1, 2),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1),
            GatedConv2d(latent_channels, latent_channels * 2, 3, 1, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 2, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1)
        )

        self.refine_combine = nn.Sequential(
            GatedConv2d(latent_channels * 8, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            PCAttention(latent_channels * 4),
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1),
            PCAttention(latent_channels * 2),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1),
            GatedConv2d(latent_channels, latent_channels // 2, 3, 1, 1),
            GatedConv2d(latent_channels // 2, out_channels, 3, 1, 1),
            nn.Tanh()   # 원래 gated conv 코드
            # nn.Sigmoid()  #-> 이거 쓰고 싶으면 dataset.py에서 정규화 -1 ~ 1 범위로 다시 해야함
        )

    def forward(self, image, mask, large_mask):
        # image : [B, 3, H, W] 짜리 박살난 이미지 1장
        # mask: 1 for mask region; 0 for unmask region (알고리즘으로 만든)
        # large_mask: 사각형으로 키운
        # Coarse
        first_in = torch.cat((image, mask), dim=1)
        first_out = self.coarse(first_in)  # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (image.shape[2], image.shape[3]))  # default : Nearest interpolation

        # Refinement
        second_masked_img = image * (1-large_mask) + first_out * large_mask
        second_in = torch.cat([second_masked_img, large_mask], dim=1)  # 애초에 mask랑 concat해서 들어가지 않으면 gated conv를 할 수 없으니 concat함.
        refine_conv = self.refine_conv(second_in)
        refine_atten = self.refine_atten_1(second_in)

        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = image * (1-large_mask) + second_out * large_mask
        second_out = nn.functional.interpolate(second_out, (image.shape[2], image.shape[3]))
        return first_out, second_out

# -----------------------------------------------
#                  Discriminator
# -----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator1(nn.Module):
    def __init__(self, in_channel = 3, latent_channel=64):
        super(PatchDiscriminator1, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(in_channel, latent_channel, 7, 1, 3, sn=True)
        self.block2 = Conv2dLayer(latent_channel, latent_channel * 2, 4, 2, 1, sn=True)
        self.block3 = Conv2dLayer(latent_channel * 2, latent_channel * 4, 4, 2, 1, sn=True)
        self.block4 = Conv2dLayer(latent_channel * 4, latent_channel * 4, 4, 2, 1, sn=True)
        self.block5 = Conv2dLayer(latent_channel * 4, latent_channel * 4, 4, 2, 1, sn=True)
        self.block6 = Conv2dLayer(latent_channel * 4, latent_channel * 4, 4, 2, 1, sn=True)
        self.flatten = nn.Flatten()

    def forward(self, img):
        x = img
        x = self.block1(x)  # out: [B, 64, 224, 224]
        x = self.block2(x)  # out: [B, 128, 112, 112]
        x = self.block3(x)  # out: [B, 256, 56, 56]
        x = self.block4(x)  # out: [B, 256, 28, 28]
        x = self.block5(x)  # out: [B, 256, 14, 14]
        x = self.block6(x)  # out: [B, 256, 7, 7]
        x = self.flatten(x)

        return x

class PatchDiscriminator2(nn.Module):
    def __init__(self, in_channel = 4, latent_channel=64):
        super(PatchDiscriminator2, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(in_channel, latent_channel, 7, 1, 3, sn=True) # 유지
        self.block2 = Conv2dLayer(latent_channel, latent_channel * 2, 4, 2, 1, sn=True) # 절반
        self.block3 = Conv2dLayer(latent_channel * 2, latent_channel * 4, 4, 2, 1, sn=True) # 절반
        self.block4 = Conv2dLayer(latent_channel * 4, latent_channel * 4, 4, 2, 1, sn=True) # 절반
        self.block5 = Conv2dLayer(latent_channel * 4, latent_channel * 4, 4, 2, 1, sn=True) # 절반
        self.block6 = Conv2dLayer(latent_channel * 4, latent_channel * 4, 4, 2, 1, sn=True) # 절반
        self.flatten = nn.Flatten()

    def forward(self, img, large_mask):  # img : gen2 거친 복원된 3채널 이미지
        x_masked = img * large_mask
        x = torch.cat((x_masked, large_mask), 1)
        x = self.block1(x)  # out: [B, 64, 224, 224]
        x = self.block2(x)  # out: [B, 128, 112, 112]
        x = self.block3(x)  # out: [B, 256, 56, 56]
        x = self.block4(x)  # out: [B, 256, 28, 28]
        x = self.block5(x)  # out: [B, 256, 14, 14]
        x = self.block6(x)  # out: [B, 256, 7, 7]
        x = self.flatten(x)
        return x