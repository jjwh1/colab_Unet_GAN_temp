import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super(MLP, self).__init__()

        self.adaptor = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1))

    def forward(self, x):
        return self.adaptor(x)


class Fitnets(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super(Fitnets, self).__init__()

        self.adaptor = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def forward(self, x):
        return self.adaptor(x)



#################PCatt######################

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
        b, c, h, w = x.size()
        ca_weights = self.global_avg_pool(x).view(b, c)
        ca_weights = self.relu(self.fc1(ca_weights))
        ca_weights = self.sigmoid(self.fc2(ca_weights)).view(b, c, 1, 1)
        return x * ca_weights


class PCatt(nn.Module):
    def __init__(self, input_channels, teacher_channels):
        super(PCatt, self).__init__()
        self.pixel_attention = PixelAttention(input_channels)

        # 1x1 Conv to match teacher feature channels
        self.feature_projection = nn.Conv2d(input_channels, teacher_channels, kernel_size=1)

        self.conv = nn.Conv2d(input_channels + teacher_channels, teacher_channels, kernel_size=1)
        # Channel Attention for the concatenated features
        self.channel_attention = ChannelAttention(teacher_channels)

    

    def forward(self, x):
        # Pixel attention path
        pixel_attended = self.pixel_attention(x)  # (B, input_channels, H, W)

        # 1x1 conv to match teacher channel
        projected = self.feature_projection(x)  # (B, teacher_channels, H, W)

        # Concat along channel dimension
        fused = torch.cat([pixel_attended, projected], dim=1)  # (B, teacher_channels*2, H, W)
        fused = self.conv(fused)
        # Channel attention applied to concatenated feature
        output = self.channel_attention(fused)

        

        return output


class PCmul(nn.Module):
    def __init__(self, input_channels, teacher_channels):
        super(PCmul, self).__init__()
        self.pixel_attention = PixelAttention(input_channels)

        # 1x1 Conv to match teacher feature channels
        self.feature_projection = nn.Conv2d(input_channels, teacher_channels, kernel_size=1)

        self.conv = nn.Conv2d(input_channels + teacher_channels, teacher_channels, kernel_size=1)
        # Channel Attention for the concatenated features
        self.channel_attention = ChannelAttention(teacher_channels)



    def forward(self, x):
        # Pixel attention path
        pixel_attended = self.pixel_attention(x)  # (B, input_channels, H, W)

        # 1x1 conv to match teacher channel
        projected = self.feature_projection(x)  # (B, teacher_channels, H, W)
        projected2 = self.feature_projection(x)  # (B, teacher_channels, H, W)

        mul_projected = projected * projected2

        # Concat along channel dimension
        fused = torch.cat([pixel_attended, mul_projected], dim=1)  # (B, teacher_channels*2, H, W)
        fused = self.conv(fused)
        # Channel attention applied to concatenated feature
        output = self.channel_attention(fused)

        return output


class PCconv(nn.Module):
    def __init__(self, input_channels, teacher_channels):
        super(PCconv, self).__init__()
        self.pixel_attention = PixelAttention(input_channels)

        # 1x1 Conv to match teacher feature channels
        self.feature_projection = nn.Conv2d(input_channels, teacher_channels, kernel_size=1)

        self.conv = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1)
        # Channel Attention for the concatenated features
        self.channel_attention = ChannelAttention(teacher_channels)

    def forward(self, x):
        # Pixel attention path
        pixel_attended = self.pixel_attention(x)  # (B, input_channels, H, W)

        # 1x1 conv to match teacher channel
        projected = self.feature_projection(pixel_attended)  # (B, teacher_channels, H, W)
        

        # Channel attention applied to concatenated feature
        projected = self.channel_attention(projected)

        return projected
###########################################################

class PCatt_justP(nn.Module):
    def __init__(self, input_channels, teacher_channels):
        super(PCatt_justP, self).__init__()
        self.pixel_attention = PixelAttention(input_channels)

        # 1x1 Conv to match teacher feature channels
        self.feature_projection = nn.Conv2d(input_channels, teacher_channels, kernel_size=1)

        self.conv = nn.Conv2d(input_channels + teacher_channels, teacher_channels, kernel_size=1)
        # Channel Attention for the concatenated features
        # self.channel_attention = ChannelAttention(teacher_channels)



    def forward(self, x):
        # Pixel attention path
        pixel_attended = self.pixel_attention(x)  # (B, input_channels, H, W)

        # 1x1 conv to match teacher channel
        projected = self.feature_projection(x)  # (B, teacher_channels, H, W)

        # Concat along channel dimension
        fused = torch.cat([pixel_attended, projected], dim=1)  # (B, teacher_channels*2, H, W)
        fused = self.conv(fused)
        # Channel attention applied to concatenated feature
        # output = self.channel_attention(fused)

        return fused
    


class PCatt_justC(nn.Module):
    def __init__(self, input_channels, teacher_channels):
        super(PCatt_justC, self).__init__()
        # self.pixel_attention = PixelAttention(input_channels)

        # 1x1 Conv to match teacher feature channels
        self.feature_projection = nn.Conv2d(input_channels, teacher_channels, kernel_size=1)

        self.conv = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1)
        # Channel Attention for the concatenated features
        self.channel_attention = ChannelAttention(teacher_channels)



    def forward(self, x):
        # Pixel attention path
        # pixel_attended = self.pixel_attention(x)  # (B, input_channels, H, W)

        # 1x1 conv to match teacher channel
        projected = self.feature_projection(x)  # (B, teacher_channels, H, W)

        # Concat along channel dimension
        # fused = torch.cat([pixel_attended, projected], dim=1)  # (B, teacher_channels*2, H, W)
        fused = self.conv(projected)
        # Channel attention applied to concatenated feature
        output = self.channel_attention(fused)

        return output


class SP(nn.Module):
    def __init__(self, student_channels, teacher_channels, reduction='batchmean'):
        super(SP, self).__init__()
        self.reduction = reduction
        self.channel_mapper = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)  # 채널수 맞춰주기


    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)  # 배치 차원 유지, 나머지 차원 펼치기
        return F.normalize(torch.matmul(z, torch.t(z)), dim=1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):

        student_outputs = self.channel_mapper(student_outputs)  # 채널 변환 적용
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2  # Frobenius norm

    def forward(self, teacher_outputs, student_outputs):
        batch_size = teacher_outputs.shape[0]
        spkd_loss = self.compute_spkd_loss(teacher_outputs, student_outputs)
        return spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss



###############################FGD######################
# https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py 기반 GPT 질문해서 얻은 코드
class FGD(nn.Module):
    """
    FGD (Focal and Global Distillation) Block
    논문: "Focal and Global Knowledge Distillation for Detectors"

    - Teacher Feature와 Student Feature를 입력으로 받아서 Student Feature를 Teacher의 채널 수로 변환 (1x1 Conv)
    - Teacher와 Student Feature 각각 FGD Block 적용
    - L2 Loss 계산하여 반환
    """

    def __init__(self, student_channels, teacher_channels, ratio=2):
        super(FGD, self).__init__()

        # Student Feature의 채널을 Teacher의 채널과 맞춰주기 위한 1x1 Conv
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None  # 채널이 같다면 변환 불필요

        # Context Modeling
        self.context_conv = nn.Conv2d(teacher_channels, 1, kernel_size=1)  # 공통 Context Modeling
        self.softmax = nn.Softmax(dim=2)  # 공간적 Attention Softmax

        # Transform: Channel-wise Attention (Teacher & Student 동일)
        self.transform = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // ratio, kernel_size=1),
            nn.LayerNorm([teacher_channels // ratio, 1, 1]),  # LayerNorm 적용
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels // ratio, teacher_channels, kernel_size=1)
        )

    def forward(self, student_feature, teacher_feature):
        """
        Args:
            student_feature: (B, C_s, H, W) - 학생 모델의 Feature Map
            teacher_feature: (B, C_t, H, W) - 교사 모델의 Feature Map
        Returns:
            loss: L2 Loss (MSE) between transformed student & teacher feature maps
        """
        B, C_t, H, W = teacher_feature.shape  # Teacher의 채널 수 C_t

        # Step 1: Student Feature를 Teacher의 채널과 맞춤
        if self.align is not None:
            student_feature = self.align(student_feature)  # (B, C_t, H, W)

        # Step 2: Teacher & Student Feature 각각 FGD Block 적용
        teacher_transformed = self.apply_fgd(teacher_feature)  # Teacher FGD 적용
        student_transformed = self.apply_fgd(student_feature)  # Student FGD 적용

        # Step 3: L2 Loss 계산
        loss = F.mse_loss(student_transformed, teacher_transformed)  # L2 Loss

        return loss

    def apply_fgd(self, x):
        """
        FGD Block을 적용하는 함수
        Args:
            x: (B, C, H, W) - Feature Map
        Returns:
            out: (B, C, H, W) - FGD 적용된 Feature Map
        """
        B, C, H, W = x.shape

        # Context Modeling
        context = self.context_conv(x)  # (B, 1, H, W)
        context = context.view(B, 1, H * W)  # (B, 1, HW)
        context = self.softmax(context)  # (B, 1, HW)

        # Apply context to input feature map
        x_flatten = x.view(B, C, H * W)  # (B, C, HW)
        context = torch.matmul(x_flatten, context.transpose(1, 2))  # (B, C, 1)
        context = context.view(B, C, 1, 1)  # (B, C, 1, 1)

        # Transform
        transform = self.transform(context)  # (B, C, 1, 1)

        # Add transformed context to original feature map
        out = x + transform  # (B, C, H, W)

        return out



############################CWD############################
# https://github.com/irfanICMLL/TorchDistiller/blob/main/SemSeg-distill/utils/criterion.py
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap
class CWD(nn.Module):

    def __init__(self, student_channels, teacher_channels, norm_type='channel', divergence='kl', temperature=4.0):

        super(CWD, self).__init__()

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 4.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

        self.conv = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def forward(self, preds_S, preds_T):
        preds_S=self.conv(preds_S)
        n, c, h, w = preds_S.shape
        # import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)

        # item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        # import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature ** 2)

##################################################################

############################AT######################################
# https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py
class AT(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super(AT, self).__init__()
        self.conv = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), 2)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)

        return am

    def forward(self, fm_s, fm_t):
        fm_s = self.conv(fm_s)
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss
####################################################################


##########################FAM#########################

#https://github.com/cuong-pv/FAM-KD/blob/main/distillers/FAM_KD.py

import torch.optim as optim
import math
import pdb
import torch.nn.init as init

import math

# source https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=4, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        # print(out_channels)
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x, y):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(y)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class FAM_module(nn.Module):
    def __init__(self, in_channels, out_channels, shapes):
        super(FAM_module, self).__init__()

        """
        feat_s_shape, feat_t_shape
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes
        #  print(self.shapes)
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        # self.out_channels = feat_t_shape[1]
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes, dtype=torch.cfloat))
        self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1)

        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        if isinstance(x, tuple):
            x, cuton = x
        else:
            cuton = 0.1
        batchsize = x.shape[0]
        x_ft = torch.fft.fft2(x, norm="ortho")
        #  print(x_ft.shape)
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        batch_fftshift = batch_fftshift2d(out_ft)

        # do the filter in here
        h, w = batch_fftshift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
        # the value of center pixel is zero.
        batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0
        # test with batch shift
        out_ft = batch_ifftshift2d(batch_fftshift)
        out_ft = torch.view_as_complex(out_ft)
        # Return to physical space
        out = torch.fft.ifft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho").real
        out2 = self.w0(x)
        return self.rate1 * out + self.rate2 * out2


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


def batch_fftshift2d(x):
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None)
                  if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None)
                  if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


class FAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=4, bias=False, shapes=256):
        """
        Fourier Attention Module (FAM)
        입력 Feature Map에 대해 AttentionConv을 거친 후 FAM_module을 적용하여 최종 출력을 반환하는 클래스.

        Args:
            in_channels (int): 입력 feature map의 채널 수
            out_channels (int): 출력 feature map의 채널 수
            kernel_size (int): AttentionConv에서 사용할 커널 크기
            stride (int): AttentionConv에서 사용할 stride
            padding (int): AttentionConv에서 사용할 padding
            groups (int): AttentionConv에서 사용할 그룹 수
            bias (bool): Convolution 연산에서 bias 사용 여부
            shapes (int): FAM_module에서 사용할 주파수 도메인 크기
        """
        super(FAM, self).__init__()

        # Cross Attention Module
        self.attention = AttentionConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

        # Fourier Attention Module
        self.fam_module = FAM_module(
            in_channels=out_channels,  # AttentionConv의 출력 채널이 FAM_module의 입력 채널이 됨
            out_channels=out_channels,  # FAM_module의 출력 채널 설정
            shapes=shapes
        )

    def forward(self, x, y):
        """
        Forward pass.
        1. AttentionConv을 통해 학생-교사 특징 간 Cross Attention 수행.
        2. FAM_module을 사용하여 Fourier 변환 기반 주파수 필터링 수행.

        Args:
            x (torch.Tensor): 입력 Feature Map (학생 네트워크의 특징)
            y (torch.Tensor): 보조 Feature Map (교사 네트워크의 특징)

        Returns:
            torch.Tensor: 최종 변환된 Feature Map
        """
        # Step 1: AttentionConv 적용
        attn_out = self.attention(x, y)  # [batch, out_channels, H, W]

        # Step 2: Fourier Attention Module 적용
        fam_out = self.fam_module(attn_out)  # [batch, out_channels, H, W]

        return fam_out

#######################################################################


###########################OFD###############################

#https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ofd.py

'''
Modified from https://github.com/clovaai/overhaul-distillation/blob/master/CIFAR-100/distiller.py
'''
class OFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OFD, self).__init__()
        self.connector = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        fm_s = self.connector(fm_s)

        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t)**2 * mask)

        return loss

    def get_margin(self, fm, eps=1e-6):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask

        margin = masked_fm.sum(dim=(0,2,3), keepdim=True) / (mask.sum(dim=(0,2,3), keepdim=True)+eps)

        return margin
    
#############################################################

###############################SRD##################################
# 주원이형한테 받음
class SRD(nn.Module):
    """
    Args:
        s_dim: the dimension of student's feature
        t_dim: the dimension of teacher's feature
    """
    def __init__(self, s_dim, t_dim, alpha=2):
        super(SRD, self).__init__()
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.alpha = alpha

        self.embed = nn.Linear(s_dim, t_dim).to('cuda')
        self.bn_s = torch.nn.BatchNorm1d(t_dim, eps=1e-5, affine=False).to('cuda')
        self.bn_t = torch.nn.BatchNorm1d(t_dim, eps=1e-5, affine=False).to('cuda')

    def forward_simple(self, z_s, z_t):
        f_s = z_s
        f_t = z_t

        # must reshape the transformer repr
        b = f_s.shape[0]
        f_s = f_s.transpose(1, 2).view(b, -1, 14, 14)
        f_s = self.embed(f_s)

        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)

        return F.mse_loss(f_s, f_t)

    def forward(self, z_s, z_t):
        b, c, h, w = z_s.shape
        b1, c1, h1, w1 = z_t.shape
        z_s = z_s.view(b, c, -1).mean(2)
        z_t = z_t.view(b1, c1, -1).mean(2)
        f_s = z_s
        f_t = z_t
        f_s = self.embed(f_s)
        n, d = f_s.shape

        f_s_norm = self.bn_s(f_s)
        f_t_norm = self.bn_t(f_t)

        c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
        c_diff = c_st - torch.ones_like(c_st)

        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(4.0)
        eps = 1e-5
        loss = torch.log(c_diff.sum() + eps)
        return loss

#############################################################