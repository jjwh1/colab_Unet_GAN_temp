import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
import math
from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
import transformer
from utils import PatchPositionEmbeddingSine

# pip install scikit-image, pip install einops 필요 !!!!!!!!!!!!
# https://github.com/zhenglab/TransCNN-HAE/releases/tag/v1.0 참고


class TransCNN(nn.Module):
    def __init__(self):
        super(TransCNN, self).__init__()
        dim = 224

        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
            nn.Linear(4*4*3, dim)
        )
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=2, num_encoder_layers=9, dim_feedforward=dim*2, activation='gelu')
        self.cnn_dec = CNNDecoder(224, 3, 'ln', 'lrelu', 'reflect')
        
        b = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_pos = PatchPositionEmbeddingSine(ksize=4, stride=4)
        self.input_pos = input_pos.unsqueeze(0).repeat(b, 1, 1, 1).to(device)
        self.input_pos = self.input_pos.flatten(2).permute(2, 0, 1)

    def forward(self, inputs):
        patch_embedding = self.patch_to_embedding(inputs)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2), src_pos=self.input_pos)
        bs, L, C  = patch_embedding.size()
        content = content.permute(1,2,0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L)))
        output = self.cnn_dec(content)
        return output


class CNNDecoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, norm='bn', activ='relu', pad_type='reflect'):
        super(CNNDecoder, self).__init__()
        self.model = []
        dim = input_dim
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        dim //= 2
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.conv3 = Conv2dBlock(dim//2, output_dim, 5, 1, 2, norm='none', activation='tanh', pad_type=pad_type)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        output = self.conv3(x2)
        return output


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        # elif norm == 'adain_ori':
        #     self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        # elif norm == 'remove_render':
        #     self.norm = RemoveRender(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # # initialize convolution
        # if norm == 'sn':
        #     self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        # else:
        #     self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)


        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
