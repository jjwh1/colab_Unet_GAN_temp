import torch
from torch import nn
from torch.nn import functional as F



class SqEx(nn.Module):
    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class PCN(nn.Module):
    def __init__(self, n_features, epsilon=1e-5):
        super(PCN, self).__init__()
        self.epsilon = epsilon
        self.sqex = SqEx(n_features)

    def forward(self, x, m):
        _t = self._compute_T(x, m)
        _beta = self.sqex(x)
        context_feat = _beta * (_t * m) + (1. - _beta) * (x * m)
        preserved_feat = x * (1. - m)
        return context_feat + preserved_feat

    def _compute_T(self, x, m):
        X_p = x * m
        X_q = x * (1. - m)
        X_p_mean = self._compute_weighted_mean(X_p, m).unsqueeze(-1).unsqueeze(-1)
        X_p_std = self._compute_weighted_std(X_p, m).unsqueeze(-1).unsqueeze(-1)
        X_q_mean = self._compute_weighted_mean(X_q, m).unsqueeze(-1).unsqueeze(-1)
        X_q_std = self._compute_weighted_std(X_q, m).unsqueeze(-1).unsqueeze(-1)
        return ((X_p - X_p_mean) / X_p_std) * X_q_std + X_q_mean

    def _compute_weighted_mean(self, x, m):
        return torch.sum(x * m, dim=(2, 3)) / (torch.sum(m) + self.epsilon)

    def _compute_weighted_std(self, x, m):
        _mean = self._compute_weighted_mean(x, m).unsqueeze(-1).unsqueeze(-1)
        return torch.sqrt((torch.sum(torch.pow(x * m - _mean, 2), dim=(2, 3)) /
                          (torch.sum(m) + self.epsilon)) + self.epsilon)

class ResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(ResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.BatchNorm2d(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.conv2(out)
        # out = self.n2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.elu2(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.n1 = nn.BatchNorm2d(channels_out)
        self.elu1 = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.n1(out)
        out = self.elu1(out)
        return out


class PCBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(PCBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.pcn = PCN(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x, m):
        residual = x
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.conv2(out)
        _, _, h, w = out.size()
        out = self.pcn(out, F.interpolate(m, (h, w), mode="nearest"))
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.elu2(out)
        return out


if __name__ == '__main__':
    pcb = PCBlock(channels_in=3, channels_out=32, kernel_size=5, stride=1, padding=2)
    inp = torch.rand((4, 3, 256, 256))
    mask = torch.rand((4, 1, 256, 256))
    out = pcb(inp, mask)
    print(out.size())