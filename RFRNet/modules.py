import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeConsistentAttention(nn.Module):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(KnowledgeConsistentAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = nn.Parameter(torch.ones(1))

    def forward(self, foreground, masks):
        bz, nc, h, w = foreground.size()
        if masks.size(3) != foreground.size(3):
            masks = F.interpolate(masks, foreground.size()[2:])
        background = foreground.clone()
        background = background
        conv_kernels_all = background.view(bz, nc, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)
        output_tensor = []
        att_score = []
        for i in range(bz):
            feature_map = foreground[i:i + 1]
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv2d(feature_map, conv_kernels, padding=self.patch_size // 2)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.avg_pool2d(conv_result, 3, 1, padding=1) * 9
            attention_scores = F.softmax(conv_result, dim=1)
            if self.att_scores_prev is not None:
                attention_scores = (self.att_scores_prev[i:i + 1] * self.masks_prev[i:i + 1] + attention_scores * (
                            torch.abs(self.ratio) + 1e-7)) / (self.masks_prev[i:i + 1] + (torch.abs(self.ratio) + 1e-7))
            att_score.append(attention_scores)
            feature_map = F.conv_transpose2d(attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2)
            final_output = feature_map
            output_tensor.append(final_output)
        self.att_scores_prev = torch.cat(att_score, dim=0).view(bz, h * w, h, w)
        self.masks_prev = masks.view(bz, 1, h, w)
        return torch.cat(output_tensor, dim=0)


class AttentionModule(nn.Module):

    def __init__(self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1]):
        assert isinstance(patch_size_list,
                          list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(
            stride_list), "the input_lists should have same lengths"
        super(AttentionModule, self).__init__()

        self.att = KnowledgeConsistentAttention(patch_size_list[0], propagate_size_list[0], stride_list[0])
        self.num_of_modules = len(patch_size_list)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground, mask):
        outputs = self.att(foreground, mask)
        outputs = torch.cat([outputs, foreground], dim=1)
        outputs = self.combiner(outputs)
        return outputs


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################




class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask=None):

        if mask is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
            self.update_mask.to(input)
            self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output