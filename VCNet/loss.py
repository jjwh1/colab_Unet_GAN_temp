import torch
import numpy as np

from torch import nn
from torch.autograd import Variable
import torch.autograd as autograd

import functools
import torch
from torch import nn
from torch.nn import functional as F

from vgg import VGG19FeatLayer





cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples, masks=None):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates) if masks is None else D(interpolates, masks)
    fake = Variable(torch.ones_like(d_interpolates), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




class WeightedBCELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super(WeightedBCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, out, target, weights=None):
        out = out.clamp(self.epsilon, 1 - self.epsilon)
        if weights is not None:
            assert len(weights) == 2
            loss = weights[1] * (target * torch.log(out)) + weights[0] * ((1 - target) * torch.log(1 - out))
        else:
            loss = target * torch.log(out) + (1 - target) * torch.log(1 - out)
        return torch.neg(torch.mean(loss))





class SemanticConsistencyLoss(nn.Module):
    def __init__(self, content_layers=None):
        super(SemanticConsistencyLoss, self).__init__()
        self.feat_layer = VGG19FeatLayer()
        if content_layers is not None:
            self.feat_content_layers = content_layers
        else:
            self.feat_content_layers = {'relu3_2': 1.0}

    def _l1_loss(self, o, t):
        return torch.mean(torch.abs(o - t))

    def forward(self, out, target):
        out_vgg_feats = self.feat_layer(out)
        target_vgg_feats = self.feat_layer(target)
        content_loss_lst = [self.feat_content_layers[layer] * self._l1_loss(out_vgg_feats[layer], target_vgg_feats[layer])
                            for layer in self.feat_content_layers]
        content_loss = functools.reduce(lambda x, y: x + y, content_loss_lst)
        return content_loss


class IDMRFLoss(nn.Module):
    def __init__(self):
        super(IDMRFLoss, self).__init__()
        self.feat_layer = VGG19FeatLayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, feature_maps):
        return feature_maps / torch.sum(feature_maps, dim=1, keepdim=True)

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        return cdist / (torch.min(cdist, dim=1, keepdim=True)[0] + epsilon)

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def _mrf_loss(self, o, t):
        o_feats = o - torch.mean(t, 1, keepdim=True)
        t_feats = t - torch.mean(t, 1, keepdim=True)
        o_normalized = o_feats / torch.norm(o_feats, p=2, dim=1, keepdim=True)
        t_normalized = t_feats / torch.norm(t_feats, p=2, dim=1, keepdim=True)

        cosine_dist_l = []
        b_size = t.size(0)

        for i in range(b_size):
            t_feat_i = t_normalized[i:i + 1, :, :, :]
            o_feat_i = o_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(t_feat_i)

            cosine_dist_i = F.conv2d(o_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)

        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, out, target):
        out_vgg_feats = self.feat_layer(out)
        target_vgg_feats = self.feat_layer(target)

        style_loss_list = [self.feat_style_layers[layer] * self._mrf_loss(out_vgg_feats[layer], target_vgg_feats[layer])
                           for layer in self.feat_style_layers]

        content_loss_list = [self.feat_content_layers[layer] * self._mrf_loss(out_vgg_feats[layer], target_vgg_feats[layer])
                             for layer in self.feat_content_layers]

        return functools.reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style + \
            functools.reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content