#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pdb import set_trace as stx

img2mse = lambda x, y, z : torch.mean((x - y) ** 2 * z)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# unit exposure loss
def point_constraint(model, gt, iteration):

    # if iteration == 600:
    #     stx()
    
    # ln_x = torch.zeros([3,1]).cuda()

    ln_x = -4.0 * torch.ones([3,1]).cuda()

    r_h = ln_x
    g_h = ln_x
    b_h = ln_x

    for i in range(len(model.tone_mapper_r)):
        if i == 0:
            r_l = model.tone_mapper_r[i](r_h)
        else:
            r_l = model.tone_mapper_r[i](r_l)

    for i in range(len(model.tone_mapper_g)):
        if i == 0:
            g_l = model.tone_mapper_g[i](g_h)
        else:
            g_l = model.tone_mapper_g[i](g_l)
    
    # stx()
    for i in range(len(model.tone_mapper_b)):
        if i == 0:
            b_l = model.tone_mapper_b[i](b_h) # model.tone_mapper_b[0](b_h)
        else:
            b_l = model.tone_mapper_b[i](b_l)

    # stx()

    rgb_l = torch.cat([r_l, g_l, b_l], -1)
    
    # stx()
    return img2mse(rgb_l, gt, 1)



def total_variation_loss(img):
    """
    计算图像的 Total Variation Loss。
    参数:
    - img: 要处理的图像，维度应为 (channels, height, width)
    
    返回:
    - total variation loss 的值
    """
    # 计算图像在水平方向上的像素差异
    horizontal_tv = torch.pow(img[:, :, :-1] - img[:, :, 1:], 2).sum()

    # 计算图像在垂直方向上的像素差异
    vertical_tv = torch.pow(img[:, :-1, :] - img[:, 1:, :], 2).sum()

    # 将两个方向的差异相加得到总的 TV loss
    tv_loss = horizontal_tv + vertical_tv
    
    return tv_loss

