# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import torch
import torch.nn.functional as F
from torch.nn import ReflectionPad2d
from math import exp

"""
PyTorch re-implementation of MS-SSIM from
https://github.com/jorge-pessoa/pytorch-msssim

Equivalent with Tensorflow MS-SSIM
"""


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2))
                          for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True,
         full=False, val_range=None):
    # Value range can be different from 255.
    # Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device, non_blocking=True)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel)\
        - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel)\
        - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel)\
        - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim_fn(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False, full=True):
    weights = torch.FloatTensor([
        0.0448, 0.2856, 0.3001, 0.2363, 0.1333
        ]).to(img1.device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(
            img1,
            img2,
            window_size=window_size,
            size_average=size_average,
            full=True,
            val_range=val_range
        )
        mssim.append(sim)
        mcs.append(cs)

        # Padding
        pix_pad_bottom = int((2 - (img1.size()[2] % 2)) % 2)
        pix_pad_right = int((2 - (img1.size()[3] % 2)) % 2)

        padding = ReflectionPad2d((0, pix_pad_right, 0, pix_pad_bottom))

        img1 = padding(img1)
        img2 = padding(img2)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models,
    # not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    # !
    # !
    # ! The important part is here: Eventhough
    # !     torch.prod(pow1[:-1] * pow2[-1]) (a)
    # ! is equivalent to
    # !     torch.prod(pow1[:-1]) * pow2[-1] (b)
    # ! It is not numerically equivalent. (b) must be used to be aligned
    # ! with clic
    output = torch.prod(pow1[:-1]) * pow2[-1]

    return output
