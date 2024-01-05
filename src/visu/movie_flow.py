# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import numpy as np

def channel_last_to_channel_first(x):
    """Change a [H, W, C] tensor into a [C, H, W] tensor

    Args:
        x (tensor): channel last [H, W, C] tensor

    Returns:
        tensor: channel first [C, H, W] tensor
    """
    h, w, c = x.size()
    img_yuv_channel_first = torch.empty((c, h, w))
    for i in range(c):
        img_yuv_channel_first[i, :, :] = x[:, :, i]
    return img_yuv_channel_first

def _color_wheel():
    # Original inspiration: http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # RGB

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    # YG
    colorwheel[col: YG + col, 0] = 255 - \
        np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col: YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col: GC + col, 1] = 255
    colorwheel[col: GC + col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC

    # CB
    colorwheel[col: CB + col, 1] = 255 - \
        np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col: CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col: BM + col, 2] = 255
    colorwheel[col: BM + col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM

    # MR
    colorwheel[col: MR + col, 2] = 255 - \
        np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col: MR + col, 0] = 255

    return colorwheel


def _compute_color(u, v):
    colorwheel = _color_wheel()
    idxNans = np.where(np.logical_or(
        np.isnan(u),
        np.isnan(v)
    ))
    u[idxNans] = 0
    v[idxNans] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(np.multiply(u, u) + np.multiply(v, v))

    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1)
    k0 = fk.astype(np.uint8)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]


    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1-f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        col[~idx] *= 0.5
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)  # RGB
        # img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8) # BGR

    return img.astype(np.uint8)


def _normalize_flow(flow, max_val=-1):
    """[summary]
    * <max_val>
    ?       If <max_val> = -1: the representation is automatically scaled to
    ?   be adapted to the flow. That means that the most saturated color is
    ?   for the biggest flow value.
    ?       If <max_val> = X: The most vivid color of the flow representation
    ?   corresponds to the <max_val> value. Useful when generating flow videos.
    """
    UNKNOWN_FLOW_THRESH = 1e9
    # UNKNOWN_FLOW = 1e10

    height, width, nBands = flow.shape
    if not nBands == 2:
        raise AssertionError("Image must have two bands. [{h},{w},{nb}] shape given instead".format(
            h=height, w=width, nb=nBands))

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Fix unknown flow
    idxUnknown = np.where(np.logical_or(
        abs(u) > UNKNOWN_FLOW_THRESH,
        abs(v) > UNKNOWN_FLOW_THRESH
    ))
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max([-999, np.max(u)])
    maxv = max([-999, np.max(v)])
    minu = max([999, np.min(u)])
    minv = max([999, np.min(v)])

    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))

    # For the maximal radius to correspond to a (max_val, max_val) flow
    # This corresponds to the norm of a (max_val, max_val) flow
    if max_val != -1:
        maxrad = np.sqrt(2 * max_val ** 2)
    else:
        # Adaptive scaling, legacy code
        maxrad = max([-1, np.max(rad)])

    # if flags['debug']:
    #     print("Max Flow : {maxrad:.4f}. Flow Range [u, v] -> [{minu:.3f}:{maxu:.3f}, {minv:.3f}:{maxv:.3f}] ".format(
    #         minu=minu, minv=minv, maxu=maxu, maxv=maxv, maxrad=maxrad
    #     ))

    eps = np.finfo(np.float32).eps
    u = u/(maxrad + eps)
    v = v/(maxrad + eps)

    return u, v


def _flow2color(flow, max_val=-1):
    u, v = _normalize_flow(flow, max_val=max_val)
    img = _compute_color(u, v)
    return img


def flow_to_img(flow: torch.tensor, max_flow: int = 30) -> torch.tensor:
    """Transform one motion map flow to one image

    Args:
        flow (torch.tensor): [1, 2, H, W] tensor with the optical flow
        max_flow (int): Maximum color intensity corresponds to this motion

    Return:
        torch.tensor: ?
    """

    # Change [B, 2, H, W] to [2, H, W]
    flow = flow.squeeze(0).detach().cpu()
    # Change [2, H, W] to [H, W, 2]
    flow = torch.cat((flow[0, :, :].unsqueeze(2), flow[1, :, :].unsqueeze(2)), dim=2)
    flow = flow.numpy()

    img = _flow2color(flow, max_val=max_flow)
    img_yuv = torch.from_numpy(img)
    return img_yuv
