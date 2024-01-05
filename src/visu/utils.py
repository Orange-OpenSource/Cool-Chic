# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import torch
import io
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from typing import Dict, List, Literal, Union
from einops import rearrange
from matplotlib.colors import Normalize

from encoding_management.coding_structure import FrameData
from utils.yuv import rgb2yuv, write_yuv
from visu.movie_flow import flow_to_img

class MidpointNormalize(Normalize):
    """
    Force the center of the colormap (i.e. mid point) to be 0.
    Thus white always corresponds to 0
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class AlphaBetaNormalize(Normalize):
    """
    Stretch the color map around 0.5 to accomodate the alpha beta, visualisation.
    White range is bigger when scaling factor increases.
    MidpointNormalize has scaling factor=0.5
    """
    def __init__(self, scaling_param=0.):
        self.scaling_param = scaling_param
        Normalize.__init__(self, 0., 1., None)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        sc = self.scaling_param
        x, y = [0., 0.5 * (1 - sc), 0.5 * (1 + sc), 1.], [0, 0.45, 0.55, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def channel_last_to_channel_first(x: Tensor) -> Tensor:
    """Change a [H, W, C] tensor into a [C, H, W] tensor

    Args:
        x (Tensor): channel last [H, W, C] tensor

    Returns:
        Tensor: channel first [C, H, W] tensor
    """
    h, w, c = x.size()
    img_yuv_channel_first = torch.empty((c, h, w))
    for i in range(c):
        img_yuv_channel_first[i, :, :] = x[:, :, i]
    return img_yuv_channel_first


def matplotlib_colormap_to_img(
    data: Tensor,
    cmap_type: str,
    cmap_norm: str
    ) -> Dict[str, Tensor]:
    """Plot a [H, W] tensor data as a color map. Then extract the RGB values
    of this color map and convert them to an image both in YUV and RGB domain.
    As such it returns a dictionary with two [3, H, W] tensors in [0., 1.], ready to be
    saved as a .yuv file or as a png.

    Args:
        data (Tensor): a [H, W] tensor representing some data that we want
            to visualize as a feature map.
        cmap_param (dict): some parameters for matplotlib

    Returns:
        Dict[str, Tensor]: Two [1 3, H, W] tensors describing the heatmap as a YUV image and a RGB image
    """
    # A bit hackish, but it assures that we have one pixel latent = one pixel on screen
    # From https://stackoverflow.com/questions/8056458/display-image-with-a-zoom-1-with-matplotlib-imshow-how-to

    DPI = 80
    MARGIN = 0.
    # Make a figure big enough to accommodate an axis of W pixels by H pixels
    # as well as the ticklabels, etc...
    figsize = (1 + MARGIN) * data.size()[1] / DPI, (1 + MARGIN) * data.size()[0] / DPI
    plt.figure(figsize=figsize, dpi=DPI)

    # No margin at all
    ax = plt.gca()
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # Create heatmap and save it into the BytesIO buffer.
    ax.imshow( data, cmap=cmap_type, norm=cmap_norm, interpolation='none')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)

    # Then, open it with PIL Image to extract the RGB values
    im = Image.open(img_buf)
    plt.close()

    # Ignore alpha channel
    heatmap_rgb = torch.from_numpy(np.copy(np.asarray(im)[:, :, :-1]))
    # Change [H, W, C] -> [C, H, W]
    heatmap_rgb = channel_last_to_channel_first(heatmap_rgb)
    # Clamp all values in [0, 255]
    heatmap_rgb = torch.clamp(heatmap_rgb, 0., 255.)
    heatmap_rgb = rearrange(heatmap_rgb, 'c h w -> 1 c h w', c=3)
    # Convert RGB to YUV
    heatmap_yuv = rgb2yuv(heatmap_rgb)
    return {'yuv': heatmap_yuv / 255., 'rgb': heatmap_rgb / 255.}


def save_visualisation(
    x: Union[Tensor, FrameData, List[Tensor]],
    path_save: str,
    mode: Literal['image', 'flow', 'feature', 'alpha', 'beta', 'rate', 'sigma'] = 'image',
    format: Literal['yuv', 'png'] = 'yuv',
):

    if mode == 'image':
        assert isinstance(x, FrameData), f'Using save_visualisation() with ' \
            f'mode="image" requires x to be a FrameData!'

        if format == 'yuv':
            # Save the image directly
            write_yuv(x, path_save, norm=True)
        elif format == 'png':
            to_pil_image(rearrange(x.data.data, '1 c h w -> c h w', c=3)).save(path_save + '.png')

    elif mode == 'flow':
        # [B, C, H, W]
        rgb_data = rearrange(
            channel_last_to_channel_first(flow_to_img(x, max_flow=15)),
            'c h w -> 1 c h w'
        )

        if format == 'yuv':
            write_yuv(
                FrameData(bitdepth=8, frame_data_type='yuv444', data=rgb2yuv(rgb_data)),
                filename=path_save,
                norm=False
            )
        elif format == 'png':
            to_pil_image(rearrange(rgb_data, '1 c h w -> c h w', c=3)).save(path_save + '.png')

    # In this else: only things plotted as a matplotlib colormap
    else:
        # If the data to plot is a list of tensors, we upsample them to the
        # biggest resolution (supposed to be the one of the first element in the list.)
        # And concatenate them alongside the channel axis
        if isinstance(x, List):
            x = torch.cat(
                [
                    F.interpolate(cur_latent, size=x[0].size()[-2:], mode='nearest')
                    for cur_latent in x
                    if cur_latent.size()[1] > 0      # Ignore feature map with no channel...
                ], dim=1
            )

        # Even if we had a list earlier we now have a tensor. Concatenate
        # the different features spatially
        x = rearrange(x, '1 c h w -> 1 1 h (c w)')

        if mode == 'feature':
            heatmap_as_image = matplotlib_colormap_to_img(
                rearrange(x.detach().cpu(), '1 1 h w -> h w'),
                cmap_type='seismic',
                cmap_norm=MidpointNormalize(midpoint=0, vmax=x.abs().max(), vmin=-x.abs().max())
            )

        elif mode == 'alpha':
            # 1 - alpha is clearer to see than plain alpha
            heatmap_as_image = matplotlib_colormap_to_img(
                rearrange(1 - x.detach().cpu(), '1 1 h w -> h w'),
                cmap_type='inferno',
                cmap_norm=MidpointNormalize(midpoint=0.5, vmax=1.0, vmin=0)
            )

        elif mode == 'beta':
            heatmap_as_image = matplotlib_colormap_to_img(
                rearrange(x.detach().cpu(), '1 1 h w -> h w'),
                cmap_type='seismic',
                cmap_norm=AlphaBetaNormalize(scaling_param=0.3)
            )

        elif mode == 'rate':
            heatmap_as_image = matplotlib_colormap_to_img(
                rearrange(x.detach().cpu(), '1 1 h w -> h w'),
                cmap_type='inferno',
                cmap_norm=MidpointNormalize(midpoint=1.0, vmax=2.0, vmin=0)
            )

        # Save the colormap either as a YUV file or a PNG file
        if format == 'yuv':
            write_yuv(
                FrameData(bitdepth=8, frame_data_type='yuv444', data=heatmap_as_image.get('yuv')),
                filename=path_save,
                norm=True   # heatmap_as_img data is in [0., 1]
            )
        elif format == 'png':
            to_pil_image(rearrange(heatmap_as_image.get('rgb'), '1 c h w -> c h w', c=3)).save(path_save + '.png')
