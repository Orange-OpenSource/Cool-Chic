# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
from typing import Union
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from model_management.yuv import DictTensorYUV, write_yuv, yuv_dict_to_device
from models.cool_chic import CoolChicEncoder


@torch.no_grad()
def save_decoded_image(model: CoolChicEncoder, path_save: str):
    """Save the image decoded by a model into str.

    Args:
        model (CoolChicEncoder): A trained CoolChicEncoder
        path_save (str): Where to save the decoded image
    """
    decoded_image = model.forward().get('x_hat')
    save_png_or_yuv(decoded_image, model.param.img_type, path_save)


def save_png_or_yuv(data: Union[Tensor, DictTensorYUV], img_type: str, path_save: str):
    """Save a data to path_save with different methods according to img_type.

    Args:
        data (Union[Tensor, DictTensorYUV]): Data to be saved (4D tensor or dict of 4D tensors)
        img_type (str): Either 'rgb444' or 'yuv420_8b' or 'yuv420_10b
        path_save (str): Where to save the data.
    """
    if img_type == 'rgb444':
        data = data.cpu()
        to_pil_image(data.squeeze(0)).save(path_save)
    elif 'yuv420' in img_type:
        data = yuv_dict_to_device(data, 'cpu')
        write_yuv(data, path_save, bitdepth=10 if '10b' in img_type else 8)
