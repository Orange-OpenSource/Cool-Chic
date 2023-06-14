# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
from models.cool_chic import CoolChicEncoder
from models.synthesis import NoiseQuantizer, STEQuantizer


def save_model(model: CoolChicEncoder, path: str):
    """Save an entire model @ path

    Args:
        model (CoolChicEncoder): The model to save
        path (str): Where to save
    """
    # PyTorch does not like saving torch.autograd.function object
    # So we set them to None before saving them... we'll restore them afterwards
    model.noise_quantizer = None
    model.ste_quantizer = None
    torch.save(model, path)

    # Restore the noise_quantizer and ste_quantizer object
    model.noise_quantizer = NoiseQuantizer()
    model.ste_quantizer = STEQuantizer()
    STEQuantizer.ste_derivative = model.param.ste_derivative


def load_model(path: str) -> CoolChicEncoder:
    """Load a model from path

    Args:
        path (str): Path of the file where the model is saved

    Returns:
        CoolChicEncoder: The loaded module
    """
    model = torch.load(path, map_location='cpu')

    # Restore the noise_quantizer and ste_quantizer object as explained in
    # the save_model function
    model.noise_quantizer = NoiseQuantizer()
    model.ste_quantizer = STEQuantizer()
    STEQuantizer.ste_derivative = model.param.ste_derivative

    return model
