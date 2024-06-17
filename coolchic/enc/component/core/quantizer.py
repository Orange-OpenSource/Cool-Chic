# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import math
from typing import Literal, Optional
import typing

import torch
from torch import Tensor


@torch.jit.script
def softround(x: Tensor, t: float) -> Tensor:
    """Perform the softround function as introduced in section 4.1 of the paper
    `Universally Quantized Neural Compression, Agustsson & Theis
    <https://arxiv.org/pdf/2006.09952.pdf>`_, defined as follows:

    .. math::

        \\mathrm{softround}(x, t) = \\lfloor x \\rfloor +
        \\frac{\mathrm{tanh}(\\frac{\\Delta}{t})}{2\\ \\mathrm{\\mathrm{tanh}(\\frac{1}{2t})}}
        + \\frac{1}{2}, \\text{ with } \\Delta = x - \\lfloor x \\rfloor - \\frac{1}{2}.

    Args:
        x: Input tensor to be quantized.
        t: Soft round temperature :math:`t`. Setting :math:`t = 0` corresponds
            to the actual quantization i.e. ``round(x)``. As :math:`t` grows
            bigger, the function approaches identity i.e. :math:`\\lim_{t
            \\rightarrow \\infty} \\mathrm{softround}(x, t) = x`. In practice
            :math:`t \geq 1` is already quite close to identity.


    Returns:
        Soft-rounded tensor
    """
    floor_x = torch.floor(x)
    delta = x - floor_x - 0.5
    return floor_x + 0.5 * torch.tanh(delta / t) / math.tanh(1 / (2 * t)) + 0.5


@torch.jit.script
def generate_kumaraswamy_noise(
    uniform_noise: Tensor, kumaraswamy_param: float
) -> Tensor:
    """
    Reparameterize a random variable ``uniform_noise`` following a uniform
    distribution :math:`\\mathcal{U}(0, 1)` to a random
    variable following a `kumaraswamy distribution
    <https://en.wikipedia.org/wiki/Kumaraswamy_distribution>`_ as proposed in
    the paper `C3: High-performance and low-complexity neural compression from a
    single image or video, Kim et al. <https://arxiv.org/abs/2312.02753>`_

    The kumaraswamy distribution is defined on the interval :math:`(0, 1)` with
    the following PDF:

    .. math::

        f(x;a,b) = 1 - (1 - x^a)^b

    Here, it is only parameterized through a single parameter
    ``kumaraswamy_param`` corresponding to :math:`a` in the above equation. The
    second parameter :math:`b` is set to as a function of :math:`a` so that the
    mode of the distribution is always :math:`\\frac{1}{2}`. Setting :math:`a=1`
    gives the uniform distribution :math:`\\mathcal{U}(0, 1)`. Increasing the
    value of :math:`a` gives more "pointy" distribution.

    The resulting kumaraswamy noise is shifted so that it lies in
    :math:`(-\\frac{1}{2}, \\frac{1}{2})`.

    Args:
        uniform_noise: A uniform noise in :math:`[0, 1]` with any size.
        kumaraswamy_param: Parameter :math:`a` of a Kumaraswamy
            distribution. Set it to 1 for a uniform noise.

    Returns:
        A kumaraswamy noise with identical dim to ``uniform_noise`` in
        :math:`[-\\frac{1}{2}, \\frac{1}{2}]`.
    """
    # This relation between a and b allows to always have a mode of 0.5
    a = kumaraswamy_param
    b = (2**a * (a - 1) + 1) / a

    # Use the inverse of the repartition function to sample a kumaraswamy noise in [0., 1.]
    # Shift the noise to have it in [-0.5, 0.5]
    kumaraswamy_noise = (1 - (1 - uniform_noise) ** (1 / b)) ** (1 / a) - 0.5

    return kumaraswamy_noise


POSSIBLE_QUANTIZATION_NOISE_TYPE = Literal["kumaraswamy", "gaussian", "none"]
POSSIBLE_QUANTIZER_TYPE = Literal["softround_alone", "softround", "hardround", "ste", "none"]


def quantize(
    x: Tensor,
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
    soft_round_temperature: Optional[float] = 0.3,
    noise_parameter: Optional[float] = 1.0,
) -> Tensor:
    """Quantize an input :math:`x` to an output :math:`y` simulating the
    quantization. There is different mode possibles, described by
    ``quantizer_type``:

    - ``none``: :math:`y = x + n` with :math:`n` a random noise (more details
      below)

    - ``softround_alone``: :math:`y = \\mathrm{softround}(x, t)` with :math:`t`
      the ``soft_round_temperature``.

    - ``softround``: :math:`y = \\mathrm{softround}(\\mathrm{softround}(x, t) +
      n, t)` with :math:`t` the ``soft_round_temperature`` and :math:`n` a
      random noise (more details below)

    - ``hardround``: :math:`y = \\mathrm{round}(x)`

    - ``ste``: :math:`y = \\mathrm{round}(x)` (backward done through softround)

    The noise is parameterized by ``quantizer_noise_type`` and
    ``noise_parameter``. This last parameter has a different role for the
    different noise type:

    - ``gaussian``: ``noise_parameter`` is the standard deviation of the
      gaussian distribution

    - ``kumaraswamy``: ``noise_parameter`` corresponds to the :math:`a`
      parameter of the kumaraswamy distribution. 1 means uniform distribution
      and increasing it leads to more more and more probability of being into
      the center.

    Softround is parameterized by ``soft_round_temperature`` denoted as
    :math:`t`. Setting :math:`t = 0` corresponds to the actual quantization i.e.
    ``round(x)``. As :math:`t` grows bigger, the function approaches identity
    i.e. :math:`\\lim_{t \\rightarrow \\infty} \\mathrm{softround}(x, t) = x`.
    In practice :math:`t \geq 1` is already quite close to identity.,

    .. note::

        Why do we apply twice the softround when ``quantizer_type`` is
        ``softround``? It follows the operations described in `C3:
        High-performance and low-complexity neural compression from a single
        image or video, Kim et al. <https://arxiv.org/abs/2312.02753>`_ i.e.

        1. Use a soft round function instead of the non-differentiable round
           function
        2. Add a random noise to prevent the network from learning the inverse
           softround function
        3. Re-apply the soft round function as advocated in `Universally
           Quantized Neural Compression, Agustsson & Theis
           <https://arxiv.org/pdf/2006.09952.pdf>`_


    Args:
        x: Tensor to be quantized.
        quantizer_noise_type: noise type. Defaults to ``"kumaraswamy"``.
        quantizer_type: quantizer type. Defaults to ``"softround"``.
        soft_round_temperature: Soft round temperature. This is used for
            softround modes as well as the ste mode to simulate the derivative in
            the backward. Defaults to 0.3.
        noise_parameter: noise distribution parameter. Defaults to 1.0.

    Returns:
        Quantized tensor
    """
    # ----- Check user input
    # TODO: How long is it to do such assert?
    assert quantizer_noise_type in typing.get_args(POSSIBLE_QUANTIZATION_NOISE_TYPE), (
        f"quantizer_noise_type must be in {POSSIBLE_QUANTIZATION_NOISE_TYPE}"
        f" found {quantizer_noise_type}"
    )

    assert quantizer_type in typing.get_args(POSSIBLE_QUANTIZER_TYPE), (
        f"quantizer_type must be in {POSSIBLE_QUANTIZER_TYPE}" f"found {quantizer_type}"
    )

    # If we use only the softround **alone**, or hardround we do not need
    # any noise addition. Otherwise, we need a type of noise, i.e. either
    # kumaraswamy or gaussian noise.
    if quantizer_type in ["softround_alone", "hardround", "ste", "none"]:
        if quantizer_noise_type != "none":
            s = (
                f"Using quantizer type {quantizer_type} does not require"
                "to have any random noise.\nSwitching the "
                f"quantizer_noise_type from {quantizer_noise_type} to none."
            )
            print(s)
        quantizer_noise_type = "none"
    else:
        assert quantizer_noise_type != "none", (
            "Using quantizer_noise_type = 'none' is only possible with "
            "quantizer_type = 'softround_alone', 'ste' or 'hardround'.\nTrying"
            f" to use {quantizer_type} which do require some kind of random"
            "noise such as 'gaussian' or 'kumaraswamy'."
        )

    # ------- Actually quantize
    match quantizer_noise_type:
        case "none":
            pass
        case "gaussian":
            noise = torch.randn_like(x, requires_grad=False) * noise_parameter
        case "kumaraswamy":
            noise = generate_kumaraswamy_noise(
                torch.rand_like(x, requires_grad=False), noise_parameter
            )
        case _:
            print(f"Unknown quantizer_noise_type {quantizer_noise_type}")

    match quantizer_type:
        case "none":
            return x + noise
        case "softround_alone":
            return softround(x, soft_round_temperature)
        case "softround":
            return softround(
                softround(x, soft_round_temperature) + noise,
                soft_round_temperature,
            )
        case "ste":
            # From the forward point of view (i.e. entering into the torch.no_grad()), we have
            # y = softround(x) - softround(x) + round(x) = round(x). From the backward point of view
            # we have y = softround(x) meaning that dy / dx = d softround(x) / dx.
            # TODO: check whether it works?
            y = softround(x, soft_round_temperature)
            with torch.no_grad():
                y = y - softround(x, soft_round_temperature) + torch.round(x)
            return y
        case "hardround":
            return torch.round(x)
        case _:
            print(f"Unknown quantizer_type {quantizer_type}")
