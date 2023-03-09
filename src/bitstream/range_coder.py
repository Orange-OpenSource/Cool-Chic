# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import os
import torch
import constriction
import numpy as np
import torch

from typing import Optional, Tuple
from torch import Tensor

class RangeCoder:
    def __init__(self, n_ctx_rowcol: int, AC_MAX_VAL: int, Q_PROBA: int = 5):
        """Instantiate a range coder object.

        Args:
            n_ctx_rowcol (int): Only for the latent variables. Number of row & columns
                of context used by the ARM. This impact the wavefront coding order.
            AC_MAX_VAL (int): All symboled seen by the range coder must be in
                [AC_MAX_VAL, AC_MAX_VAL]
            Q_PROBA (int, optional): To avoid floating point drift, all values of
                mu and scale will be quantized with an accuracy of 1 / Q_PROBA.
                Defaults to 5.
        """
        # Higher: more accurate but less reliable probability model
        # Actual q_step is 1 / Q_PROBA
        self.Q_PROBA = Q_PROBA

        # Data are in [-AC_MAX_VAL, AC_MAX_VAL - 1]
        self.AC_MAX_VAL = AC_MAX_VAL

        self.alphabet = np.arange(-self.AC_MAX_VAL, self.AC_MAX_VAL + 1)
        self.model_family = constriction.stream.model.QuantizedLaplace(
            -self.AC_MAX_VAL, self.AC_MAX_VAL + 1
        )

        self.n_ctx_rowcol = n_ctx_rowcol

    def quantize_proba_parameters(self, x: Tensor) -> Tensor:
        """Apply a quantization to the input x to reduce floating point drift.

        Args:
            x (Tensor): The value to quantize

        Returns:
            Tensor: the quantize value
        """
        return torch.round(x * self.Q_PROBA) / self.Q_PROBA


    def encode(
        self,
        out_file: str,
        x: Tensor,
        mu: Tensor,
        scale: Tensor,
        CHW: Optional[Tuple[int, int, int]] = None,
    ):
        """Encode a 1D tensor x, using two 1D tensors mu and scale for the
        element-wise probability model of x.

        Args:
            x (Tensor): [B] tensor of values to be encoded
            mu (Tensor): [B] tensor describing the expectation of x
            scale (Tensor): [B] tensor with the standard deviations of x
        """
        # Re-arrange the data for wave front coding
        if CHW is not None:
            flat_coding_order = self.generate_coding_order(CHW, self.n_ctx_rowcol).flatten()
            # ! Stable is absolutely mandatory otherwise it won't work!
            index_coding_order = flat_coding_order.argsort(stable=True)

            # Reindex according to wavefront coding order
            x = x[index_coding_order]
            mu = mu[index_coding_order]
            scale = scale[index_coding_order]

        mu = self.quantize_proba_parameters(mu)
        scale = self.quantize_proba_parameters(scale)

        # proba = laplace_cdf(x + 0.5, mu, scale) - laplace_cdf(x - 0.5, mu, scale)
        # entropy_rate_bit = -torch.log2(torch.clamp_min(proba, min = 2 ** -16)).sum()

        x = x.numpy().astype(np.int32)
        mu = mu.numpy().astype(np.float64)
        scale = scale.numpy().astype(np.float64)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(x, self.model_family, mu, scale)
        # encoder.get_compressed().tofile(out_file)

        with open(out_file, 'wb') as f_out:
            f_out.write(encoder.get_compressed())

    def load_bitstream(self, in_file: str):
        """Load a bitstream file and instantiate a decoder attribute
        to the class.

        Args:
            in_file (str): Absolute path of the bitstream file
        """
        bitstream = np.fromfile(in_file, dtype=np.uint32)
        self.decoder = constriction.stream.queue.RangeDecoder(bitstream)


    def decode(self, mu: Tensor, scale: Tensor) -> Tensor:
        """Decode [B] parameters from a pre-loaded bitstream bile.

        Args:
            mu (Tensor): A 1d [B] tensor describing the expectation of
                the symbols to decode.
            scale (Tensor): A 1d [B] tensor describing the scale of
                the symbols to decode.

        Returns:
            Tensor: The [B] decoded symbols.
        """

        mu = self.quantize_proba_parameters(mu)
        scale = self.quantize_proba_parameters(scale)


        mu = mu.numpy().astype(np.float64)
        scale = scale.numpy().astype(np.float64)
        x = self.decoder.decode(self.model_family, mu, scale)

        x = torch.tensor(x).to(torch.float)

        return x


    def generate_coding_order(self, CHW: Tuple[int, int, int], n_ctx_rowcol: int) -> Tensor:
        """Generate a channel-independent wavefront coding order tensor. I.e. for each C
        channels and with N row & columns of context, return something like:
            0                 1                   2                  3
            N + 1             N + 2               N + 3              N + 4
            2 * (N + 1)       2 * (N + 1) + 1     2 * (N + 1) + 2    2 * (N + 1) + 3

        This tensor indicates the (de)coding order of the symbols in a 2D image.
        Channels are all (de)coded in parallel.

        Args:
            CHW (Tuple[int, int, int]): Channel, Height, Width.
            n_ctx_rowcol (int): Number of context rows & columns used.

        Returns:
            Tensor: A [C, H, W] tensor indicating the wavefront coding order.
        """
        C, H, W = CHW

        # Edge case: we have an image whose width is smaller than the number of
        # context row/column in this case: no wavefront
        if W < n_ctx_rowcol:
            coding_order = torch.arange(0, H * W).view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
            return coding_order

        # Generate something like
        # 0                 1                   2                  3
        # N + 1             N + 2               N + 3              N + 4
        # 2 * (N + 1)       2 * (N + 1) + 1     2 * (N + 1) + 2    2 * (N + 1) + 3
        # with N the number of row/column of context

        # Repeat the first line H times
        first_line = torch.arange(W).view(1, -1).repeat((H, 1))
        row_increment = torch.arange(H) * (n_ctx_rowcol + 1)
        row_increment = row_increment.view(-1, 1)

        # This is the spatial coding order i.e. a [H, W] tensor
        # since we code the C channels in parallel we just have to repeat
        # the spatial coding order along each channels
        coding_order = first_line + row_increment
        coding_order = coding_order.view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
        return coding_order
