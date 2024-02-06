# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import constriction
import numpy as np
import torch

from typing import Optional, Tuple
from torch import Tensor

from utils.misc import MAX_ARM_MASK_SIZE, Q_PROBA_DEFAULT

class RangeCoder:
    def __init__(self, AC_MAX_VAL: int, Q_PROBA: int = Q_PROBA_DEFAULT):

        # Higher: more accurate but less reliable probability model
        # Actual q_step is 1 / Q_PROBA
        self.Q_PROBA = Q_PROBA

        # Data are in [-AC_MAX_VAL, AC_MAX_VAL - 1]
        self.AC_MAX_VAL = AC_MAX_VAL

        self.alphabet = np.arange(-self.AC_MAX_VAL, self.AC_MAX_VAL + 1)
        self.model_family = constriction.stream.model.QuantizedLaplace(
            -self.AC_MAX_VAL, self.AC_MAX_VAL + 1
        )

        # Every context is inscribed inside as mask of size MAX_ARM_MASK_SIZE
        # For a given mask size N (odd number e.g. 3, 5, 7), we have at most
        # (N * N - 1) / 2 context pixels in it.
        # Example, a 9x9 mask as below has 40 context pixel (indicated with 1s)
        # available to predict the pixel '*'
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 * 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        # Here n_ctx_rowcol = 4 rows and columns of context

        self.n_ctx_rowcol = int((MAX_ARM_MASK_SIZE - 1) / 2)

    def quantize_proba_parameters(self, x: Tensor) -> Tensor:
        """Apply a quantization to the input x to reduce floating point
        drift.

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
            flat_coding_order = self.generate_coding_order(CHW).flatten()
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
        bitstream = np.fromfile(in_file, dtype=np.uint32)
        self.decoder = constriction.stream.queue.RangeDecoder(bitstream)


    def decode(self, mu: Tensor, scale: Tensor) -> Tensor:

        mu = self.quantize_proba_parameters(mu)
        scale = self.quantize_proba_parameters(scale)

        mu = mu.numpy().astype(np.float64)
        scale = scale.numpy().astype(np.float64)
        x = self.decoder.decode(self.model_family, mu, scale)

        x = torch.tensor(x).to(torch.float)

        return x


    def generate_coding_order(self, CHW: Tuple[int, int, int], device: str = "cpu") -> Tensor:

        C, H, W = CHW

        # Edge case: we have an image whose width is smaller than the number of
        # context row/column in this case: no wavefront
        if W < self.n_ctx_rowcol:
            coding_order = torch.arange(0, H * W).view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
            return coding_order

        # Generate something like
        # 0                 1                   2                  3
        # N + 1             N + 2               N + 3              N + 4
        # 2 * (N + 1)       2 * (N + 1) + 1     2 * (N + 1) + 2    2 * (N + 1) + 3
        # with N the number of row/column of context

        # Repeat the first line H times
        first_line = torch.arange(W, device=device).view(1, -1).repeat((H, 1))
        row_increment = torch.arange(H, device=device) * (self.n_ctx_rowcol + 1)
        row_increment = row_increment.view(-1, 1)

        # This is the spatial coding order i.e. a [H, W] tensor
        # since we code the C channels in parallel we just have to repeat
        # the spatial coding order along each channels
        coding_order = first_line + row_increment
        coding_order = coding_order.view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
        return coding_order
