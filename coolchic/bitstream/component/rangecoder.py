# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

from io import BytesIO

import constriction
import numpy as np
import torch
from torch import Tensor

from coolchic.bitstream.component.constants import (
    AC_MAX_VAL,
    N_POSSIBLE_MU,
    N_POSSIBLE_SCALE,
    PATH_MU_SCALE_TABLE,
)


class RangeCoder:
    def __init__(self):
        # Data are in [-AC_MAX_VAL, AC_MAX_VAL - 1]
        self.AC_MAX_VAL = AC_MAX_VAL

        self.model_family = constriction.stream.model.QuantizedLaplace(
            -self.AC_MAX_VAL, self.AC_MAX_VAL - 1
        )

        self.encoder = constriction.stream.queue.RangeEncoder()

        mu_scale_table = np.load(PATH_MU_SCALE_TABLE).astype(np.float32)
        self.all_mu = mu_scale_table[:N_POSSIBLE_MU]
        self.all_scale = mu_scale_table[N_POSSIBLE_MU:]

        if self.all_scale.size != N_POSSIBLE_SCALE:
            raise ValueError(
                f"The number of scales loaded from {PATH_MU_SCALE_TABLE} should be "
                f"{N_POSSIBLE_SCALE}. Found {self.all_scale.size}"
            )

    def encode(
        self,
        x: Tensor,
        idx_mu_scale: Tensor,
    ) -> bytes:
        """Encode a 1D tensor x, using two 1D tensors mu and scale for the
        element-wise probability model of x.

        Args:
            x (Tensor): [B] tensor of values to be encoded
            idx_mu_scale (Tensor): [B, 2] tensor describing the expectation and stdev of x
        """

        x = x.numpy().astype(np.int32)
        mu = self.all_mu.take(idx_mu_scale[:, 0], mode="clip")
        scale = self.all_scale.take(idx_mu_scale[:, 1], mode="clip")
        self.encoder.encode(x, self.model_family, mu, scale)

        # with open("tmp.bin", "ab") as f_out:
        #     f_out.write(encoder.get_compressed())
        # print(os.path.getsize("tmp.bin"))

        # encoder = constriction.stream.stack.AnsCoder()
        # encoder.encode_reverse(x, self.model_family, mu, scale)

    def reset_encoder(self) -> None:
        self.encoder = constriction.stream.queue.RangeEncoder()

    def get_bitstream_bytes(self) -> bytes:
        # Simulate writing to a binary file.
        output = BytesIO(self.encoder.get_compressed())
        bytes_to_write = output.getvalue()
        return bytes_to_write

    def load_bitstream(self, raw_bytes: bytes):
        bitstream = np.frombuffer(raw_bytes, dtype=np.uint32)
        self.decoder = constriction.stream.queue.RangeDecoder(bitstream)
        # self.decoder = constriction.stream.stack.AnsCoder(bitstream)

    # from line_profiler import profile
    # @profile
    def decode(self, idx_mu_scale: Tensor) -> Tensor:

        idx_mu_scale = idx_mu_scale.numpy()
        mu = self.all_mu.take(idx_mu_scale[:, 0], mode="clip")
        scale = self.all_scale.take(idx_mu_scale[:, 1], mode="clip")

        x = self.decoder.decode(self.model_family, mu, scale)
        return torch.from_numpy(x)
