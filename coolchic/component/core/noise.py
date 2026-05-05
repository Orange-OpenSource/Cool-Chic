# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import math
from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor


@dataclass
class CommonGaussianNoiseGenerator:
    """Random generator of normally distributed values. Will always generate the same
    values in the same order.
    """

    seed: int = field(init=False, default=18101995)
    a: int = field(init=False, default=7**5)
    m: int = field(init=False, default=2**31 - 1)
    pi: float = field(init=False, default=3.14159265359)

    def grand(self):
        self.seed = (self.a * self.seed) % self.m
        u1 = self.seed / self.m
        self.seed = (self.a * self.seed) % self.m
        u2 = self.seed / self.m

        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * self.pi * u2)

    def sample(self, size: List[int]) -> Tensor:
        """Return a random tensor of a given size.

        Args:
            size (List[int]): Dimension of the random tensor.

        Returns:
            Tensor: The random tensor
        """

        numel = torch.prod(torch.tensor(size))
        if numel <= 0:
            raise ValueError(
                f"Random tensor must have at least 1 elements. Found size = {size}, numel={numel}."
            )

        res = [self.grand() for _ in range(numel)]
        res = torch.tensor(res, dtype=torch.float).view(size)
        return res
