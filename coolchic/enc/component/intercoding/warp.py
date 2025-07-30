# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from dataclasses import dataclass, field, fields
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

POSSIBLE_WARP_MODE = Literal[
    "torch_nearest", "torch_bilinear", "torch_bicubic", "bilinear", "bicubic", "sinc"
]


@dataclass
class WarpParameter:
    """Dataclass storing the parameters of the motion compensation warping"""

    filter_size: int

    # While we provide an explicit implementation for bilinear and
    # bicubic filtering, the actual PyTorch grid_sample implementation
    # is faster. Set this flag to False to use our explicit implementation
    # nevertheless.
    use_torch_if_available: bool = True

    # The actual mode is derived from the desired filter_size
    mode: Optional[POSSIBLE_WARP_MODE] = field(init=False, default=None)

    # At inference time, the flow is constrained to have only
    # <fractional_accuracy> possible values.
    fractional_accuracy: int = field(init=False, default=64)

    def __post_init__(self):
        assert self.filter_size % 2 == 0, (
            f"Warp filter size should be even. Found filter_size={self.filter_size}."
        )

        assert self.filter_size >= 2, (
            f"Warp filter size should be >= 2. Found filter_size={self.filter_size}."
        )

        match self.filter_size:
            case 2:
                self.mode = (
                    "torch_bilinear" if self.use_torch_if_available else "bilinear"
                )
            case 4:
                self.mode = (
                    "torch_bicubic" if self.use_torch_if_available else "bicubic"
                )
            case _:
                self.mode = "sinc"

    def pretty_string(self) -> str:
        """Return a pretty string presenting the WarpParameter.

        Returns:
            str: Pretty string ready to be printed.
        """
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = ""
        for k in fields(self):
            s += f"{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n"
        s += "\n"
        return s


class Warper(nn.Module):
    def __init__(self, param: WarpParameter, img_size: Tuple[int, int]):
        """Instantiate a warper module, parameterized by `param`.

        Args:
            param: Warping parameters (filter length, fractional accuracy).
            img_size: [Height, Width].
        """
        super().__init__()

        self.param = param
        self.filter_size = param.filter_size

        # Leverage grid sample for these modes
        self.native_pytorch_warp = self.param.mode in [
            "torch_nearest",
            "torch_bilinear",
            "torch_bicubic",
        ]

        if self.native_pytorch_warp:
            B = 1
            H, W = img_size

            tensor_hor = (
                torch.linspace(-1.0, 1.0, W, dtype=torch.float32)
                .view(1, 1, 1, W)
                .expand(B, -1, H, -1)
            )
            tensor_ver = (
                torch.linspace(-1.0, 1.0, H, dtype=torch.float32)
                .view(1, 1, H, 1)
                .expand(B, -1, -1, W)
            )
            self.register_buffer(
                "backward_grid",
                torch.cat([tensor_hor, tensor_ver], 1),
                persistent=False,
            )

        # Custom implementation of different interpolation filters, including
        # some already offered by PyTorch e.g. bilinear and bicubic
        else:
            # We always interpolate a point x in [0., 1.[ based on filter_size
            # neighbors. With filter_size = 4, we have
            # left_neighbor = -1 and right_neighbor = 2 e.g.
            #
            # -1    0    x    1    2 ==> x is a weighted sum of these 4 neighbors

            left_top_neighbor = -int(self.filter_size // 2) + 1
            right_bot_neighbor = int(self.filter_size // 2)

            grids = []
            # + 1 because right half is included
            for i in range(left_top_neighbor, right_bot_neighbor + 1):
                for j in range(left_top_neighbor, right_bot_neighbor + 1):
                    grid = self.coords_grid(*img_size)  # H, W
                    grid[:, 0] += j
                    grid[:, 1] += i
                    grids.append(grid)

            # register_buffer for automatic device management. We set persistent to false
            # to simply use the "automatically move to device" function, without
            # considering grids as a parameters (i.e. returned by self.parameters())
            #
            # self.grids dimension is [filter_size ** 2, 2, H, W]

            # grids is [filter_size ** 1, 2, H, W]. For each pixel in the HxW frame,
            # it stores the x and y indices of each of the filter_size ** 2 neighboring
            # values.
            # self.grids is something like, with filter_size = 4
            #   self.grids[:, :, i, j] = [
            #       [j - 1, i - 1],
            #       [j    , i - 1],
            #       [j + 1, i - 1],
            #       [j + 2, i - 1],
            #       [j - 1, i    ],
            #       [j    , i    ],
            #       [j + 1, i    ],
            #       [j + 2, i    ],
            #           ...
            #       [j + 2, i + 2],
            # ]
            self.register_buffer(
                "grids",
                torch.cat(grids, dim=0),
                persistent=False,
            )

            if self.param.mode == "sinc":
                # self.half_filter_size = int(self.filter_size // 2)

                # Corresponds to \kappa_i in eq. 6 in "Efficient Sub-pixel Motion
                # Compensation in Learned Video Codecs".
                self.register_buffer(
                    "relative_neighbor_idx",
                    # + 1 so that it is included
                    torch.arange(left_top_neighbor, right_bot_neighbor + 1).view(
                        1, -1, 1, 1
                    ),
                    persistent=False,
                )

            elif self.param.mode == "bicubic":
                # ! Exactly like pytorch bicubic grid sample
                a = -0.75
                bicubic_init = torch.tensor(
                    [
                        [0, a, -2 * a, a],
                        [1, 0, -(a + 3), a + 2],
                        [0, -a, (2 * a + 3), -(a + 2)],
                        [0, 0, a, -a],
                    ]
                )
                self.register_buffer("B", bicubic_init, persistent=False)

            elif self.param.mode == "bilinear":
                # ! Exactly like pytorch bilinear grid sample
                bilinear_init = torch.tensor(
                    [
                        [1.0, -1.0],
                        [0.0, 1.0],
                    ]
                )
                self.register_buffer("B", bilinear_init, persistent=False)

    def coords_grid(self, h: int, w: int) -> Tensor:
        """Return a [1, 2, H, W] tensor, where the 1st feature gives the
        column index and the 2nd the row index. For instance:

        .. code:

            coords_grid(3, 5) =
                    tensor([[[[0., 1., 2., 3., 4.],
                              [0., 1., 2., 3., 4.],
                              [0., 1., 2., 3., 4.]],

                             [[0., 0., 0., 0., 0.],
                              [1., 1., 1., 1., 1.],
                              [2., 2., 2., 2., 2.]]]])

        Args:
            h: height of the grid
            w: width of the grids

        Returns:
            Tensor: Tensor giving the column and row indices.
        """
        # [H, W]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

        # [1, 2, H, W]
        return torch.stack([x, y], dim=0).unsqueeze(0).float()

    def get_coeffs(self, s: Tensor) -> Tensor:
        """Generate interpolation coefficients from the fractional displacement
        s in [0, 1[.

        Args:
            s: Fractional displacement for each pixel, shape is [1, 1, H, W].

        Returns:
            Tensor: Corresponding interpolation coefficients for each pixel,
                shape is [1, filter_size, H, W].
        """

        if self.param.mode in ["sinc"]:
            s = torch.repeat_interleave(s, self.filter_size, dim=1)
            # Correspond to eq. 6 in "Efficient Sub-pixel Motion
            # Compensation in Learned Video Codecs".
            window = torch.cos(
                torch.pi * (s - self.relative_neighbor_idx) / self.filter_size
            )
            coeffs = window * torch.sinc(s - self.relative_neighbor_idx)
        elif self.param.mode in ["bilinear", "bicubic"]:
            # All these modes behave similarly: we derive the filter coeffs as:
            # coeffs = B @ t_exponents
            #
            # coeffs      --> the N taps of the filter
            # t_exponents --> a [t^0, .., t^deg] vector
            # B           --> a [N, deg + 1] matrix allowing to generate coeff
            #                 for any value of t in [0., 1.]
            #
            h, w = s.size()[-2:]
            s = rearrange(s, "1 1 h w -> (h w) 1")

            # From here s shape is [HW, max_deg] i.e. each row is s^0, .. , s^max_deg
            max_deg = self.B.size()[1]
            s_exponents = torch.cat(
                [torch.pow(s, exponent=expo) for expo in range(max_deg)], dim=1
            )
            coeffs = F.linear(s_exponents, self.B, bias=None)
            coeffs = rearrange(
                coeffs,
                "(h w) n_coef -> 1 n_coef h w",
                n_coef=self.filter_size,
                h=h,
                w=w,
            )

        return coeffs

    def interpolate_1d(self, neighbors: Tensor, fractional_flow: Tensor) -> Tensor:
        """Performs the interpolations of neighboring integer values to get the
        value located at fractional_flow.

        Args:
            neighbors: [B, filter_size, C, H, W]. We compute B x C x H x W
                interpolations in parallel. Each of them has filter_size neighbors.

            fractional_flow: [1, 1, H, W]. Fractional flow for each warping.
                All channels share the same fractional_flow, hence C=1 here.
                All batches share the same fractional_flow, hence B=1 here.
                This is used to shift 2d blocks of pixels with B=filter_size.

        Returns:
            Tensor: [B, C, H, W] the interpolated neighbors
        """
        coeffs = self.get_coeffs(fractional_flow)
        # Add a one-dimensional channel index in between filter_size and h w
        # this will be broadcasted to all the channels in neighbors
        coeffs = rearrange(coeffs, "b filter_size h w -> b filter_size 1 h w")

        res = torch.sum(neighbors * coeffs, dim=1)
        return res

    def forward(self, x: Tensor, flow: Tensor) -> Tensor:
        """Warp a [1, C, H, W] tensor using a [1, 2, H, W] optical flow. The
        optical flow is expressed in absolute pixel i.e. an horizontal motion of
        -3 means that the pixel output pixel at [i, j] is equal to x[i - 3, j].

        y = warp(x, flow) --> y[i, j] = x[i + vertical flow, j + horizontal flow]

        where vertical flow is flow[:, 0, :, :] (row-wise) and horizontal flow
        is flow[:, 1, :, :] (column wise).

        As such, the value in flow describes the motion from y (the warping
        result) to x (the reference.)

        Args:
            x: Tensor to be warped, shape is [1, C, H, W]
            flow: Motion to warp x: shape is [1, 2, H, W].

        Returns:
            Tensor: Warped tensor [1, C, H, W]
        """

        _, C, H, W = x.size()

        if self.training:
            q_flow = flow
        else:
            q_flow = (
                torch.round(flow * self.param.fractional_accuracy)
                / self.param.fractional_accuracy
            )

        if self.native_pytorch_warp:
            q_flow = torch.cat(
                [
                    q_flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                    q_flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
                ],
                dim=1,
            )

            grid = self.backward_grid + q_flow

            output = nn.functional.grid_sample(
                x,
                grid.permute(0, 2, 3, 1),
                mode=self.param.mode.replace("torch_", ""),
                padding_mode="border",
                align_corners=True,
            )

            return output

        else:
            # We first apply the integer part of the flow using simple re-indexing
            # i.e. grid_sample with mode="nearest".
            # Then we interpolate to get the fractional flow value.

            # No need to backward through that!
            rounded_flow = torch.floor(q_flow)
            fractional_flow = q_flow - rounded_flow

            # neighbors = self.grids + rounded_flow.expand(self.filter_size**2, 2, H, W)
            # grids is [filter_size ** 1, 2, H, W]. Rounded flow is [1, 2, H, W]
            # shift the position of all neighbors by the integer displacement
            neighbors = self.grids + rounded_flow

            # Transform absolute pixel values to relative values in [-1, 1]
            normalized_neighbors = torch.cat(
                [
                    # fmt: off
                    2 * torch.clamp(neighbors[:, 0:1, :, :], min=0, max=W - 1) / (W - 1)
                    - 1,
                    2 * torch.clamp(neighbors[:, 1:2, :, :], min=0, max=H - 1) / (H - 1)
                    - 1,
                    # fmt:on
                ],
                dim=1,
            )

            # warped_x_rounded_flow shape is [filter_size ** 2, C, H, W]
            # Each CxHxW has filter_size ** 2 neighboring values
            warped_x_rounded_flow = torch.nn.functional.grid_sample(
                x.expand(self.filter_size**2, C, H, W),
                normalized_neighbors.permute(0, 2, 3, 1),
                align_corners=True,
                mode="nearest",
                padding_mode="border",
            )

            # Split the filter_size ** 2 neighboring values into filter_size rows
            # and filter_size columns.
            stacked_lines = torch.stack(
                torch.split(warped_x_rounded_flow, self.filter_size, dim=0), dim=0
            )

            # First interpolate the filter_size rows alongside columns
            # stacked_lines is [filter_size, filter_size, C, H, W]
            # interpolated_lines is [filter_size, C, H, W]
            interpolated_lines = self.interpolate_1d(
                stacked_lines, fractional_flow[:, 0:1]
            )

            # interpolated_lines.unsqueeze(0) is [1, filter_size, C, H, W]
            # rows are shifted to the column dimension through unsqueeze.
            interpolated_column = self.interpolate_1d(
                interpolated_lines.unsqueeze(0), fractional_flow[:, 1:2]
            )
            # interpolated_column shape is [1, C, H, W]
            return interpolated_column


def vanilla_warp_fn(x: Tensor, flow: Tensor, mode: str = "bicubic") -> Tensor:
    """Motion compensation (warping) of a tensor [B, C, H, W] with a 2-d displacement
    [B, 2, H, W]. This function does not allows for longer filters than 4 taps
    (bicubic) and does not quantize the flows to a given subpixel accuracy in
    eval mode.

    Some code in this function is inspired from
    https://github.com/microsoft/DCVC/blob/main/DCVC-FM/src/models/block_mc.py
    License: MIT

    Args:
        x: Tensor to be motion compensated [B, C, H, W].
        flow: Displacement [B, C, H, W]. flow[:, 0, :, :] corresponds to
            the horizontal displacement. flow[:, 1, :, :] is the vertical displacement.

    Returns:
        Tensor: Motion compensated tensor [B, C, H, W].
    """
    B, _, H, W = x.size()
    cur_device = x.device

    tensor_hor = (
        torch.linspace(-1.0, 1.0, W, device=cur_device, dtype=torch.float32)
        .view(1, 1, 1, W)
        .expand(B, -1, H, -1)
    )
    tensor_ver = (
        torch.linspace(-1.0, 1.0, H, device=cur_device, dtype=torch.float32)
        .view(1, 1, H, 1)
        .expand(B, -1, -1, W)
    )
    backward_grid = torch.cat([tensor_hor, tensor_ver], 1)

    flow = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        dim=1,
    )

    grid = backward_grid + flow

    output = nn.functional.grid_sample(
        x,
        grid.permute(0, 2, 3, 1),
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )

    return output


if __name__ == "__main__":
    # Check that our custom warping works as PyTorch
    h, w = 480, 732
    dummy_img = torch.rand((1, 3, h, w))
    dummy_flow = torch.randn((1, 2, h, w)) * 30

    print("Checking that Cool-chic warping behave similarly to PyTorch grid_sample.")
    print("PSNR should be above 60 dB\n")

    s = f"{'Warping mode':<20}{'PSNR PyTorch / Cool-chic [dB]':<30}\n"
    for filter_size in [2, 4]:
        warper = Warper(
            WarpParameter(
                filter_size=filter_size,
                # Do not use torch, that's what we want to compare
                use_torch_if_available=False,
            ),
            [h, w],
        )
        warp_coolchic = warper.forward(dummy_img, dummy_flow)

        mode = warper.param.mode
        warp_torch = vanilla_warp_fn(
            dummy_img, dummy_flow, mode=mode.replace("torch_", "")
        )

        mse = (warp_torch - warp_coolchic).square().mean()
        psnr = -10 * torch.log10(mse)

        str_psnr = f"{psnr:7.4f}"
        s += f"{mode:<20}{str_psnr:<30}\n"

    print(s)

    print("timing...")
    import time

    dummy_target = torch.rand_like(dummy_img)
    device = "cuda:0"

    dummy_img = dummy_img.to(device)
    dummy_flow = dummy_flow.to(device)

    print(
        f"{'Warping mode':<20}"
        f"{'Time torch [s]':<30}"
        f"{'Time Cool-chic warper [s]':<30}"
        f"{'Ratio Cool-chic / torch':<30}"
    )
    for filter_size in [2, 4, 8, 12]:
        N = 200
        time_torch = 0
        time_coolchic = 0
        cool_chic_warper = Warper(
            WarpParameter(
                filter_size=filter_size,
                # Do not use torch, that's what we want to compare
                use_torch_if_available=False,
            ),
            [h, w],
        )
        cool_chic_warper.to(device)
        cool_chic_warper.eval()
        cool_chic_warper = torch.compile(
            cool_chic_warper,
            dynamic=False,
            mode="reduce-overhead",
            fullgraph=True,
        )

        for idx in range(N):
            start_time = time.time()
            mode = cool_chic_warper.param.mode

            if filter_size in [2, 4]:
                warp_torch = vanilla_warp_fn(
                    dummy_img, dummy_flow, mode=mode.replace("torch_", "")
                )
                if device == "cuda:0":
                    torch.cuda.synchronize()

                # First N // 2 iterations are warm-up for more accurate time measurements
                if idx > N // 2:
                    time_torch += time.time() - start_time
            else:
                time_torch = "N/A"

            start_time = time.time()
            warp_coolchic = cool_chic_warper(dummy_img, dummy_flow)
            if device == "cuda:0":
                torch.cuda.synchronize()

            # First N // 2 iterations are warm-up for more accurate time measurements
            if idx > N // 2:
                time_coolchic += time.time() - start_time

        if time_torch != "N/A":
            ratio = f"{time_coolchic / time_torch:.3f}"
            time_torch = f"{time_torch:.4f}"
        else:
            ratio = "N/A"

        time_coolchic = f"{time_coolchic:.4f}"

        print(
            f"{mode + str(cool_chic_warper.param.filter_size):<20}"
            f"{time_torch:<30}"
            f"{time_coolchic:<30}"
            f"{ratio:<30}"
        )
