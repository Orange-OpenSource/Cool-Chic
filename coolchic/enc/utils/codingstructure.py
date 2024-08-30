# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

"""Utilities to define the coding structures."""

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from enc.utils.misc import POSSIBLE_DEVICE

# The different frame types:
#       - I frames have no reference (intra)
#       - P frames have 1 single (past) reference
#       - B frames have 2 (past & future) references.
FRAME_TYPE = Literal["I", "P", "B"]

#   A GOP is defined as something starting with an intra frames and followed
# by an arbitrary number of inter (P or B) frames. As such the number of frames
# in the GOP is the number of inter frames + 1, i.e.
#
#   number_of_frames_in_gop = intra_period + 1 = number_of_inter_frames_in_gop + 1
#
#
#   E.g.:
#       I0 ---> P1 ---> P2 ---> P3 ---> P4 ---> P5 ---> P6 ---> P7 ---> P8
#
# Or a hierarchical random access GOP with nested B-frames (RA).
#   E.g.:
#          I0 -----------------------> P4 ------------------------> P8
#           \----------> B2 <---------/ \----------> B6 <----------/
#            \--> B1 <--/ \--> B3 <--/   \--> B5 <--/  \--> B7 <--/
#
# Here, both GOPs have an intra period of 8 (i.e. 8 inter-frames) in
# between two I-frames. First GOP P-period is 1, while second P period is 4
# which is the distance of the P-frame prediction.
#
FRAME_DATA_TYPE = Literal["rgb", "yuv420", "yuv444"]
POSSIBLE_BITDEPTH = Literal[8, 10]


class DictTensorYUV(TypedDict):
    """``TypedDict`` representing a YUV420 frame..

    .. hint::

        ``torch.jit`` requires I/O of modules to be either ``Tensor``, ``List``
        or ``Dict``. So we don't use a python dataclass here and rely on
        ``TypedDict`` instead.

    Args:
        y (Tensor): :math:`([B, 1, H, W])`.
        u (Tensor): :math:`([B, 1, \\frac{H}{2}, \\frac{W}{2}])`.
        v (Tensor): :math:`([B, 1, \\frac{H}{2}, \\frac{W}{2}])`.
    """

    y: Tensor
    u: Tensor
    v: Tensor


def yuv_dict_to_device(yuv: DictTensorYUV, device: POSSIBLE_DEVICE) -> DictTensorYUV:
    """Send a ``DictTensor`` to a device.

    Args:
        yuv: Data to be sent to a device.
        device: The requested device

    Returns:
        Data on the appropriate device.
    """
    return DictTensorYUV(
        y=yuv.get("y").to(device), u=yuv.get("u").to(device), v=yuv.get("v").to(device)
    )


# ============================== YUV upsampling ============================= #
def convert_444_to_420(yuv444: Tensor) -> DictTensorYUV:
    """From a 4D YUV 444 tensor :math:`(B, 3, H, W)`, return a
    ``DictTensorYUV``. The U and V tensors are down sampled using a nearest
    neighbor downsampling.

    Args:
        yuv444: YUV444 data :math:`(B, 3, H, W)`

    Returns:
        YUV420 dictionary of 4D tensors
    """
    assert yuv444.dim() == 4, f"Number of dimension should be 5, found {yuv444.dim()}"

    b, c, h, w = yuv444.size()
    assert c == 3, f"Number of channel should be 3, found {c}"

    # No need to downsample y channel but it should remain a 5D tensor
    y = yuv444[:, 0, :, :].view(b, 1, h, w)

    # Downsample U and V channels together
    uv = F.interpolate(yuv444[:, 1:3, :, :], scale_factor=(0.5, 0.5), mode="nearest")
    u, v = uv.split(1, dim=1)

    yuv420 = DictTensorYUV(y=y, u=u, v=v)
    return yuv420


def convert_420_to_444(yuv420: DictTensorYUV) -> Tensor:
    """Convert a DictTensorYUV to a 4D tensor:math:`(B, 3, H, W)`.
    The U and V tensors are up sampled using a nearest neighbor upsampling

    Args:
        yuv420: YUV420 dictionary of 4D tensor

    Returns:
        YUV444 Tensor :math:`(B, 3, H, W)`
    """
    u = F.interpolate(yuv420.get("u"), scale_factor=(2, 2))
    v = F.interpolate(yuv420.get("v"), scale_factor=(2, 2))
    yuv444 = torch.cat((yuv420.get("y"), u, v), dim=1)
    return yuv444


# ============================== YUV upsampling ============================= #


@dataclass
class FrameData:
    """FrameData is a dataclass storing the actual pixel values of a frame and
    a few additional information about its size, bitdepth of color space.

    Args:
        bitdepth (POSSIBLE_BITDEPTH): Bitdepth, either ``"8"`` or ``"10"``.
        frame_data_type (FRAME_DATA_TYPE): Data type, either ``"rgb"``,
            ``"yuv420"``, ``"yuv444"``.
        data (Union[Tensor, DictTensorYUV]): The actual RGB or YUV data
    """
    bitdepth: POSSIBLE_BITDEPTH
    frame_data_type: FRAME_DATA_TYPE
    data: Union[Tensor, DictTensorYUV]

    # Filled up by the __post_init__() function
    # ==================== Not set by the init function ===================== #
    #: Height & width of the video :math:`(H, W)`
    img_size: Tuple[int, int] = field(init=False)
    #: Number of pixels :math:`H \times W`
    n_pixels: int = field(init=False)  # Height x Width
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        if self.frame_data_type == "rgb" or self.frame_data_type == "yuv444":
            self.img_size = self.data.size()[-2:]
        elif self.frame_data_type == "yuv420":
            self.img_size = self.data.get("y").size()[-2:]

        self.n_pixels = self.img_size[0] * self.img_size[1]

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        """Push the data attribute to the relevant device **in place**.

        Args:
            device: The device on which the model should run.
        """
        if self.frame_data_type == "rgb" or self.frame_data_type == "yuv444":
            self.data = self.data.to(device)
        elif self.frame_data_type == "yuv420":
            self.data = yuv_dict_to_device(self.data, device)


@dataclass
class Frame:
    """Dataclass representing a frame to be encoded. It contains useful info
    like the display & coding indices, the indices of its references as well
    as the data of the decoded references and the original (*i.e.* uncompressed)
    frame.

    Args:
        coding_order (int): Frame with ``coding_order=0`` is coded first.
        display_order (int): Frame with ``display_order=0`` is displayed first.
        depth (int): Depth of the frame in the GOP. 0 for Intra, 1 for P-frame,
            2 or more for B-frames. Roughly corresponds to the notion of
            temporal layers in conventional codecs.
            Defaults to 0.
        seq_name (str): Name of the video. Mainly used for logging purposes.
            Defaults to ``""``.
        data (Optional[FrameData]): Data of the uncompressed image to be coded.
            Defaults to ``None``.
        already_encoded (bool): ``True`` if the frame has already been coded
            by the VideoEncoder. Defaults to False
        index_references (List[int]): Index of the frame(s) used as references,
            in **display_order**. Leave empty when no reference are available
            *i.e.* for I-frame. Defaults to ``[]``.
        ref_data (List[FrameData]): The actual data describing the decoded
            references. Leave empty when no reference are available
            *i.e.* for I-frame. Defaults to ``[]``.
    """
    coding_order: int
    display_order: int
    depth: int = 0
    seq_name: str = ""
    data: Optional[FrameData] = None
    decoded_data: Optional[FrameData] = None
    already_encoded: bool = False
    index_references: List[int] = field(default_factory=lambda: [])

    # Filled up by the set_refs_data() function.
    refs_data: List[FrameData] = field(default_factory=lambda: [])

    # ==================== Not set by the init function ===================== #
    #: Automatically set from the number of entry in ``self.index_references``.
    frame_type: FRAME_TYPE = field(init=False)
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        assert len(self.index_references) <= 2, (
            "A frame can not have more than 2 references.\n"
            f"Found {len(self.index_references)} references for frame {self.display_order} "
            f"(display order).\n Exiting!"
        )

        if len(self.index_references) == 2:
            self.frame_type = "B"
        elif len(self.index_references) == 1:
            self.frame_type = "P"
        else:
            self.frame_type = "I"

    def set_frame_data(
        self,
        data: Union[Tensor, DictTensorYUV],
        frame_data_type: FRAME_DATA_TYPE,
        bitdepth: POSSIBLE_BITDEPTH,
    ) -> None:
        """Set the data representing the frame i.e. create the ``FrameData``
        object describing the actual frame.

        Args:
            data: RGB or YUV value of the frame.
            frame_data_type: Data type.
            bitdepth: Bitdepth.
        """
        self.data = FrameData(
            bitdepth=bitdepth, frame_data_type=frame_data_type, data=data
        )

    def set_decoded_data(self, decoded_data: FrameData) -> None:
        """Set the data representing the decoded frame.

        Args:
            refs_data: Data of the reference(s)
        """
        # ! There might be a memory management issue here (deep copy vs. shallow copy)
        self.decoded_data = decoded_data

    def set_refs_data(self, refs_data: List[FrameData]) -> None:
        """Set the data representing the reference(s).

        Args:
            refs_data: Data of the reference(s)
        """
        assert len(refs_data) == len(self.index_references), (
            f"Trying to load data for "
            f"{len(refs_data)} references but current frame only has {len(self.index_references)} "
            f"references. Frame type is {self.frame_type}."
        )

        # ! There might be a memory management issue here (deep copy vs. shallow copy)
        self.refs_data = refs_data

    def upsample_reference_to_444(self) -> None:
        """Upsample the references from 420 to 444 **in place**. Do nothing
        if this is already the case.
        """
        upsampled_refs = []
        for ref in self.refs_data:
            if ref.frame_data_type == "yuv420":
                ref.data = convert_420_to_444(ref.data)
                ref.frame_data_type = "yuv444"

            upsampled_refs.append(ref)

        self.refs_data = upsampled_refs

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        """Push the data attribute to the relevant device **in place**.

        Args:
            device: The device on which the model should run.
        """
        if self.data is not None:
            self.data.to_device(device)

        for index_ref in range(len(self.refs_data)):
            if self.refs_data[index_ref] is not None:
                self.refs_data[index_ref].to_device(device)


@dataclass
class CodingStructure:
    """Dataclass representing the organization of the video *i.e.* which
    frames are coded using which references.

    A few examples:

    .. code-block::

        # A low-delay P configuration
        # I0 ---> P1 ---> P2 ---> P3 ---> P4 ---> P5 ---> P6 ---> P7 ---> P8
        intra_period=8 p_period=1

        # A hierarchical Random Access configuration
        # I0 -----------------------------------------------------> P8
        # \-------------------------> B4 <-------------------------/
        #  \----------> B2 <---------/ \----------> B6 <----------/
        #   \--> B1 <--/ \--> B3 <--/   \--> B5 <--/  \--> B7 <--/
        intra_period=8 p_period=8

        # There is no more prediction from I0 to P8. Instead the GOP in split in
        # half so that there is no inter frame with reference further than --p_period

        # I0 -----------------------> P4 ------------------------> P8
        #  \----------> B2 <---------/ \----------> B6 <----------/
        #   \--> B1 <--/ \--> B3 <--/   \--> B5 <--/  \--> B7 <--/
        intra_period=8 p_period=4

    A coding is composed of a few hyper-parameters and most importantly a
    list of ``Frame`` describing the different frames to code.

    Args:
        intra_period (int): Number of inter frames in the GOP. As such,
            the first (intra) frame of two successive GOPs would be spaced by
            `intra_period` inter frames. Set this to 0 for all intra coding.
        p_period (int): Distance to the furthest P prediction in the GOP. Set
            this to 1 for low-delay P or to ``intra_period`` for the usual
            random access configuration.
        seq_name (str): Name of the video. Mainly used for logging purposes.
            Defaults to ``""``.
    """
    intra_period: int
    p_period: int = 0
    seq_name: str = ""

    # ==================== Not set by the init function ===================== #
    #: All the frames to code, deduced from the GOP type, intra period and P period.
    #: Frames are index in display order (i.e. temporal order). frames[0] is the 1st
    #: frame, while frames[-1] is the last one.
    frames: List[Frame] = field(init=False)
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        self.frames = self.compute_gop(self.intra_period, self.p_period)

    def compute_gop(self, intra_period: int, p_period: int) -> List[Frame]:
        """Return a list of frames with one intra followed by ``intra_period``
        inter frames. The relation between the inter frames is implied by
        p_period. See examples in the class description.

        Args:
            intra_period: Number of inter frames in the GOP.
            p_period: Distance between I0 and the first P frame or between
                subsequent P-frames.

        Returns:
            List describing the frames to code.
        """
        # I-frame
        frames = [
            Frame(
                coding_order=0,
                display_order=0,
                index_references=[],
                seq_name=self.seq_name,
            )
        ]

        if intra_period == 0 and p_period == 0:
            print("Intra period is 0 and P period is 0: all intra coding!")
            return frames

        assert intra_period % p_period == 0, (
            f"Intra period must be divisible by P period."
            f" Found intra_period = {intra_period} ; p_period = {p_period}."
        )

        # In the example of RA GOP given above, the number of chained GOP is 2.
        n_chained_gop = intra_period // p_period

        for index_chained_gop in range(n_chained_gop):
            for index_frame_in_gop in range(1, p_period + 1):
                display_order = index_frame_in_gop + index_chained_gop * p_period

                depth_frame_in_gop = self.get_frame_depth_in_gop(index_frame_in_gop)

                # References display order are located at +/- delta_time_ref
                # from the current frame display order
                delta_time_ref = p_period // 2 ** (depth_frame_in_gop - 1)

                # First frame is an intra
                # Last frame of each chained GOP is a P-frame
                if index_frame_in_gop == p_period:
                    refs = [display_order - delta_time_ref]
                # Otherwise we have a B-frame
                else:
                    refs = [
                        display_order - delta_time_ref,
                        display_order + delta_time_ref,
                    ]

                if depth_frame_in_gop != 0:
                    # Coding order of the first frame with this depth in
                    # the current chained gop.
                    # Until depth = 3 (included), the depth **is** the coding order since
                    # all temporal layer whose depth is 0, 1, 2, 3 have a single frame.
                    # For depth >= 4, we must take into account that each previous layer
                    # of depth d_i < 4 has had 2 ** d_i - 1 frames in it.
                    coding_order_in_gop = depth_frame_in_gop + sum(
                        [2 ** (x - 2) - 1 for x in range(3, depth_frame_in_gop)]
                    )

                    # When depth >= 4 we have multiple frames per layer, this takes it into account
                    # to obtain the proper coding order
                    coding_order_in_gop += (index_frame_in_gop - delta_time_ref) // (
                        2 * delta_time_ref
                    )
                else:
                    coding_order_in_gop = 0
                coding_order = index_chained_gop * p_period + coding_order_in_gop

                frames.append(
                    Frame(
                        coding_order=coding_order,
                        display_order=display_order,
                        index_references=refs,
                        depth=depth_frame_in_gop,
                        seq_name=self.seq_name,
                    )
                )

        return frames

    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""

        COL_WIDTH = 14

        s = "Coding configuration:\n"
        s += "---------------------\n"

        s += f'{"Frame type":<{COL_WIDTH}}\t{"Coding order":<{COL_WIDTH}}\t{"Display order":<{COL_WIDTH}}\t'
        s += f'{"Ref 1":<{COL_WIDTH}}\t{"Ref 2":<{COL_WIDTH}}\t{"Depth":<{COL_WIDTH}}\t{"Encoded"}\n'

        for idx_coding_order in range(len(self.frames)):
            cur_frame = self.get_frame_from_coding_order(idx_coding_order)

            s += f"{cur_frame.frame_type:<{COL_WIDTH}}\t"
            s += f"{cur_frame.coding_order:<{COL_WIDTH}}\t"
            s += f"{cur_frame.display_order:<{COL_WIDTH}}\t"

            if len(cur_frame.index_references) > 0:
                s += f"{cur_frame.index_references[0]:<{COL_WIDTH}}\t"
            else:
                s += f'{"/":<{COL_WIDTH}}\t'

            if len(cur_frame.index_references) > 1:
                s += f"{cur_frame.index_references[1]:<{COL_WIDTH}}\t"
            else:
                s += f'{"/":<{COL_WIDTH}}\t'

            s += f"{cur_frame.depth:<{COL_WIDTH}}\t"

            s += f"{cur_frame.already_encoded:<{COL_WIDTH}}\t"

            s += "\n"
        return s

    def get_number_of_frames(self) -> int:
        """Return the number of frames in the coding structure.

        Returns:
            Number of frames in the coding structure.
        """
        return len(self.frames)

    def get_max_depth(self) -> int:
        """Return the maximum depth of a coding configuration

        Returns:
            Maximum depth of the coding configuration
        """
        return max([frame.depth for frame in self.frames])

    def get_all_frames_of_depth(self, depth: int) -> List[Frame]:
        """Return a list with all the frames for a given depth

        Args:
            depth: Depth for which we want the frames.

        Returns:
            List of frames with the given depth
        """
        return [frame for frame in self.frames if frame.depth == depth]

    def get_max_coding_order(self) -> int:
        """Return the maximum coding order of a coding configuration

        Returns:
            Maximum coding order of the coding configuration
        """
        return max([frame.coding_order for frame in self.frames])

    def get_frame_from_coding_order(self, coding_order: int) -> Optional[Frame]:
        """Return the frame whose coding order is equal to ``coding_order``.
        Return ``None`` if no frame has been found.

        Args:
            coding_order: Coding order for which we want the frame.

        Returns:
            Frame whose coding order is equal to ``coding_order``.
        """
        for frame in self.frames:
            if frame.coding_order == coding_order:
                return frame
        return None

    def get_max_display_order(self) -> int:
        """Return the maximum display order of a coding configuration

        Returns:
            Maximum display order of the coding configuration
        """
        return max([frame.display_order for frame in self.frames])

    def get_frame_from_display_order(self, display_order: int) -> Optional[Frame]:
        """Return the frame whose display order is equal to ``display_order``.
        Return None if no frame has been found.

        Args:
            display_order: Coding order for which we want the frame.

        Returns:
            Frame whose coding order is equal to ``display_order``.
        """
        for frame in self.frames:
            if frame.display_order == display_order:
                return frame
        return None

    def set_encoded_flag(self, coding_order: int, flag_value: bool) -> None:
        """Set the flag ``self.already_encode`` of the frame whose coding
        order is ``coding_order`` to the value ``flag_value``.

        Args:
            coding_order: Coding order of the frame for which we'll change the flag
            flag_value: Value to be set
        """
        for frame in self.frames:
            if frame.coding_order == coding_order:
                frame.already_encoded = flag_value

    def unload_all_decoded_data(self) -> None:
        """Remove the data describing the decoded data from the memory. This
        is used before saving the coding structure. The decoded data can be retrieved
        by re-inferring the trained model."""
        for idx_display_order in range(self.get_number_of_frames()):
            # if hasattr(self.frames[idx_display_order], "decoded_data"):
            #     del self.frames[idx_display_order].decoded_data
            # TODO: Set to None and rely on the garbage collector to
            # TODO: delete this?
            self.frames[idx_display_order].decoded_data = None

    def unload_all_original_frames(self) -> None:
        """Remove the data describing the original frame from the memory. This
        is used before saving the coding structure. The original frames can be
        retrieved by reloading the sequence"""
        for idx_display_order in range(self.get_number_of_frames()):
            # if hasattr(self.frames[idx_display_order], "data"):
            # del self.frames[idx_display_order].data
            # TODO: Set to None and rely on the garbage collector to
            # TODO: delete this?
            self.frames[idx_display_order].data = None

    def unload_all_references_data(self) -> None:
        """Remove the data describing all the references from the memory. This
        is used before saving the coding structure. The reference data can be
        retrieved by re-inferring the trained model."""
        for idx_display_order in range(self.get_number_of_frames()):
            # if hasattr(self.frames[idx_display_order], "refs_data"):
            #     del self.frames[idx_display_order].refs_data
            # TODO: Set to None and rely on the garbage collector to
            # TODO: delete this?
            self.frames[idx_display_order].refs_data = None

    def get_frame_depth_in_gop(self, idx_frame: int) -> int:
        """Return the depth of a frame with index <idx_frame> within a hierarchical GOP.

        Some notes:
            - ``idx_frame == 0`` **always** corresponds to an intra frame i.e. depth = 0
            - ``idx_frame == p_period`` is the P-frame i.e. depth = 1
            - This should be used separately for the successive chained GOPs.

        Args:
            idx_frame: Display order of the frame in the GOP.
            p_period: P-period. Should be a power of two.

        Returns:
            Depth of the frame in the GOP.
        """
        assert idx_frame <= self.p_period, (
            f"idx_frame should be <= to p_period."
            f" P-period is {self.p_period}, Index frame is {idx_frame}."
        )

        assert math.log2(self.p_period) % 1 == 0, (
            f"p_period should be a power of 2." f" P-period is {self.p_period}."
        )

        if idx_frame == 0:
            return 0

        # Compute the depth
        depth = int(math.log2(self.p_period) + 1)
        for i in range(int(math.log2(self.p_period)), 0, -1):
            if idx_frame % 2**i == 0:
                depth -= 1

        return int(depth)
