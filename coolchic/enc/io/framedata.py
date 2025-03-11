from dataclasses import dataclass, field
from enc.io.types import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH
from typing import Any, Tuple


@dataclass
class FrameData:
    """FrameData is a dataclass storing the actual pixel values of a frame and
    a few additional information about its size, bitdepth of color space.

    Args:
        bitdepth: Bitdepth, should be in``[8, 9, 10, 11, 12, 13, 14, 15, 16]``.
        frame_data_type: Data type, either ``"rgb"``, ``"yuv420"``, ``"yuv444"``.
        data: The actual RGB or YUV data
    """

    bitdepth: POSSIBLE_BITDEPTH
    frame_data_type: FRAME_DATA_TYPE
    data: Any #: Union[Tensor, DictTensorYUV]

    # Filled up by the __post_init__() function
    # ==================== Not set by the init function ===================== #
    #: Height & width of the video :math:`(H, W)`
    img_size: Tuple[int, int] = field(init=False)
    #: Number of pixels :math:`H \times W`
    n_pixels: int = field(init=False)  # Height x Width
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        if self.frame_data_type == "yuv420":
            self.img_size = self.data.get("y").size()[-2:]
        else:
            self.img_size = self.data.size()[-2:]
        self.n_pixels = self.img_size[0] * self.img_size[1]

    def to_string(self) -> str:
        """Pretty string describing the frame data."""
        s = "Frame data information:\n"
        s += "-----------------------\n"
        s += f"{'Resolution (H, W)':<26}: {self.img_size[0]}, {self.img_size[1]}\n"
        s += f"{'Bitdepth':<26}: {self.bitdepth}\n"
        s += f"{'Data type':<26}: {self.frame_data_type}"

        return s
