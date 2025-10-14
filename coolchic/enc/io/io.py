from enc.utils.codingstructure import FrameData
from enc.io.types import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH
from enc.io.format.ppm import read_ppm, write_ppm
from enc.io.format.yuv import read_yuv, write_yuv
from enc.io.format.png import read_png, write_png

import os


def load_frame_data_from_file(file_path: str, idx_display_order: int) -> FrameData:
    """Load the idx_display_order-th frame from a .yuv file or .png file. For the latter,
    idx_display_order must be equal to 0 as there is only one frame in a png.

    Args:
        file_path (str): Absolute path of the file from which the frame is loaded.
        idx_display_order (int): Index of the frame in display order

    Returns:
        FrameData: The loaded frame, wrapped as a FrameData object.
    """
    POSSIBLE_EXT = [".yuv", ".png", ".ppm"]
    assert file_path[-4:] in POSSIBLE_EXT, (
        "The function load_frame_data_from_file() expects a file ending with "
        f"{POSSIBLE_EXT}. Found {file_path}"
    )

    if file_path.endswith(".yuv"):
        # ! We only consider yuv420 and 444 planar
        bitdepth: POSSIBLE_BITDEPTH = 8 if "_8b" in file_path else 10
        frame_data_type: FRAME_DATA_TYPE = "yuv420" if "420" in file_path else "yuv444"
        data = read_yuv(file_path, idx_display_order, frame_data_type, bitdepth)

    elif file_path.endswith(".png"):
        frame_data_type: FRAME_DATA_TYPE = "rgb"
        data, bitdepth = read_png(file_path)

    elif file_path.endswith(".ppm"):
        frame_data_type: FRAME_DATA_TYPE = "rgb"
        data, bitdepth = read_ppm(file_path)

    return FrameData(bitdepth, frame_data_type, data)

def save_frame_data_to_file(frame_data: FrameData, file_path: str) -> None:
    """Save the data of a FrameData into a PNG, PPM or YUV file.
    file_path extension must match the FrameData type e.g. PNG or PPM for
    RGB and YUV for YUV420 or YUV444

    Args:
        frame_data (FrameData): The data to save
        file_path (str): Absolute path of the file from which the frame is stored.
    """

    POSSIBLE_EXT = [".yuv", ".png", ".ppm"]

    cur_extension = os.path.splitext(file_path)[1]
    assert cur_extension in POSSIBLE_EXT, (
        "The function save_frame_data_to_file() expects a file ending with "
        f"{POSSIBLE_EXT}. Found {file_path}"
    )

    if cur_extension == ".png":

        assert frame_data.frame_data_type == "rgb", (
            "The function save_frame_data_to_file() can only save a RGB data "
            f"into a PNG file. Found frame_data_type = {frame_data.frame_data_type}."
        )

        assert frame_data.bitdepth == 8, (
            "The function save_frame_data_to_file() can only write 8-bit data "
            f"into a PNG file. Found bitdepth = {frame_data.bitdepth}."
        )

        write_png(frame_data.data, file_path)

    elif cur_extension == ".ppm":

        assert frame_data.frame_data_type == "rgb", (
            "The function save_frame_data_to_file() can only save a RGB data "
            f"into a PPM file. Found frame_data_type = {frame_data.frame_data_type}."
        )

        write_ppm(frame_data.data, frame_data.bitdepth, file_path, norm=True)

    elif cur_extension == ".yuv":

        assert frame_data.frame_data_type in ["yuv420", "yuv444"], (
            "The function save_frame_data_to_file() can only save a YUV data "
            f"into a YUV file. Found frame_data_type = {frame_data.frame_data_type}."
        )

        write_yuv(
            frame_data.data,
            frame_data.bitdepth,
            frame_data.frame_data_type,
            file_path,
            norm=True,
        )
