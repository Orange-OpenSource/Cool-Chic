# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

"""Utilities to define the coding structures."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from enc.io.framedata import FrameData


# The different frame types:
#       - I frames have no reference (intra)
#       - P frames have 1 single (past) reference
#       - B frames have 2 (past & future) references.
FRAME_TYPE = Literal["I", "P", "B"]


@dataclass
class Frame:
    """Dataclass representing a frame to be encoded. It contains useful info
    like the display & coding indices, the indices of its references as well
    as the data of the decoded references and the original (*i.e.* uncompressed)
    frame.

    Args:
        coding_order (int): Frame with ``coding_order=0`` is coded first.
        display_order (int): Frame with ``display_order=0`` is displayed first.
        frame_offset (int): Shift the position of the 0-th frame of the video.
            If frame_offset=15 skip the first 15 frames of the video. That is
            the display index 0 corresponds to the 16th frame.
            This is only used to load the data + for logging purposes
            Defaults to 0.
        depth (int): Depth of the frame in the GOP. 0 for Intra, 1 for P-frame,
            2 or more for B-frames. Roughly corresponds to the notion of
            temporal layers in conventional codecs.
            Defaults to 0.
        seq_name (str): Name of the video. Mainly used for logging purposes.
            Defaults to ``""``.
        data (Optional[FrameData]): Data of the uncompressed image to be coded.
            Defaults to ``None``.
        index_references (List[int]): Index of the frame(s) used as references,
            in **display_order**. Leave empty when no reference are available
            *i.e.* for I-frame. Defaults to ``[]``.
        ref_data (List[FrameData]): The actual data describing the decoded
            references. Leave empty when no reference are available
            *i.e.* for I-frame. Defaults to ``[]``.
    """

    coding_order: int
    display_order: int
    frame_offset: int = 0
    depth: int = 0
    seq_name: str = ""
    data: Optional[FrameData] = None
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

        # The reference further in the past is always first.
        self.index_references.sort()

        if len(self.index_references) == 2:
            self.frame_type = "B"
        elif len(self.index_references) == 1:
            self.frame_type = "P"
        else:
            self.frame_type = "I"

    def set_frame_data(self, data: FrameData) -> None:
        """Set the data representing the frame i.e. create the ``FrameData``
        object describing the actual frame.

        Args:
            data: FrameData object representing the frame.
        """
        self.data = data

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

    def pretty_string(self, show_header: bool = False, show_bottom_line: bool = False) -> str:
        """Return a string describing the frame.

        Args:
            show_header: Also print column nam. Defaults to False.
            show_bottom_line: Print a line below the frame description to close
                the array. Defaults to False.

        Returns:
            str: Pretty string describing the frame
        """
        COL_WIDTH = 18

        s = ""
        single_col = f"+{'-' * (COL_WIDTH - 2)}"
        vertical_line = single_col * 6 + "+"

        if show_header:
            s += vertical_line + "\n"
            # Column name
            s += f'|{"Frame type":^{COL_WIDTH-2}}|'
            s += f'{"Coding order":^{COL_WIDTH-2}}|'
            s += f'{"Display order":^{COL_WIDTH-2}}|'
            s += f'{"Ref 1 (disp)":^{COL_WIDTH-2}}|'
            s += f'{"Ref 2 (disp)":^{COL_WIDTH-2}}|'
            s += f'{"Depth":^{COL_WIDTH-2}}|'
            s += "\n"
            s += vertical_line + "\n"

        ref_1 = str(self.index_references[0]) if len(self.index_references) > 0 else "/"
        ref_2 = str(self.index_references[1]) if len(self.index_references) > 1 else "/"

        s += f"|{self.frame_type:^{COL_WIDTH-2}}|"
        s += f"{self.coding_order:^{COL_WIDTH-2}}|"
        s += f"{self.display_order:^{COL_WIDTH-2}}|"
        s += f"{ref_1:^{COL_WIDTH-2}}|"
        s += f"{ref_2:^{COL_WIDTH-2}}|"
        s += f"{self.depth:^{COL_WIDTH-2}}|"
        s += "\n"

        if show_bottom_line:
            s += vertical_line + "\n\n"

        return s


@dataclass
class CodingStructure:
    """Dataclass representing the organization of the video *i.e.* which
    frames are coded using which references.

    A few examples:

    .. code-block::

        # A low-delay P configuration
        # I0
        # \------> P1
        #             \-------> P2
        #                         \------> P3
        #                                     \-------> P4
        --n_frames=5 --intra_pos=0 --p_pos=1-4

        # A hierarchical Random Access configuration, with a closed GOP
        # I0
        # \-------------------------------------------------------------------------------------> P8
        # \----------------------------------------> B4 <----------------------------------------/
        # \-----------------> B2 <------------------/  \------------------> B6 <-----------------/
        # \------> B1 <------/  \-------> B3 <------/  \------> B5 <-------/  \------> B7 <------/
        --n_frames=8 --intra_pos=0 --p_pos=-1

        # A hierarchical Random Access configuration, with an open GOP
        # I0                                                                                      I8
        # \----------------------------------------> B4 <----------------------------------------/
        # \-----------------> B2 <------------------/  \------------------> B6 <-----------------/
        # \------> B1 <------/  \-------> B3 <------/  \------> B5 <-------/  \------> B7 <------/
        --n_frames=8 --intra_pos=0,-1

        # Or some very peculiar structures...
        # I0
        #   \---------------------------------------------------------------> P6
        #   \-----------------------------> B3 <-----------------------------/  \-----------------> P8
        #   \------> B1 <------------------/  \------> B4 <------------------/  \------> B7 <------/
        #              \------> B2 <-------/             \------> B5 <-------/
        --n_frames=8 --intra_pos=0 --p_pos=6,8

    A coding is composed of a few hyper-parameters and most importantly a
    list of ``Frame`` describing the different frames to code.

    Args:
        n_frames (int): Number of frames in the coding structure
        frame_offset (int): Shift the position of the 0-th frame of the video.
            If frame_offset=15 skip the first 15 frames of the video. That is
            the display index 0 corresponds to the 16th frame.
        intra_pos (List[int]): Position of all the intra frames in display
            order
        p_pos (List[int]): Position of all the P frames in display order
        seq_name (str): Name of the video. Mainly used for logging purposes.
            Defaults to ``""``.
    """

    seq_name: str = ""

    n_frames: int = 1
    frame_offset: int = 0
    # Intra and P positions are given in **display** order
    # Always start with an intra
    intra_pos: List[int] = field(default_factory=lambda: [0])
    p_pos: List[int] = field(default_factory=lambda: [])

    # ==================== Not set by the init function ===================== #
    #: All the frames to code, deduced from the GOP type, intra period and P period.
    #: Frames are index in display order (i.e. temporal order). frames[0] is the 1st
    #: frame, while frames[-1] is the last one.
    frames: List[Frame] = field(init=False)
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        self.intra_pos.sort()
        self.p_pos.sort()

        first_frame_is_intra = self.intra_pos[0] == 0
        assert first_frame_is_intra, (
            "First frame of the video should an intra frame. Change --intra_pos "
            "to include the frame 0."
        )

        last_frame_is_intra = self.intra_pos[-1] == self.n_frames - 1
        last_frame_is_p = self.p_pos[-1] == self.n_frames - 1 if self.p_pos else False
        assert last_frame_is_intra or last_frame_is_p, (
            "Last frame of the video should be either an intra frame or a P "
            "frame. Add -1 to --intra_pos or --p_pos to include the last frame."
        )

        if len(self.intra_pos) != len(set(self.intra_pos)):
            print(
                f"Found duplicate elements in --intra_pos: {self.intra_pos}.\n"
                "They are automatically removed."
            )

        if len(self.p_pos) != len(set(self.p_pos)):
            print(
                f"Found duplicate elements in --p_pos: {self.p_pos}.\n"
                "They are automatically removed."
            )

        common_elements = list(set(self.intra_pos).intersection(self.p_pos))
        assert not common_elements, (
            "Frames can not be an I-frame and a P-frame at the same time!\n"
            f"Found --intra_pos={self.intra_pos} --p_pos={self.p_pos}.\n"
            f"Frame(s) {common_elements} are in both arguments, they should "
            "be present only in one of them."
        )

        self.frames = self.compute_coding_struct(
            self.n_frames, self.intra_pos, self.p_pos
        )

    def compute_coding_struct(
        self, n_frames: int, intra_pos: List[int], p_pos: List[int]
    ) -> List[Frame]:
        """Construct a coding structure of n_frames. The algorithm works as
        follows.

        Step 1:
        -------

            Position all the intra frames following ``intra_pos``.

        Step 2:
        -------

            Position all the P frames following ``p_pos``. A P-frame use the
            closest frame in the past as a reference.

        Step 3:
        -------

            Automatically fill the remaining frames with hierarchical B-frames.
            This is achieved by iterating on the list of frames and inserting
            B-frames in between already added frames each time there is a gap.
            For instance:

                frames =    [I0, P4]
                        ==> [I0, B2, P4]            # Fill the middle frame
                        ==> [I0, B1, B2, P4]        # Fill the middle frame
                        ==> [I0, B1, B2, B3 P4]     # Fill the middle frame

        Args:
            n_frames (int): Number of frames in the coding structure
            intra_pos (List[int]): Position of all the intra frames in display
                order
            p_pos (List[int]): Position of all the P frames in display order

        Returns:
            List[Frame]: List of all the frames within the coding structure.
        """

        frames = []

        # ----- Step 1: fill all the intra frames
        for idx_display_order in intra_pos:
            frames.append(
                Frame(
                    coding_order=len(frames),
                    display_order=idx_display_order,
                    index_references=[],
                    depth=0,  # All intra depth is 0
                    seq_name=self.seq_name,  # Not very elegant... but useful!
                    frame_offset=self.frame_offset,
                )
            )

            frames.sort(key=lambda x: x.display_order)

        def get_closest_past_ref(idx_display_order: int, frames: List[Frame]) -> Frame:
            """Return the biggest display_order present in frames that is still
            smaller than idx_display_order. It corresponds to the index of the
            closest past reference.

            **Everything is in display order**.

            Args:
                idx_display_order (int): Display index of the frame for which
                    we want to find the closest past reference.
                frames (List[Frames]): List of the already coded (available) frames.

            Returns:
                int: Display order of the closest past reference.
            """
            frames.sort(key=lambda x: x.display_order)

            # The * P-frame will used the P3 frame as reference regardless
            # of the actual display order of the * P-frame (from 4 to 7)
            # I0  P3 * I8
            closest_frame = frames[0]

            for frame in frames:
                if frame.display_order >= idx_display_order:
                    break
                closest_frame = frame

            return closest_frame

        def get_closest_future_ref(idx_display_order: int, frames: List[Frame]) -> int:
            """Return the smallest display_order present in frames that is still
            bigger than idx_display_order. It corresponds to the index of the
            closest future reference.

            **Everything is in display order**.

            Args:
                idx_display_order (int): Display index of the frame for which
                    we want to find the closest future reference.
                frames (List[Frames]): List of the already coded (available) frames.

            Returns:
                int: Display order of the closest future reference.
            """
            frames.sort(key=lambda x: x.display_order, reverse=True)

            # The * P-frame will used the I8 frame as reference regardless
            # of the actual display order of the * P-frame (from 4 to 7)
            # I0  P3 * I8
            closest_frame = frames[0]

            for frame in frames:
                if frame.display_order <= idx_display_order:
                    break
                closest_frame = frame

            return closest_frame

        # ----- Step 2: fill all the P frames
        for idx_display_order in p_pos:
            past_ref = get_closest_past_ref(idx_display_order, frames)
            frames.append(
                Frame(
                    coding_order=len(frames),
                    display_order=idx_display_order,
                    index_references=[past_ref.display_order],
                    depth=past_ref.depth + 1,
                    seq_name=self.seq_name,
                    frame_offset=self.frame_offset,
                )
            )

            frames.sort(key=lambda x: x.display_order)

        # ----- Step 3: Fill out the blanks with B-frames in a hierarchical manner
        # Stop when we've filled the coding structure with n_frames
        while len(frames) < n_frames:
            # Iterate on the frames list and stop each time we find a "gap".
            # Create a B frame right in the middle of this gap.
            for i in range(n_frames):
                # Case 1: we've already constructed this frame
                already_coded_frames = [x.display_order for x in frames]
                if i in already_coded_frames:
                    continue

                # Case 2: we need to construct a new frame
                past_ref = get_closest_past_ref(i, frames)
                future_ref = get_closest_future_ref(i, frames)

                # The display order of the frame being creating is equal to the
                # past reference + half the distance between its 2 references
                ref_distance = future_ref.display_order - past_ref.display_order
                idx_display_order = past_ref.display_order + ref_distance // 2
                frames.append(
                    Frame(
                        coding_order=len(frames),
                        display_order=idx_display_order,
                        index_references=[
                            past_ref.display_order,
                            future_ref.display_order,
                        ],
                        depth=max([past_ref.depth, future_ref.depth]) + 1,
                        seq_name=self.seq_name,
                        frame_offset=self.frame_offset,
                    )
                )

                frames.sort(key=lambda x: x.display_order)

                # Loop once more!
                break

        return frames

    def pretty_structure_diagram(self) -> str:
        """Return a nice diagram presenting the coding structure. Like:

        .. code::

            I0 -----------------------------------------------------> P8
            \-------------------------> B4 <-------------------------/
             \----------> B2 <---------/ \----------> B6 <----------/
              \--> B1 <--/ \--> B3 <--/   \--> B5 <--/  \--> B7 <--/


        Returns:
            str: A string describing the coding structure. Ready to be printed.
        """

        # Handle edge case where there is a single frame to be coded
        if self.n_frames == 1:
            return "I0"

        _LENGTH_PRINT = 10 * self.n_frames

        all_x_pos = [
            round(x / (self.n_frames - 1) * _LENGTH_PRINT) for x in range(self.n_frames)
        ]
        # print(all_x_pos)

        lines = []
        # print(f"{'frame':<8}{'spacing':<8}{'l_len':<8}{'r_len':<8}{'x_pos':<8}")

        for depth in range(self.get_max_depth() + 1):
            current_x_pos = 0
            s = ""
            for frame in self.get_all_frames_of_depth(depth):
                frame_str = f"{frame.frame_type}{frame.display_order}"

                # No ref, whitespace left and right
                if frame.frame_type == "I":
                    spacing = all_x_pos[frame.display_order] - current_x_pos
                    s += f"{' ' * spacing}{frame_str}"
                    current_x_pos += spacing + len(frame_str)

                # Only past ref, \----> on the left, only whitespace on the right
                elif frame.frame_type == "P":
                    # All frames requires at least to character e.g. B4 +
                    # one more additional character for each additional digit
                    len_left_ref_str = (
                        2
                        + int(frame.index_references[0] >= 10)
                        + int(frame.index_references[0] >= 100)
                    )
                    spacing = (
                        all_x_pos[frame.index_references[0]]
                        + len_left_ref_str
                        - current_x_pos
                    )
                    left_arrow_len = (
                        all_x_pos[frame.display_order]
                        - all_x_pos[frame.index_references[0]]
                        - len_left_ref_str
                    )

                    s += f"{' ' * spacing}"
                    s += f"\{'-' * (left_arrow_len - 3)}> "
                    s += f"{frame_str}"

                    # print(f"{frame_str:<8}{spacing:<8}{left_arrow_len:<8}{' ':<8}{current_x_pos:<8}")
                    current_x_pos += spacing + left_arrow_len + len(frame_str)

                # Past and future \---> on the left, <---/ on the right
                elif frame.frame_type == "B":
                    # All frames requires at least to character e.g. B4 +
                    # one more additional character for each additional digit
                    len_left_ref_str = (
                        2
                        + int(frame.index_references[0] >= 10)
                        + int(frame.index_references[0] >= 100)
                    )

                    spacing = (
                        all_x_pos[frame.index_references[0]]
                        + len_left_ref_str
                        - current_x_pos
                    )
                    left_arrow_len = (
                        all_x_pos[frame.display_order]
                        - all_x_pos[frame.index_references[0]]
                        - len_left_ref_str
                    )
                    right_arrow_len = (
                        all_x_pos[frame.index_references[1]]
                        - all_x_pos[frame.display_order]
                        - len(frame_str)
                    )

                    s += f"{' ' * spacing}"
                    s += f"\{'-' * (left_arrow_len - 3)}> "
                    s += f"{frame_str}"
                    s += f" <{'-' * (right_arrow_len - 3)}/"

                    # print(f"{frame_str:<8}{spacing:<8}{left_arrow_len:<8}{right_arrow_len:<8}{current_x_pos:<8}")
                    current_x_pos += (
                        spacing + left_arrow_len + right_arrow_len + len(frame_str)
                    )

            lines.append(s)

        results = "\n".join(lines)
        return results

    def pretty_string(self, print_detailed_struct: bool = False) -> str:
        """Return a pretty string formatting the data within the class

        Args:
            print_detailed_struct: True to print the detailed coding structure

        Returns:
            str: a pretty string ready to be printed out

        """

        COL_WIDTH = 18

        s = "Coding configuration:\n"
        s += "---------------------\n"

        s += f"{'n_frames':<26}: {self.n_frames}\n"
        s += f"{'frame_offset':<26}: {self.frame_offset}\n"
        s += f"{'seq_name':<26}: {self.seq_name}\n"
        s += f"{'intra_pos':<26}: {', '.join([str(x) for x in self.intra_pos])}\n"
        s += f"{'p_pos':<26}: {', '.join([str(x) for x in self.p_pos])}\n\n"

        if not print_detailed_struct:
            return s

        # Print row after tow
        for idx_coding_order in range(len(self.frames)):
            cur_frame = self.get_frame_from_coding_order(idx_coding_order)
            s += cur_frame.pretty_string(
                show_header=idx_coding_order == 0,
                show_bottom_line=idx_coding_order == len(self.frames) - 1
            )

        s += self.pretty_structure_diagram()

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

    def get_all_frames_using_one_ref(self, display_order_ref: int) -> List[Frame]:
        """Return a list of frames using the frame <display_order_ref> as
        a reference.

        Args:
            display_order_ref: Display order of the frame that is used as reference

        Returns:
            List[Frame]: List of frames using one given frame as a reference.
        """

        res = []
        for frame in self.frames:
            if display_order_ref in frame.index_references:
                res.append(frame)
        return res
