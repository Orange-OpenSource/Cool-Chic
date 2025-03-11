import argparse
import sys

from enc.utils.codingstructure import CodingStructure
from enc.utils.parsecli import get_coding_structure_from_args


def get_job_dependencies(coding_structure: CodingStructure) -> str:
    """Output a bash-like script file with dependencies for the coding
    structure. Everything is indexed in display order.

    Example
    -------

    With this coding structure,

    I0                                                          I5
    \-------------------> B2 <-------------------------------/
    \-------> B1 <-------/  \-------> B3 <-------------------/
                                        \-------> B4 <-------/

    it produces a string containing:

    jobid_0000=$(/usr/local/bin/sbatch --parsable  frame0000.sh)
    jobid_0005=$(/usr/local/bin/sbatch --parsable  frame0005.sh)
    jobid_0002=$(/usr/local/bin/sbatch --parsable --dependency=afterok:${jobid_0000}:${jobid_0005} frame0002.sh)
    jobid_0001=$(/usr/local/bin/sbatch --parsable --dependency=afterok:${jobid_0000}:${jobid_0002} frame0001.sh)
    jobid_0003=$(/usr/local/bin/sbatch --parsable --dependency=afterok:${jobid_0002}:${jobid_0005} frame0003.sh)
    jobid_0004=$(/usr/local/bin/sbatch --parsable --dependency=afterok:${jobid_0003}:${jobid_0005} frame0004.sh)

    Args:
        coding_structure (CodingStructure): The coding structure for which we
            want to list the dependencies.

    Returns:
        str: A string with a pseudo bash script listing all the dependencies.
            See above for an example.
    """

    SBATCH_PATH = "/usr/local/bin/sbatch"
    s = ""

    def get_display_idx_str(display_order: int) -> str:
        return str(display_order).zfill(4)

    # We **must** launch job in coding order to retrieve the job id for
    # the dependencies. Otherwise, everything is in display order!
    for coding_order in range(0, coding_structure.get_max_coding_order() + 1):
        frame = coding_structure.get_frame_from_coding_order(coding_order)
        frame_display_idx_str = get_display_idx_str(frame.display_order)

        # No dependencies
        if frame.index_references:
            dependencies = []

            for idx_ref_display_order in frame.index_references:
                idx_ref_display_order = coding_structure.get_frame_from_display_order(
                    idx_ref_display_order
                ).display_order
                dependencies.append(get_display_idx_str(idx_ref_display_order))

            dependencies = ":".join(
                [
                    "${" + f"jobid_{idx_ref_display_order}" + "}"
                    for idx_ref_display_order in dependencies
                ]
            )
            dependencies = f"--dependency=afterok:{dependencies}"

        # No reference = no dependency
        else:
            dependencies = ""

        s += (
            f"jobid_{frame_display_idx_str}="
            f"$({SBATCH_PATH} --parsable "
            f"{dependencies} "
            f"frame{frame_display_idx_str}.sh)"
            "\n"
        )

    return s


def get_raw_coding_structure(coding_structure: CodingStructure) -> str:
    """Get an easily parsable string representing the coding structure.

    Example
    -------

    With this coding structure,

    I0                                                          I5
    \-------------------> B2 <-------------------------------/
    \-------> B1 <-------/  \-------> B3 <-------------------/
                                        \-------> B4 <-------/

    it produces a string containing:

    coding	display	type	depth	ref1	ref2
    0	0	I	0
    1	5	I	0
    2	2	B	1	0	5
    3	1	B	2	0	2
    4	3	B	2	2	5
    5	4	B	3	3	5

    Args:
        coding_structure (CodingStructure): The coding structure for which we
            want to list the frames.

    Returns:
        str: Easily parsable string describing the coding structure. See above.
    """

    s = "coding\tdisplay\ttype\tdepth\tref1\tref2\n"
    for coding_order in range(0, coding_structure.get_max_coding_order() + 1):
        frame = coding_structure.get_frame_from_coding_order(coding_order)
        s += (
            f"{frame.coding_order}\t"
            f"{frame.display_order}\t"
            f"{frame.frame_type}\t"
            f"{frame.depth}\t"
            f"{frame.index_references[0] if frame.index_references else ''}\t"
            f"{frame.index_references[1] if len(frame.index_references) > 1 else ''}\t"
            "\n"
        )
    return s



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ----- CodingStructure parameters, copy of coolchic/encode.py args
    parser.add_argument(
        "--intra_pos",
        help="Display index of the intra frames. "
        "Format is 0,4,7 if you want the frame 0, 4 and 7 to be intra frames. "
        "-1 can be used to denote the last frame, -2 the 2nd to last etc. "
        "x-y is a range from x (included) to y (included). This does not work "
        "with the negative indexing. "
        "0,4-7,-2 ==> Intra for the frame 0, 4, 5, 6, 7 and the 2nd to last."
        "Frame 0 must be an intra frame.",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--p_pos",
        help="Display index of the P frames. " "Same format than --intra_pos ",
        type=str,
        default="",
    )
    parser.add_argument(
        "--n_frames",
        help="How many frames to code",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--frame_offset",
        help="Shift the position of the 0-th frame of the video. "
        "If --frame_offset=15 skip the first 15 frames of the video.",
        type=int,
        default=0,
    )

    # ----- What do we want to generate
    parser.add_argument(
        "--job_dependency", help="Print slurm job dependencies",
        action="store_true"
    )

    parser.add_argument(
        "--raw_coding_struct", help="Print raw coding structure in a tsv-like format",
        action="store_true"
    )
    args = parser.parse_args()

    if args.job_dependency and args.raw_coding_struct:
        print(
            "It is not possible to use --job_dependency and --raw_coding_struct"
            " at the same time. Use one or the other"
        )
        sys.exit(1)

    coding_structure = CodingStructure(**get_coding_structure_from_args(args))

    if args.job_dependency:
        print(get_job_dependencies(coding_structure))

    if args.raw_coding_struct:
        print(get_raw_coding_structure(coding_structure))