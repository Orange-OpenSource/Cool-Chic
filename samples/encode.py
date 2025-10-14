import argparse
import csv
import io
import os
import subprocess

from typing import Literal

PATH_COOL_CHIC_ENCODE = f"{os.path.dirname(__file__)}/../coolchic/encode.py"
PATH_COOL_CHIC_CFG = f"{os.path.dirname(__file__)}/../cfg/"
PATH_GET_CODING_STRUCT = f"{os.path.dirname(__file__)}/../coolchic/getcodingstruct.py"


def get_frame_config(frame_type: Literal["I", "P", "B"], depth: int, lmbda: float) -> str:
    """Return a string with the different command line arguments for
    encode.py script e.g. --start_lr=5e-3 --n_itr=1000 --dec_cfg_residue=lop.cfg

    These arguments are based on the frame_type and the depth in the GOP.
    """

    if frame_type == "I":
        config = (
            f"--enc_cfg={PATH_COOL_CHIC_CFG}enc/intra/fast_10k.cfg "
            f"--dec_cfg_residue={PATH_COOL_CHIC_CFG}dec/intra/hop.cfg "
            f"--lmbda={lmbda} "
        )
    elif frame_type == "P":
        config = (
            f"--enc_cfg={PATH_COOL_CHIC_CFG}enc/inter/tunable.cfg "
            f"--dec_cfg_residue={PATH_COOL_CHIC_CFG}dec/residue/mop.cfg "
            f"--dec_cfg_motion={PATH_COOL_CHIC_CFG}dec/motion/mop.cfg "
            f"--start_lr=5e-3 "
            f"--n_itr_pretrain_motion=3000 "
            f"--n_itr=10000 "
            f"--lmbda={lmbda} "
        )

    elif frame_type == "B":
        # At least 1000 iterations
        n_itr = max(10000 - 2000 * depth, 1000)
        n_itr_motion = max(5000 - 1000 * depth, 1000)

        config = (
            f"--enc_cfg={PATH_COOL_CHIC_CFG}enc/inter/tunable.cfg "
            f"--n_itr_pretrain_motion={n_itr_motion} "
            f"--n_itr={n_itr} "
            # Lambda adaptive given the depth for B-frame
            f"--lmbda={1.5**depth * lmbda} "
        )

        if depth == 1:
            config += (
                f"--dec_cfg_residue={PATH_COOL_CHIC_CFG}dec/residue/mop.cfg "
                f"--dec_cfg_motion={PATH_COOL_CHIC_CFG}dec/motion/mop.cfg "
            )
        else:
            config += (
                f"--dec_cfg_residue={PATH_COOL_CHIC_CFG}dec/residue/lop.cfg "
                f"--dec_cfg_motion={PATH_COOL_CHIC_CFG}dec/motion/lop.cfg "
            )

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O args -----
    parser.add_argument(
        "-i",
        "--input",
        help="Path of the input image. Either .png or .yuv",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path of the compressed bitstream. If empty, no bitstream is written",
        type=str,
        default="",
    )
    parser.add_argument(
        "--workdir", help="Path of the working_directory", type=str, default="."
    )
    # ----- I/O args

    # Video coding structure -----
    # Default parameters are suited for image coding (1 Intra frame)
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
        help="Display index of the P frames. Same format than --intra_pos ",
        type=str,
        default="",
    )
    parser.add_argument(
        "--n_frames",
        help="How many frames to code",
        type=int,
        default=1,
    )
    # ----- Video coding structure

    parser.add_argument("--lmbda", help="Rate constraint", type=float, default=1e-3)
    parser.add_argument(
        "--extra_args",
        help='Example --extra_args="--tune=wasserstein"',
        type=str,
        default=""
    )

    parser.add_argument(
        "--debug", action="store_true", help="Short encodings only for debugging."
    )
    args = parser.parse_args()

    # Gather all coding structure related arguments in a single string
    coding_struct_args = (
        f"--intra_pos={args.intra_pos} --p_pos={args.p_pos} --n_frames={args.n_frames} "
    )

    raw_coding_struct = subprocess.getoutput(
        f"python3 {PATH_GET_CODING_STRUCT} {coding_struct_args} --raw_coding_struct"
    )
    # coding_config = pd.read_csv(
    #     io.StringIO(raw_coding_struct), sep="\t", index_col=False
    # )

    list_frame_by_coding_idx = []
    with io.StringIO(raw_coding_struct) as file:
        my_reader = csv.reader(file, delimiter="\t")
        for idx_row, row in enumerate(my_reader):
            if idx_row == 0:
                header = row
            else:
                list_frame_by_coding_idx.append({k: v for k, v in zip(header, row)})

    # Encode each frame successively
    # All bitstreams in coding order
    all_bitstreams = []
    for coding_idx in range(args.n_frames):
        cur_bitstream = (
            os.path.dirname(args.output)
            + f"/coding-{str(coding_idx).zfill(4)}-"
            + os.path.basename(args.output)
        )


        if args.debug:
            if list_frame_by_coding_idx[coding_idx].get("type") == "I":
                intra_or_residue = "intra"
            else:
                intra_or_residue = "residue"
            config = (
                f"--enc_cfg={PATH_COOL_CHIC_CFG}enc/debug.cfg "
                f"--dec_cfg_residue={PATH_COOL_CHIC_CFG}dec/{intra_or_residue}/vlop.cfg "
                f"--dec_cfg_motion={PATH_COOL_CHIC_CFG}dec/motion/lop.cfg "
            )
        else:
            config = get_frame_config(
                frame_type=list_frame_by_coding_idx[coding_idx].get("type"),
                depth=int(list_frame_by_coding_idx[coding_idx].get("depth")),
                lmbda=args.lmbda,
            )

        config += f" {args.extra_args} "

        cmd = (
            f"python3 {PATH_COOL_CHIC_ENCODE} "
            # Recopy all common arguments from the command line
            f"--input={args.input} "
            f"--workdir={args.workdir} "
            f"{coding_struct_args} "  # Defined above
            # Frame-wise arguments
            f"--output={cur_bitstream} "
            f"--coding_idx={coding_idx} "
            # Training duration, decoder architectures
            f"{config} "
        )
        subprocess.call(cmd, shell=True)

        all_bitstreams.append(cur_bitstream)

    # Concatenate all bitstreams together, in coding order
    cat_cmd = "cat " + " ".join(all_bitstreams) + f" > {args.output}"
    subprocess.call(cat_cmd, shell=True)

    # Remove all the intermediate bitstreams
    rm_cmd = "rm " + " ".join(all_bitstreams)
    subprocess.call(rm_cmd, shell=True)
