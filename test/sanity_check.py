import glob
import os
import subprocess
from typing import Any, Dict, List

import numpy as np

SAMPLE_PATH = f"{os.path.dirname(__file__)}/../samples/"


# To be launched with python3 -m test.sanity_check from the root of the git repo
def encode(
    seq_path: str, bitstream_path: str, workdir: str, n_frames: int = 1, extra_args: str = ""
) -> None:
    """Use Cool-chic to encode a sequence into a bitstream."""

    subprocess.call(f"mkdir -p {workdir}", shell=True)
    # Delete already existing frame encoder
    subprocess.call(f"rm -f {workdir}*-frame_encoder.pt", shell=True)

    # ---- Get number of frames
    # Train the encoder
    cmd = (
        f"python3 {SAMPLE_PATH}/encode.py "
        f"--input={seq_path} "
        f"--output={bitstream_path} "
        f"--workdir={workdir} "
        f"--n_frames={n_frames} "
        f'--extra_args="{extra_args}" '
        "--debug "
    )
    if n_frames > 1:
        cmd += " --p_pos=-1 "
    subprocess.call(cmd, shell=True)


def parse_log(results_log_path: List[str]) -> Dict[str, Any]:

    res = {"rate_bpp": 0.0, "mse": 0}
    n_frames = len(results_log_path)

    # Parse the results one after the other
    for res_log in results_log_path:
        # Check the log at the encoder side
        logs = [x.rstrip("\n") for x in open(res_log).readlines()]
        keys = [x for x in logs[0].split() if x]
        vals = [x for x in logs[1].split() if x]
        results = {k: v for k, v in zip(keys, vals)}

        res["rate_bpp"] += float(results.get("rate_bpp"))
        res["mse"] += 10 ** (-float(results.get("psnr_db")) / 10)

    res["rate_bpp"] /= n_frames
    res["mse"] /= n_frames
    res["psnr_db"] = -10 * np.log10(res["mse"] + 1e-10)

    return res


if __name__ == "__main__":
    print("Starting sanity check...\n")
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

    LIST_SEQ_TO_TEST = [
        {"path": "/data/192x128_kodim15.png", "extra_args": "", "n_frames": 1},
        {"path": "/data/192x128_kodim15.png", "extra_args": "--tune=wasserstein", "n_frames": 1},
        {"path": "/data/kodim15_192x128_01p_yuv420_8b.yuv", "extra_args": "", "n_frames": 1},
        {"path": "/data/kodim15_192x128_01p_yuv420_10b.yuv", "extra_args": "", "n_frames": 1},
        {"path": "/data/kodim15_192x128_01p_yuv444_8b.yuv", "extra_args": "", "n_frames": 1},
        {"path": "/data/kodim15_192x128_01p_yuv444_10b.yuv", "extra_args": "", "n_frames": 1},
        {
            "path": "/data/D-BQSquare-5frames_224x128_60p_yuv420_8b.yuv",
            "extra_args": "",
            "n_frames": 5,
        },
    ]

    # Absolute path
    for i, seq_i in enumerate(LIST_SEQ_TO_TEST):
        LIST_SEQ_TO_TEST[i]["path"] = os.path.dirname(__file__) + LIST_SEQ_TO_TEST[i]["path"]

    TEST_WORKDIR = os.path.dirname(__file__) + "/test-workdir/"

    msg = "Final sanity check results:\n\n'"
    # For each sequence, measure that there is not much deviation between the
    # encoder logs and what we obtain at the decoder side.
    for seq_to_test in LIST_SEQ_TO_TEST:
        # Clean everything in TEST_WORKDIR
        subprocess.call(f"rm -f {TEST_WORKDIR}/*", shell=True)

        seq_path = seq_to_test.get("path")
        extra_args = seq_to_test.get("extra_args")
        n_frames = seq_to_test.get("n_frames")

        # Get all the path required to encode, decode and evaluate
        seq_name, ext = os.path.splitext(os.path.basename(seq_path))
        decoded_ext = ".yuv" if ext == ".yuv" else ".png"
        decoded_path = f"{TEST_WORKDIR}decoded-from-bitstream-{seq_name}{decoded_ext}"
        bitstream_path = f"{TEST_WORKDIR}{seq_name}.cool"

        # Encode also decode the bitstream
        encode(seq_path, bitstream_path, TEST_WORKDIR, n_frames=n_frames, extra_args=extra_args)

        enc_res = parse_log(glob.glob(f"{TEST_WORKDIR}/*-results_encoder.tsv"))
        dec_res = parse_log(glob.glob(f"{TEST_WORKDIR}/*-results_decoder.tsv"))

        enc_psnr_db = enc_res["psnr_db"]
        dec_psnr_db = dec_res["psnr_db"]
        enc_rate_bpp = enc_res["rate_bpp"]
        dec_rate_bpp = dec_res["rate_bpp"]

        s = (
            "\n\n"
            f"Sequence name: {seq_name} ; extra_args = {extra_args} \n"
            " "
            + " " * 12
            + "|"
            + f"{'Encoder estimation':^24}"
            + "|"
            + f"{'Actual results':^24}"
            + "|"
            + "\n"
            "|"
            + f"{'PSNR [dB]':^12}"
            + "|"
            + f"{f'{enc_psnr_db:5.3f}':^24}"
            + "|"
            + f"{f'{dec_psnr_db:5.3f}':^24}"
            + "|"
            + "\n"
            "|"
            + f"{'Rate [bpp]':^12}"
            + "|"
            + f"{f'{enc_rate_bpp:5.3f}':^24}"
            + "|"
            + f"{f'{dec_rate_bpp:5.3f}':^24}"
            + "|"
            + "\n"
        )
        msg += s

        delta_psnr = enc_psnr_db - dec_psnr_db
        threshold_delta_psnr = 0.3
        assert abs(delta_psnr) < threshold_delta_psnr, (
            f"Difference between the estimated PSNR at the encoder side "
            f"({enc_psnr_db:5.3f} dB) and the actual PSNR "
            f"({dec_psnr_db:5.3f} dB) is greater than the expected threshold: "
            f"+/- {threshold_delta_psnr:.2f} dB."
        )

        threshold_ratio_rate = 0.2
        ratio_rate = (dec_rate_bpp - enc_rate_bpp) / enc_rate_bpp
        assert abs(ratio_rate) < threshold_ratio_rate, (
            f"Difference between the estimated rate at the encoder side "
            f"({enc_rate_bpp:5.3f} bpp) and the actual rate "
            f"({dec_rate_bpp:5.3f} bpp) is greater than the expected threshold: "
            f"+/- {100 * ratio_rate:.2f} %."
        )

    msg += "\nSanity check completed successfully!"
    print(msg)
