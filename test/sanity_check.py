import glob
import os
import subprocess
from typing import Any, Dict, List

import numpy as np

TOOLBOX_PATH = f"{os.path.dirname(__file__)}/../toolbox/toolbox/"
SAMPLE_PATH = f"{os.path.dirname(__file__)}/../samples/"

# To be launched with python3 -m test.sanity_check from the root of the git repo
def encode(seq_path: str, bitstream_path: str, workdir: str) -> None:
    """Use Cool-chic to encode a sequence into a bitstream."""

    subprocess.call(f"mkdir -p {workdir}", shell=True)
    # Delete already existing frame encoder
    subprocess.call(f"rm -f {workdir}*-frame_encoder.pt", shell=True)

    # ---- Get number of frames
    cmd = (
        f"python3 {TOOLBOX_PATH}/identify.py "
        f"--input={seq_path} "
        f"--noheader "
    )
    out_identify = subprocess.getoutput(cmd)

    _, _, n_frames = [int(x) for x in out_identify.split()[:3] if x]

    # Train the encoder
    cmd = (
        f"python3 {SAMPLE_PATH}/encode.py "
        f'--input={seq_path} '
        f'--output={bitstream_path} '
        f'--workdir={workdir} '
        f'--n_frames={n_frames} '
        '--debug '
    )
    if n_frames > 1:
        cmd += " --p_pos=-1 "
    subprocess.call(cmd, shell=True)


def decode(bitstream_path: str, decoded_path: str) -> None:
    # Decode the file
    cmd = (
        'python3 ./coolchic/decode.py '
        f'--input={bitstream_path} '
        f'--output={decoded_path} '
        '--verbosity=2 '
        '--no_avx2 '
    )
    subprocess.call(cmd, shell=True)


def parse_encoder_log(results_log_path: List[str]) -> Dict[str, Any]:

    res = {"rate_bpp": 0., "mse": 0}
    n_frames = len(results_log_path)

    # Parse the results one after the other
    for res_log in results_log_path:
        # Check the log at the encoder side
        encoder_logs = [x.rstrip("\n") for x in open(res_log).readlines()]
        keys = [x for x in encoder_logs[0].split(' ') if x]
        vals = [x for x in encoder_logs[1].split(' ') if x]
        encoder_results = {k: v for k, v in zip(keys, vals)}

        res["rate_bpp"] += float(encoder_results.get("rate_bpp"))
        res["mse"] += 10 ** (-float(encoder_results.get("psnr_db")) / 10)

    res["rate_bpp"] /= n_frames
    res["mse"] /= n_frames
    res["psnr_db"] = -10 * np.log10(res["mse"] + 1e-10)

    return res


def evaluate_decoder_side(seq_path: str, decoded_path: str, bitstream_path: str) -> Dict[str, Any]:

    # ---- Get rate
    cmd = (
        f"python3 {TOOLBOX_PATH}/identify.py "
        f"--input={decoded_path} "
        f"--noheader "
    )
    out_identify = subprocess.getoutput(cmd)

    width, height, n_frames = [int(x) for x in out_identify.split()[:3] if x]
    n_pixels = width * height * n_frames

    rate_bytes = os.path.getsize(bitstream_path)
    rate_bpp = rate_bytes * 8 / n_pixels

    decoder_results = {
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "rate_bpp": rate_bpp
    }

    # ---- Get PSNR
    cmd = (
        f"python3 {TOOLBOX_PATH}/quality_psnr.py "
        f"--candidate={decoded_path} "
        f"--ref={seq_path} "
        f"--noheader "
    )
    out = subprocess.getoutput(cmd)

    if seq_path.endswith(".yuv"):
        psnr_db_611, psnr_db_y, psnr_db_u, psnr_db_v = map(lambda x: float(x), [x for x in out.split() if x][1:5])

        # Mimic cool-chic psnr computation: mse weighted by the number of pixels
        width_uv, height_uv = map(lambda x: int(x), [x for x in out_identify.split()[4:6] if x])
        n_pix_y = width * height
        n_pix_uv = width_uv * height_uv

        weighted_mse = (
            n_pix_y * 10 ** (-psnr_db_y / 10)
            + n_pix_uv * 10 ** (-psnr_db_u / 10)
            + n_pix_uv * 10 ** (-psnr_db_v / 10)
        ) / (n_pix_y + 2 * n_pix_uv)
        weighted_psnr = -10 * np.log10(weighted_mse + 1e-10)
        decoder_results.update({"psnr_db": weighted_psnr})

    # image
    else:
        psnr_db_rgb = float([x for x in out.split() if x][1])
        decoder_results.update({"psnr_db": float(psnr_db_rgb)})

    return decoder_results

if __name__ == '__main__':

    print("Starting sanity check...\n")
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

    LIST_SEQ = [
        "/data/192x128_kodim15.png",
        "/data/kodim15_192x128_01p_yuv420_8b.yuv",
        "/data/D-BQSquare_224x128_60p_yuv420_8b.yuv",
        "/data/kodim15_192x128_01p_yuv420_10b.yuv",
        "/data/kodim15_192x128_01p_yuv444_8b.yuv",
        "/data/kodim15_192x128_01p_yuv444_10b.yuv",
    ]

    # Absolute path
    LIST_SEQ = [os.path.dirname(__file__) + seq_path for seq_path in LIST_SEQ]

    TEST_WORKDIR = os.path.dirname(__file__) + '/test-workdir/'


    msg = "Final sanity check results:\n\n'"
    # For each sequence, measure that there is not much deviation between the
    # encoder logs and what we obtain at the decoder side.
    for seq_path in LIST_SEQ:
        # Clean everything in TEST_WORKDIR
        subprocess.call(f"rm {TEST_WORKDIR}/*", shell=True)

        # Get all the path required to encode, decode and evaluate
        seq_name, ext = os.path.splitext(os.path.basename(seq_path))
        decoded_ext = ".yuv" if ext == ".yuv" else ".ppm"
        decoded_path = f"{TEST_WORKDIR}decoded-from-bitstream-{seq_name}{decoded_ext}"
        bitstream_path = f'{TEST_WORKDIR}{seq_name}.cool'

        encode(seq_path, bitstream_path, TEST_WORKDIR)
        decode(bitstream_path, decoded_path)

        enc_res = parse_encoder_log(
            glob.glob(f"{TEST_WORKDIR}/*-results_best.tsv")
        )
        dec_res = evaluate_decoder_side(seq_path, decoded_path, bitstream_path)

        enc_psnr_db = enc_res['psnr_db']
        dec_psnr_db = dec_res['psnr_db']
        enc_rate_bpp = enc_res['rate_bpp']
        dec_rate_bpp = dec_res['rate_bpp']

        s = (
            '\n\n'
            f'Sequence name: {seq_name}\n'
            " " + " " * 12 + "|" + f"{'Encoder estimation':^24}" + "|" + f"{'Actual results':^24}" + "|" + "\n"
            "|" + f"{'PSNR [dB]':^12}" + "|" + f"{f'{enc_psnr_db:5.3f}':^24}" + "|" + f"{f'{dec_psnr_db:5.3f}':^24}" + "|" + "\n"
            "|" + f"{'Rate [bpp]':^12}" + "|" + f"{f'{enc_rate_bpp:5.3f}':^24}" + "|" + f"{f'{dec_rate_bpp:5.3f}':^24}" + "|" + "\n"
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
