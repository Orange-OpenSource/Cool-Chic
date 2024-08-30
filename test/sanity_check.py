import os
import subprocess

from PIL import Image
import numpy as np

# To be launched with python3 -m test.sanity_check from the root of the git repo

if __name__ == '__main__':

    print("Starting sanity check...\n")
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

    input_image = './test/data/192x128_kodim15.png'
    test_workdir = 'test/test-workdir/'
    bitstream_path = f'{test_workdir}bitstream.cool'
    decoded_path = f'{test_workdir}decoded.ppm'

    subprocess.call(f"mkdir -p {test_workdir}", shell=True)

    # Delete already existing test
    cmd = f'rm -f {test_workdir}video_encoder.pt'
    subprocess.call(cmd, shell=True)

    # Train the encoder
    cmd = (
        'python3 ./coolchic/encode.py '
        f'--input={input_image} '
        f'--output={bitstream_path} '
        f'--workdir={test_workdir} '
        '--enc_cfg=./cfg/enc/debug.cfg '
        '--dec_cfg=./cfg/dec/vlop.cfg '
    )
    subprocess.call(cmd, shell=True)

    # Decode the file
    cmd = (
        'python3 ./coolchic/decode.py '
        f'--input={bitstream_path} '
        f'--output={decoded_path} '
        '--no_avx2'
    )
    subprocess.call(cmd, shell=True)

    # Compute the PSNR
    flag_yuv = input_image.endswith('.yuv')

    if flag_yuv:
        print("YUV PSNR computation not implemented yet")
    else:
        # Check the log at the encoder side
        encoder_logs = [
            x.rstrip("\n")
            for x in open(f"{test_workdir}results_best.tsv", "r").readlines()
        ]
        keys = [x for x in encoder_logs[0].split(' ') if x]
        vals = [x for x in encoder_logs[1].split(' ') if x]

        encoder_results = {k: v for k, v in zip(keys, vals)}

        enc_psnr_db = float(encoder_results["psnr_db"])
        enc_rate_bpp = float(encoder_results["total_rate_bpp"])

        # Measure stuff at the decoder side

        img_a = Image.open(input_image)
        img_b = Image.open(decoded_path)
        np_img_a = np.asarray(img_a).astype(np.uint32)
        np_img_b = np.asarray(img_b).astype(np.uint32)
        mse = np.mean((np_img_a - np_img_b) ** 2)
        dec_psnr_db = 10 * np.log10(255 ** 2 / mse)

        n_pixels = img_a.height * img_a.width
        dec_rate_bpp = os.path.getsize(bitstream_path) * 8 / n_pixels

        s = '\n\nFinal sanity check results:\n\n'
        s += "|" + " " * 12 + "|" + f"{'Encoder estimation':^24}" + "|" + f"{'Actual results':^24}" + "|" + "\n"
        s += "|" + f"{'PSNR [dB]':^12}" + "|" + f"{f'{enc_psnr_db:5.3f}':^24}" + "|" + f"{f'{dec_psnr_db:5.3f}':^24}" + "|" + "\n"
        s += "|" + f"{'Rate [bpp]':^12}" + "|" + f"{f'{enc_rate_bpp:5.3f}':^24}" + "|" + f"{f'{dec_rate_bpp:5.3f}':^24}" + "|" + "\n"
        print(s)

        delta_psnr = enc_psnr_db - dec_psnr_db
        threshold_delta_psnr = 0.1
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
        print("\nSanity check completed!")
