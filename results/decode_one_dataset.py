import glob
import os
import subprocess
import argparse

POSSIBLE_DATASETS = ['kodak', 'clic20-pro-valid', 'clic22-test', 'jvet']

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help=f'Possible values {POSSIBLE_DATASETS}.')
args = parser.parse_args()

assert args.dataset_name in POSSIBLE_DATASETS, \
    f'Argument must be in {POSSIBLE_DATASETS}. Found {args.dataset_name}!'

# Get path of input folder with all the bitstreams, create the output
# directory path for the decoded image and get path of the decode scripts
current_dir_path = os.path.dirname(__file__)
bitstream_path = os.path.join(current_dir_path, args.dataset_name, 'bitstreams/')
decoded_image_path = os.path.join(bitstream_path, '../decoded/')
cool_chic_decode_path = os.path.join(current_dir_path, '../src/decode.py')
subprocess.call(f'mkdir -p {decoded_image_path}', shell=True)

# Successively decode all bitstreams
all_bitstreams = glob.glob(f'{bitstream_path}*.bin')
for idx, cur_bitstream in enumerate(all_bitstreams):
    print(f'\nDecoding image {idx:>4} / {len(all_bitstreams)}...')
    cur_decoded_path = f'{decoded_image_path}{cur_bitstream.split("/")[-1].split(".")[0]}.png'
    cmd = f'python3 {cool_chic_decode_path} -i {cur_bitstream} -o {cur_decoded_path}'
    subprocess.call(cmd, shell=True)