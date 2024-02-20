import glob
import os
import subprocess
import argparse

POSSIBLE_DATASETS_PER_CONFIG = {
    'image': ['kodak', 'clic20-pro-valid', 'jvet'],
    'video-random-access': ['clic24-valid-subset'],
    'video-low-latency': ['clic24-valid-subset'],
}

parser = argparse.ArgumentParser()
parser.add_argument('configuration', help=f'Possible values {POSSIBLE_DATASETS_PER_CONFIG.keys()}.')
parser.add_argument('dataset_name', help=f'Possible values {POSSIBLE_DATASETS_PER_CONFIG}.')
args = parser.parse_args()


assert args.configuration in POSSIBLE_DATASETS_PER_CONFIG.keys(), 'Configuration should be in '\
    f'{POSSIBLE_DATASETS_PER_CONFIG.keys()}. Found {args.configuration}.'

cur_possible_dataset = POSSIBLE_DATASETS_PER_CONFIG.get(args.configuration)

assert args.dataset_name in cur_possible_dataset, \
    f'Dataset must be in {cur_possible_dataset} for configuration {args.configuration}.' \
    f'Found {args.dataset_name}!'

# Get path of input folder with all the bitstreams, create the output
# directory path for the decoded image and get path of the decode scripts
current_dir_path = os.path.dirname(__file__)
bitstream_path = os.path.join(current_dir_path, args.configuration, args.dataset_name, 'bitstreams/')
decoded_image_path = os.path.join(bitstream_path, '../decoded/')
cool_chic_decode_path = os.path.join(current_dir_path, '../src/decode.py')
subprocess.call(f'mkdir -p {decoded_image_path}', shell=True)

# Successively decode all bitstreams
all_bitstreams = glob.glob(f'{bitstream_path}*.bin')
image_or_video = 'image' if args.configuration == 'image' else 'video'

if args.dataset_name in ['kodak', 'clic20-pro-valid']:
    extension = 'png'
else:
    extension = 'yuv'

for idx, cur_bitstream in enumerate(all_bitstreams):
    print(f'\nDecoding {image_or_video} {idx + 1:>4} / {len(all_bitstreams)}...')
    cur_decoded_path = f'{decoded_image_path}{cur_bitstream.split("/")[-1].split(".")[0]}.{extension}'
    cmd = f'python3 {cool_chic_decode_path} -i {cur_bitstream} -o {cur_decoded_path}'
    subprocess.call(cmd, shell=True)