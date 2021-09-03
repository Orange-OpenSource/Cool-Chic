# AIVC: _Artificial Intelligence for Video Coding_

__AIVC__ is a practical learned video coder which offers
* All the usual coding configurations: All Intra, Low-delay P and Random Access, with a tunable intra-period (up to 64) and GOP size
* Different rate targets, from a few hundred of kbit/s to several Mbit/s (equivalent to QP from 25 to 40 in traditional coding)
* Performance on par with the HM (HEVC Test Model) for all configurations and rate constraints. 
* Compress high-resolution videos (up to 1080p)

AIVC is an refined version of the system described in our ICLR 21 paper: [_Conditional Coding for Flexible Learned Video Compression_, Ladune _et al._](https://arxiv.org/abs/2104.09103)

__Performance screenshot goes here!__


<!-- 
You should add a [link to the full documentation]()

_{Only if that makes sense, add a nice screen-shot as below}_

![Software Screen-shot](http://placekitten.com/g/320/240 "software screen-shot") -->

## Quick start

### Download

Clone the repositories from GitHub:

```
$ git clone https://github.com/Orange-OpenSource/AIVC.git
```

### Docker container

The best way to ensure reproducibility is to run the code within a docker container built from the following Dockerfile

```
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN apt-get update && \
    pip install scipy && \
    pip install torchac # Thanks to Fabian Mentzer for the package! https://github.com/fab-jul/torchac
```

Create the docker image by executing the following command within the folder containing the Dockerfile 

```
$ docker build -t aivc .
```

Finally, launch an interactive container of the aivc docker image

```bash
$ docker run -it -v <path_to_aivc>:<path_to_aivc> aivc bash # <path_to_aivc> is the path where the repo is cloned
```

### Sanity check

Finally, launch the following script to ensure than everything is working properly.

```
$ cd AIVC/src
$ ./sanity_script.sh
```

This scripts encodes, decodes and measures the size and quality of the compressed video. It should return

```
PSNR    [dB]: 26.72133
MS-SSIM     : 0.93531
MS-SSIM [dB]: 11.89147
Size [bytes]: 28429
```

## Usage

### Structure

The script ```aivc.py``` performs 3 tasks
1. It encodes a .yuv video into a bitstream
2. It decodes a bitstream into a .yuv video
3. It measures the size of the bitstream and the quality (MS-SSIM and PSNR) of the compressed video. (_Quality measure derives from the [CLIC video track](http://compression.cc/)_)

### Data format

The input and output format is YUV 420. To be processed by the model and to measure the quality, each frame is transformed into a triplet of PNGs, one for each color channel.

## How to compress?

The ```sanity_script.sh``` provides an example of how to compress a video.

```bash
python aivc.py \
    -i ../raw_videos/BlowingBubbles_416x240_50_420 \
    -o ../compressed.yuv \
    --bitstream_out ../bitstream.bin \
    --start_frame 0 \
    --end_frame 100 \
    --coding_config RA \
    --gop_size 16 \
    --intra_period 32 \
    --model ms_ssim-2021cc-6
```

| Option          | Description                                                | Usage                                                                                                     | Example                                                                                            |
|-----------------|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| -i              | Path of the input video.                                   | Either a .yuv file or a folder containing the already extracted PNGs triplet                              | -i ../raw_videos/BlowingBubbles_416x240_50_420.yuv  -i ../raw_videos/BlowingBubbles_416x240_50_420 |
| -o              | Path of the compressed video                               | A .yuv file (the PNG triplets are generated alongside the .yuv in a dedicated folder)                     | -o ../compressed.yuv                                                                               |
| --bitstream_out | Path of the bitstream                                      | A .bin file                                                                                               | --bitstream_out ../bitstream.bin                                                                   |
| --start_frame   | Index of the first frame to compress                       | An integer, 0 corresponds to the very first frame                                                         | --start_frame 0                                                                                    |
| --end_frame     | Index of the last frame to compress                        | An integer, the last frame is included. Use -1 to compress until the last frame.                          | --end_frame 100                                                                                    |
| --coding_config | Desired coding configuration                               | RA for Random Access (I, P and B-frames) LDP for Low-delay P (I and P-frames) AI for All Intra (I-frames) | --coding_configuration RA                                                                          |
| --gop_size      | Number of frames within a hierarchical GOP (RA only)       | Must be a power of two. Min: 2, Max: 64.  This is different from the intra period! See example below.     | --gop_size 16                                                                                      |
| --intra_period  | Number of inter-frames between two intra (RA and LDP only) | Must be a power of two (RA and LDP) Must be a multiple of gop size (RA) Min: 2, Max: 64.                  | --intra_period 32                                                                                  |
| --model         | Model used to perform encoding and decoding.               | ms_ssim-2021cc-X where X in [1, 7]. 1 is the highest rate, 7 the lowest rate.                                    | --model ms_ssim-2021cc-6                                                                                  |
| --cpu           | Run on CPU                                                 |                                                                                                           | --cpu                                                                                              |

## Coding structures

### Random Access

This is a random access coding structure
   * Intra period: 16
   * GOP size 8

<img src="doc/coding_structures/RA_big.png" alt="RA" height="178"/>

### Low-delay P

This is a low-delay P coding structure
   * Intra period: 16

<img src="doc/coding_structures/LDP_big.png" alt="LDP" height="150"/>

### All Intra

Plain image coding for all the frames

<img src="doc/coding_structures/AI.png" alt="AI" height="135"/>

## Contribute

Questions, remarks, bug reports can be posted on the [AIVC google group](https://groups.google.com/g/aivc).

## Changelog

* July 2021
  * Initial release of the code
  * Models

## License

Copyright 2021 Orange

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Contact

* [Personal page](https://theoladune.github.io/)
* E-mail: theo.ladune@orange.com