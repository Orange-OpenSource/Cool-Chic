# COOL-CHIC

This repository contains an implementation of [_COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec_](https://arxiv.org/abs/2212.05458).

COOL-CHIC is a neural image codec based on __overfitting__. It offers the following features:
- Decode an image with less than __700 multiplications__ / decoded pixel
- Competitive with Ballé's hyperprior autoencoder (https://openreview.net/forum?id=rkcQFMZRb)
- Reach performance close to HEVC (HM), with +20 % BD-rate.

More information available in the paper: [_COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec_](https://arxiv.org/abs/2212.05458), Ladune et al.

# Usage

## Installation

    >>> python3 -m pip install virtualenv               # Install virtual env if needed
    >>> virtualenv venv && source venv/bin/activate     # Create and activate a virtual env named "venv"
    >>> python3 -m pip install -r requirements.txt      # Install the required packages

## Example

The scripts ```samples/code_image_gpu.sh``` and ```samples/code_image_cpu.sh``` provide examples for COOL-CHIC usage.

    >>> ./samples/code_image_gpu.sh         # Use the cpu script if you don't have a GPU (it can be quite long!)
    ... Many logs are printed to the terminal
    >>> cat samples/encoder_results.txt
    psnr,rate_latent_bpp,rate_mlp_bpp,rate_all_bpp,training_time_second,n_iteration,decoder_rate_bpp,decoder_psnr_db
    33.428055,0.7880071004231771,0.010424812324345112,0.7984318733215332,    5.7,  999.0,0.82692,33.42805

At the end of the 1000 iterations the entropy coding takes place and a bitstream is written. The ```encode.py``` script also includes decoding (which can be found at ```decode.py```). The final results can be found at ```samples/encoder_results.txt```. The different fields of the result file are:

|         Name         |                                  Meaning                                  | Unit          |
|:--------------------:|:-------------------------------------------------------------------------:|---------------|
|         **psnr**        |              Encoder-side (i.e. omitting entropy coding) PSNR             | dB            |
|    rate_latent_bpp   | Encoder-side (i.e. omitting entropy coding) rate for the latent variables | bit per pixel |
| rate_mlp_bpp         | Encoder-side (i.e. omitting entropy coding) rate synthesis and ARM MLPs   | bit per pixel |
| **rate_all_bpp**         | Encoder-side (i.e. omitting entropy coding) total rate                    | bit per pixel |
| training_time_second | Total training time                                                       | second        |
| n_iteration          | Total number of training iterations                                       | /             |
| decoder_rate_bpp     | Total rate (size of the bitstream)                                        | bit per pixel |
| decoder_psnr_db      | PSNR of the image decoded from the bitstream                              | dB

Due to a naïve implementation of the entropy coding step, decoder-side results are slightly worse than the encoder-side one (see disclaimer below). This should be solved by a better implementation.

Results from the paper are obtained using the __psnr__ and __rate_all_bpp__ rows.

The ```results/``` folder provides the detailed results from the paper as well as on the ```samples/biville.png``` image.

## Arguments

The different arguments accepted by the encoding script are explained here.

    python3 src/encode.py                               \
        --device=cuda:0                                 \   # Run on GPU
        -i samples/biville.png                          \   # Image to compress
        -o samples/bitstream.bin                        \   # Where to write the bitstream
        --decoded_img_path=samples/decoded.png          \   # Save the decoded image here
        --model_save_path=samples/model.pt              \   # Save the resulting .pt model here
        --enc_results_path=samples/encoder_results.txt  \   # Encoder-side results (prior to bitstream)
        --lmbda=0.001                                   \   # Rate constraint: J = Dist + lmbda Rate
        --start_lr=1e-2                                 \   # Initial learning rate
        --n_itr=1000                                    \   # Maximum number of iterations (should be set to 1e6)
        --layers_synthesis=12,12                        \   # 2 hidden layers of width 12 for the synthesis, as in the paper
        --layers_arm=12,12                              \   # 2 hidden layers of width 12 for the synthesis, as in the paper
        --n_ctx_rowcol=2                                \   # Use 2 rows and columns of context pixels for the ARM (C = 12) as in the paper
        --latent_n_grids=7                                  # Use 7 latent grids (from 1:1 to 1:64), as in the paper


## ⚠ ⚠ ⚠ Disclaimer ⚠ ⚠ ⚠

This repository is only a proof of concept illustrating how the ideas of our [_COOL-CHIC paper_](https://arxiv.org/abs/2212.05458) could be translated into an actual codec. As such it suffers a number of limitations, which mainly revolves around the entropy coding step.
- All operations are done using floating point arithmetic, including entropy coding. Due to floating point drift:
  - The bitstream produced by the encoder are __not__ cross-platform and can only be decoded on the same hardware that produced them.
  - Even on the same hardware, floating point arithmetic is __not guaranteed to be bit-exact__. T his implementation circumvents this issue by quantizing the probability distribution parameters (expectation $\mu$ and standard deviation $\sigma$). This has a noticeable impact on performance (around +4 % of rate according to our empirical measurements).
- The encoding process (i.e. learning the MLPs and latent representation) might suffer from failures. In may help to:
  - Do successive encoding and keep the best one
  - Tweak the learning rate

# Thanks

We used the [_constriction package_](https://github.com/bamler-lab/constriction) from Robert Bamler as our entropy coder. Many thanks for the great work!

  R. Bamler, Understanding Entropy Coding With Asymmetric Numeral Systems (ANS): a Statistician's Perspective, arXiv preprint [arXiv:2201.01741](https://arxiv.org/pdf/2201.01741.pdf).