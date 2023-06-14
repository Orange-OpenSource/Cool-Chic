# Work in progress for a new release!

## Release note:

* Convolution-based synthesis transform
* Adapted upsampling module
* Improved Straight Through Estimator derivative during training
* YUV 420 input format in 8-bit and 10-bit
* Fixed point arithmetic for the probability model allowing for cross-platform entropy (de)coding
* More convenient encoding recipes _e.g._ fast, medium, slow, placebo
* Improved compression performance: around -15% compared to the previous version
  described in [_COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image
  Codec_](https://arxiv.org/abs/2212.05458), Ladune et al.

# COOL-CHIC

COOL-CHIC is a neural image codec based on __overfitting__. It offers the following features:
- Lightweight decoder complexity: __2 000 multiplications__ / decoded pixel and as few parameters
- Compression performance on par with HEVC on RGB images

More information available in the paper: [_COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec_](https://arxiv.org/abs/2212.05458), Ladune et al.

📢📢📢 __New paper coming soon!__ 📢📢📢

# Usage

## Installation

```bash
    >>> python3 -m pip install virtualenv               # Install virtual env if needed
    >>> virtualenv venv && source venv/bin/activate     # Create and activate a virtual env named "venv"
    >>> python3 -m pip install -r requirements.txt      # Install the required packages
```

## Using Apple M1/M2 Neural Engine

For some operations, the mps backend is not yet available. If you want to use
Apple Neural Engine, you need to set the following environment variable:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Encoding an image

The script ```samples/encode_image.sh``` provides an example for encoding an image with Cool-chic:

```bash
python3 src/encode.py                               \
    --input=samples/biville.png                     \   # Image to code
    --output=samples/bitstream.bin                  \   # Compressed bitstream
    --model_save_path=samples/model.pt              \   # Save the overfitted model here for analysis purpose
    --enc_results_path=samples/encoder_results.txt  \   # Save encoder logs for analysis purpose
    --lmbda=0.0002                                  \   # Rate constraint
    --start_lr=1e-2                                 \   # Learning rate
    --recipe=instant                                \   # Encoding preset. Available: instant, faster, fast, medium, slow, placebo
    --layers_synthesis=40-1-linear-relu,3-1-linear-relu,3-3-residual-relu,3-3-residual-none \   # Synthesis architecture outdim-kernelsize-layertype-nonlinearity
    --layers_arm=24,24                              \   # ARM architecture hidden layers dimension
    --n_ctx_rowcol=3                                \   # Number of rows and columns of context
    --latent_n_grids=7                                  # Number of latent resolutions
```
## Decoding a bitstream

The script ```samples/decode_image.sh``` provides an example for decoding an image with Cool-chic:

```bash
python3 src/decode.py                               \
    --input=samples/bitstream.bin                   \   # Bitstream path
    --output=samples/biville_decoded.png                # Storage path for the decoded image.
```

## YUV420 format

Cool-chic is able to process YUV420 data with 8-bit and 10-bit representation. To do so, you need to call the ```src/encode.py``` script with an appropriately formatted name such as:

    --input=<videoname>_<W>x<H>_<framerate>_yuv420_<bitdepth>b.yuv

The encoder codes only the first frame of the video.

## ⚠ ⚠ ⚠ Disclaimer ⚠ ⚠ ⚠

This repository is only a proof of concept illustrating how the ideas of Cool-chic could be translated into an actual codec.
The encoding process (i.e. learning the MLPs and latent representation) might suffer from failures. In may help to do successive encoding and keep the best one or tweak the learning rate.

# Thanks

We used the [_constriction package_](https://github.com/bamler-lab/constriction) from Robert Bamler as our entropy coder. Many thanks for the great work!

  R. Bamler, Understanding Entropy Coding With Asymmetric Numeral Systems (ANS): a Statistician's Perspective, arXiv preprint [arXiv:2201.01741](https://arxiv.org/pdf/2201.01741.pdf).