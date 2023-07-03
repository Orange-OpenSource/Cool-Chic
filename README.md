# COOL-CHIC

Cool-chic (pronounced <span class="ipa">/kul ʃik/</span> as in French 🥖🧀🍷) is a <b>low-complexity</b>  neural image codec based on __overfitting__. With only <b>2 000 multiplications / decoded pixel</b>, it offers coding performance on par with HEVC.

More information available on [Cool-chic page](https://orange-opensource.github.io/Cool-Chic/) or in the paper: [_COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec_](https://arxiv.org/abs/2212.05458), Ladune et al.

📢📢📢 __New paper coming soon!__ 📢📢📢

# What's new?

This new Cool-Chic version provides several new features 🔥🔥🔥

* -15% rate vs. [_COOL-CHIC, Ladune et al._](https://arxiv.org/abs/2212.05458) thanks to
  * Convolution-based synthesis transform
  * Adapted upsampling module
  * Improved Straight Through Estimator derivative during training
* Friendlier usage
  * Support for YUV 420 input format in 8-bit and 10-bit
  * Fixed point arithmetic for the probability model allowing for cross-platform entropy (de)coding
  * Encoding recipes _e.g._ ```fast```, ```medium```, ```slow```, ```placebo```

# Setup

## Necessary packages

```bash
python3 -m pip install virtualenv               # Install virtual env if needed
virtualenv venv && source venv/bin/activate     # Create and activate a virtual env named "venv"
python3 -m pip install -r requirements.txt      # Install the required packages
```

## Using Apple M1/M2 Neural Engine

For some operations, the mps backend is not yet available. If you want to use
Apple Neural Engine, you need to set the following environment variable:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

# Reproducing paper results

Already encoded files are provided as bitstreams in ```results/<dataset_name>/```where ```<dataset_name>```can be ```kodak, clic20-pro-valid, clic22-test, jvet```.

For each dataset, a script is provided to decode all the bitstreams.

```bash
python3 results/decode_one_dataset.py <dataset_name> # Can take a few minutes
```

The file ```results/<dataset_name>/results.tsv``` provides the results that should be obtained.

Compared to HEVC (HM 16.20), Cool-chic bitstreams offer the following results:

| Dataset          | BD-rate vs. HEVC (HM 16.20) |
|------------------|-----------------------------|
| kodak            | <span style="color:#f50">+ 7.16 %  </span> |
| clic20-pro-valid | <span style="color:green">- 4.92 %  </span>                    |
| clic22-test      | <span style="color:green">- 4.46 %  </span>               |
| jvet             | <span style="color:#f50">+ 11.96 %  </span>              |


<br/>

# Compressing your own images

## Encoding

The script ```samples/encode_image.sh``` provides an example for encoding an image with Cool-chic:

```bash
python3 src/encode.py                               \
    --input=samples/biville.png                     \
    --output=samples/bitstream.bin                  \
    --model_save_path=samples/model.pt              \
    --enc_results_path=samples/encoder_results.txt  \
    --lmbda=0.0002                                  \
    --start_lr=1e-2                                 \
    --recipe=instant                                \
    --layers_arm=24,24                              \
    --n_ctx_rowcol=3                                \
    --latent_n_grids=7                              \
    --layers_synthesis=40-1-linear-relu,3-1-linear-relu,3-3-residual-relu,3-3-residual-none
```

| Argument                 | Role                                                                                                                                                                                                                                                                                                   | Suggested value(s)                                                         |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| ```--input```            | Image to compress                                                                                                                                                                                                                                                                                      | ```image.png```or ```image_<H>x<W>_1fps_yuv420_<bitdepth>b.yuv```          |
| ```--output```           | Output path for the bitstream                                                                                                                                                                                                                                                                          | ```image.bin```                                                            |
| ```--model_save_path```  | Save the overfitted model for analysis                                                                                                                                                                                                                                                                 | ```model.pt```                                                             |
| ```--enc_results_path``` | Save encoder logs                                                                                                                                                                                                                                                                                      | ```encoder_results.txt```                                                  |
| ```--lmbda```            | Rate constraint                                                                                                                                                                                                                                                                                        | From ```1e-4``` (high rate) to ```1e-2``` (low rate)                       |
| ```--start_lr```         | Initial learning rate                                                                                                                                                                                                                                                                                  | ```1e-2```                                                                 |
| ```--recipe```           | Encoder preset                                                                                                                                                                                                                                                                                         | ```instant, faster, fast, medium, slow, placebo```                         |
| ```--layers_arm```       | Dimension of hidden layers for the auto-regressive model                                                                                                                                                                                                                                               | ```24,24```                                                                |
| ```--n_ctx_rowcol```     | Number of rows & columns of context                                                                                                                                                                                                                                                                    | ```3```                                                                    |
| ```--latent_n_grids```   | Number of latent resolutions                                                                                                                                                                                                                                                                           | ```7```                                                                    |
| ```--layers_synthesis``` | Synthesis architecture, layers are separated with comas. |```40-1-linear-relu,3-1-linear-relu,3-3-residual-relu,3-3-residual-none```

The syntax for each layer of the synthesis is  is: ```<out_dim>-<kernel>-<type>-<non_linearity>```.  ```type``` is either```linear``` or ```residual```.  ```non_linearity``` is either ```relu```, ```leakyrelu``` or ```none```

## Decoding a bitstream

The script ```samples/decode_image.sh``` provides an example for decoding an image with Cool-chic:

```bash
python3 src/decode.py                               \
    --input=samples/bitstream.bin                   \
    --output=samples/biville_decoded.png
```

## YUV420 format

Cool-chic is able to process YUV420 data with 8-bit and 10-bit representation. To do so, you need to call the ```src/encode.py``` script with an appropriately formatted name such as:

    --input=<videoname>_<W>x<H>_<framerate>_yuv420_<bitdepth>b.yuv

The encoder codes only the first frame of the video.

# ⚠ Disclaimer ⚠

This repository is only a proof of concept illustrating how the ideas of Cool-chic could be translated into an actual codec.
The encoding process (i.e. learning the NNs and latent representation) might suffer from failures. In may help to do successive encoding and keep the best one or tweak the learning rate.

# Thanks

We used the [_constriction package_](https://github.com/bamler-lab/constriction) from Robert Bamler as our entropy coder. Many thanks for the great work!

  R. Bamler, Understanding Entropy Coding With Asymmetric Numeral Systems (ANS): a Statistician's Perspective, arXiv preprint [arXiv:2201.01741](https://arxiv.org/pdf/2201.01741.pdf).
