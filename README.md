# 📢 __Cool-chic 3.0 is out!__ 📢

Cool-chic (pronounced <span class="ipa">/kul ʃik/</span> as in French 🥖🧀🍷) is a <b>low-complexity</b>  neural image codec based on __overfitting__. With only <b>2 000 multiplications / decoded pixel</b>, it offers coding performance close to VVC.

More information available on the [Cool-chic page](https://orange-opensource.github.io/Cool-Chic/)


# Version history

* Jan. 24: version 3.0
    - Re-implement most of the encoder-side improvements proposed by [_C3: High-performance and low-complexity neural compression from a single image or video_, Kim et al.](https://arxiv.org/abs/2312.02753)
    - 15% to 20% rate decrease compared to Cool-chic 2
* July 23: version 2 [_Low-complexity Overfitted Neural Image Codec_, Leguay et al.](https://arxiv.org/abs/2307.12706)
    - Several architecture changes for the decoder: convolution-based synthesis, learnable upsampling module
    - Friendlier usage: support for YUV 420 input format in 8-bit and 10-bit & Fixed point arithmetic for cross-platform entropy (de)coding
* March 23: version 1 [_COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec_, Ladune et al.](https://arxiv.org/abs/2212.05458)

Up to come: Cool-chic 3.1 will extend Cool-chic 3.0 to video coding.


# Cool-chic 3.0 performance

The table below presents the relative rate for identical PSNR of Cool-chic 3.0
against different anchors. A negative figure indicates that Cool-chic 3.0
requires less rate for the same quality. Note that we also provide all the
bitstreams required to reproduce these results.

| Dataset          | Vs. Cool-chic 2                              | Vs. Cool-chic 1                              | Vs. [_C3_, Kim et al.](https://arxiv.org/abs/2312.02753) | Vs. HEVC (HM 16.20)                          | Vs. VVC (VTM 19.1)                           |
|------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------------------|----------------------------------------------|----------------------------------------------|
| kodak            | <span style="color:green" > - 19.0 % </span> | <span style="color:green"> - 28.8 % </span>  | <span style="color:green"> - 1.2 %  </span>              | <span style="color:green" > - 14.5 % </span> | <span style="color:#f50" > + 6.6 %   </span> |
| clic20-pro-valid | <span style="color:green" > - 15.9 % </span> | <span style="color:gray"> /  </span>         | <span style="color:#f50" >+  4.4 %   </span>             | <span style="color:green" > - 20.6 % </span> | <span style="color:#f50" > + 3.4 %   </span> |
| jvet             | <span style="color:green" > - 21.4 % </span> | <span style="color:gray"> /  </span>         | <span style="color:gray"> /  </span>                     | <span style="color:green" > - 12.1 % </span> | <span style="color:#f50" > + 27.7 %  </span> |


# Setup

## ⚠️ Python version

Python version should be at least 3.10!

```bash
python3 --version                                          # Should be at least 3.10
```

## Necessary packages

```bash
python3 -m pip install virtualenv                          # Install virtual env if needed
python3 -m virtualenv venv && source venv/bin/activate     # Create and activate a virtual env named "venv"
(venv) pip install -r requirements.txt                     # Install the required packages
```

# Decoding images with Cool-chic

Already encoded files are provided as bitstreams in ```results/<dataset_name>/```where ```<dataset_name>```can be ```kodak, clic20-pro-valid, jvet```.

For each dataset, a script is provided to decode all the bitstreams.

```bash
(venv) python results/decode_one_dataset.py <dataset_name> # Can take a few minutes
```

The file ```results/<dataset_name>/results.tsv``` provides the results that should be obtained.

<br/>

# Encoding your own images

The script ```samples/encode_image.sh``` provides an example for encoding an image with Cool-chic:

```bash
(venv) python src/encode.py                         \
    --input=samples/biville.png                     \
    --output=samples/bitstream.bin                  \
    --workdir=./my_temporary_workdir/               \
    --lmbda=0.0002                                  \
    --start_lr=1e-2                                 \
    --layers_synthesis=40-1-linear-relu,3-1-linear-relu,X-3-residual-relu,X-3-residual-none \
    --upsampling_kernel_size=8                      \
    --layers_arm=24,24                              \
    --n_ctx_rowcol=3                                \
    --n_ft_per_res=1,1,1,1,1,1,1                    \
    --n_itr=1000                                    \
    --n_train_loops=1
```


| Argument                  | Role                                                      | Suggested value(s)                                                         |
|---------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------|
| ```--input```             | Image to compress                                         | ```image.png```or ```image_<H>x<W>_1fps_yuv420_<bitdepth>b.yuv```          |
| ```--output```            | Output path for the bitstream                             | ```image.bin```                                                            |
| ```--workdir```           | Output several logs into this directory logs              | ```./my_temporary_workdir/```                                              |
| ```--lmbda```             | Rate constraint                                           | From ```1e-4``` (high rate) to ```1e-2``` (low rate)                       |
| ```--start_lr```          | Initial learning rate                                     | ```1e-2```                                                                 |
| ```--layers_synthesis```  | Synthesis architecture, layers are separated with comas.  |```40-1-linear-relu,3-1-linear-relu,X-3-residual-relu,X-3-residual-none```  |
| ```--upsampling_kernel_size``` | Size of the learned upsampling kernel.               |```8```                                                                     |
| ```--layers_arm```        | Dimension of hidden layers for the auto-regressive model  | ```24,24```                                                                |
| ```--n_ctx_rowcol```      | Number of rows & columns of context                       | ```3```                                                                    |
| ```--n_ft_per_res```      | Number of feature(s) for each latent resolution           | ```1,1,1,1,1,1,1```                                                        |
| ```--n_itr```             | Maximum number of iteration per training phase            | From ```10000``` to ```100000```                                           |
| ```--n_train_loops```     | Number of independent encoding of the frame (the best one is kept)    | From ```1``` to ```5```                                        |

The syntax for each layer of the synthesis is  is: ```<out_dim>-<kernel>-<type>-<non_linearity>```.  ```type``` is either```linear``` or ```residual```.  ```non_linearity``` is either ```relu```, ```leakyrelu``` or ```none```. If the ```out_dim``` is set to ```X```, it is automatically replaced by the number of required output channels (i.e. 3 for a RGB or YUV image)

The encoding automatically selects the best device available (e.g. a GPU if there is one). Encoding on CPU can be quite slow...

The script ```samples/decode_image.sh``` provides an example for decoding an image with Cool-chic:

```bash
(venv) python src/decode.py                         \
    --input=samples/bitstream.bin                   \
    --output=samples/biville_decoded.png
```

## YUV420 format

Cool-chic is able to process YUV420 data with 8-bit and 10-bit representation. To do so, you need to call the ```src/encode.py``` script with an appropriately formatted name such as:

    --input=<videoname>_<W>x<H>_<framerate>_yuv420_<bitdepth>b.yuv

We only encode the first frame of the video.

# How does it work?

The source directory is organized as follows:

    src/
     |______ bitstream/                             [Functions to map Cool-chic to a binary file and vice versa     ]
     |______ encoding_management/                   [Contains training (i.e. encoding) hyperparameters              ]
     |______ models/                                [The different modules composing the encoder                    ]
     |          |______ coolchic_components/        [The different neural networks (ARM, upsampling and synthesis)  ]
     |          |______ coolchic_encoder.py         [Entire Cool-chic architecture i.e. ARM + upsampling + synthesis]
     |          |______ inter_coding_module.py      [Not used for image coding                                      ]
     |          |______ frame_encoder.py            [A coolchic encoder followed by a inter_coding_module           ]
     |          |______ video_encoder.py            [One frame_encoder for each frame of the video to compress      ]
     |______ utils/                                 [Utilities                                                      ]
     |______ visu/                                  [Helper to visualize the inner behavior of Cool-chic            ]


# ⚠ Disclaimer

This repository is only a proof of concept illustrating how the ideas of Cool-chic could be translated into an actual codec.
The encoding process (i.e. learning the NNs and latent representation) might suffer from failures. It may help to do successive encoding and keep the best one or to tweak the learning rate.

# Thanks

Special thanks go to:

* Robert Bamler for the [_constriction package_](https://github.com/bamler-lab/constriction) which serves as our entropy coder. More details @ [_Understanding Entropy Coding With Asymmetric Numeral Systems (ANS): a Statistician's Perspective_, Bamler](https://arxiv.org/pdf/2201.01741.pdf).
* Hyunjik Kim, Matthias Bauer, Lucas Theis, Jonathan Richard Schwarz and Emilien Dupont for their great work enhancing Cool-chic: [_C3: High-performance and low-complexity neural compression from a single image or video_, Kim et al.](https://arxiv.org/abs/2312.02753)
