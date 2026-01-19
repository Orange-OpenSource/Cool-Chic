[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![BSD-3 License][license-shield]][license-url]
[![PyTorch][pytorch-shield]][pytorch-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<picture>
  <!-- User prefers light mode: -->
  <source srcset="docs/source/assets/coolchic-logo-light.png" media="(prefers-color-scheme: light)" alt="Cool-chic Logo" height="200"/>

  <!-- User prefers dark mode: -->
  <source srcset="docs/source/assets/coolchic-logo-dark.png"  media="(prefers-color-scheme: dark)" alt="Cool-chic Logo" height="200"/>

  <!-- User has no color preference: -->
  <img src="docs/source/assets/coolchic-logo-dark.png" alt="Cool-chic Logo" height="200"/>
</picture>
  <p align="center">
    <!-- Low-complexity neural image codec based on overfitting. -->
    <br />
    <a href="https://orange-opensource.github.io/Cool-Chic/"><strong>Explore the docs </strong></a>
    <!-- <br />
    <br /> -->
    .
    <a href="https://orange-opensource.github.io/Cool-Chic/getting_started/new_stuff.html">What's new in 4.2.0?</a>
    ·
    <a href="https://orange-opensource.github.io/Cool-Chic/results/image/compression_performance.html">Image coding performance</a>
    .
    <a href="https://orange-opensource.github.io/Cool-Chic/results/video/compression_performance.html">Video coding performance</a>
  </p>
</div>

<!-- # What's Cool-chic? -->

Cool-chic (pronounced <span class="ipa">/kul ʃik/</span> as in French 🥖🧀🍷) is
a low-complexity neural image codec based on overfitting.

* 🏆 **Coding performance**: Cool-chic offers visual quality similar to H.266/VVC with 30% less rate

* 🪶 **Lightweight decoder**: Cool-chic decoder performs only 1000 multiplications per decoded pixel

* 🚀 **Fast CPU-only decoder**: Decode a 1280x720 image in 100 ms on CPU with our decoder written in C

<br>
<!-- <div align="center"> -->

### 🎲 Cool-chic 4.2.0: Common Randomness & Wasserstein Distance! 🎲

<!-- </div> -->

Cool-chic 4.2 focuses on perceptually-oriented image coding. This release draws
heavily on the following paper: [_Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion_, Ballé et al.](https://arxiv.org/abs/2412.00505).

- **Wasserstein Distance** as a distortion metric: use ``--tune=wasserstein``
- Decoder-side **common randomness** for additional details in the decoded image
- Improved image coding performance: around **-50% rate** versus Cool-chic 4.1 for the same visual quality
- Low decoding complexity **1728 MAC / pixel**


A pair of bitstreams in ``samples/bitstreams/`` illustrates the benefits of the
``--tune=wasserstein`` options. See the [decoding
example](https://orange-opensource.github.io/Cool-Chic/getting_started/example.html#decoding-a-cool-bitstream)
to decode them and see the pictures.


Check-out the [release history](https://github.com/Orange-OpenSource/Cool-Chic/releases) to see previous versions of Cool-chic.

# Setup

See the [Cool-chic setup documentation](https://orange-opensource.github.io/Cool-Chic/getting_started/quickstart.html) for additional details

```bash
# We need to get these packages to compile the C API and bind it to python.
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y build-essential python3.10-dev pip g++
git clone https://github.com/Orange-OpenSource/Cool-Chic.git && cd Cool-Chic

# Install create and activate virtual env
python3.10 -m pip install virtualenv
python3.10 -m virtualenv venv && source venv/bin/activate

# Install Cool-chic
pip install -e .

# Sanity check
python -m test.sanity_check
```

# Encoding

See the [example page](https://orange-opensource.github.io/Cool-Chic/getting_started/example.html) for more encoding examples. A comprehensive description of the encoder parameters are available for [image](https://orange-opensource.github.io/Cool-Chic/image_compression/preset.html) and [video](https://orange-opensource.github.io/Cool-Chic/video_compression/encoding.html) compression.


```bash
# Encode an image using the default (fast) configuration
python coolchic/encode.py -i image.png -o ./bitstream.cool

# Video requires to successively encode multiples frames.
python samples/encode.py -i myTestVideo_1920x1080_24p_yuv420_8b.yuv -o bitstream.cool
```

# Decoding

Call the C decoder through a python wrapper to decode Cool-chic bitstreams.

```bash
# Decoder outputs either PPM (image) or YUV (video) files
python coolchic/decode.py -i samples/bitstreams/a365_wd.cool -o a365_wd.ppm
```

<br>

# Citation

If you use this project, please cite the original Cool-chic paper in your work.

```bibtex
@InProceedings{Ladune_2023_ICCV,
    author    = {Ladune, Th\'eo and Philippe, Pierrick and Henry, F\'elix and Clare, Gordon and Leguay, Thomas},
    title     = {COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13515-13522}
}
```

<br>
<br>

<picture>
  <!-- User has no color preference: -->
  <img src="docs/source/assets/logo_orange.png" alt="Cool-chic Logo" height="150"/>
</picture>
</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Orange-OpenSource/Cool-Chic.svg?style=for-the-badge
[contributors-url]: https://github.com/Orange-OpenSource/Cool-Chic/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Orange-OpenSource/Cool-Chic.svg?style=for-the-badge
[forks-url]: https://github.com/Orange-OpenSource/Cool-Chic/network/members
[stars-shield]: https://img.shields.io/github/stars/Orange-OpenSource/Cool-Chic.svg?style=for-the-badge
[stars-url]: https://github.com/Orange-OpenSource/Cool-Chic/stargazers
[issues-shield]: https://img.shields.io/github/issues/Orange-OpenSource/Cool-Chic.svg?style=for-the-badge
[issues-url]: https://github.com/Orange-OpenSource/Cool-Chic/issues
[license-shield]: https://img.shields.io/github/license/Orange-OpenSource/Cool-Chic.svg?style=for-the-badge
[license-url]: https://github.com/Orange-OpenSource/Cool-Chic/blob/master/LICENSE.txt
[pytorch-shield]: https://img.shields.io/badge/PyTorch-0769AD?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
