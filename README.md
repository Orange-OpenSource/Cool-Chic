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
    <a href="https://orange-opensource.github.io/Cool-Chic/getting_started/new_stuff.html">What's new in 5.0?</a>
    ·
    <a href="https://orange-opensource.github.io/Cool-Chic/results/image/compression_performance.html">Image coding performance</a>
    .
    <a href="https://orange-opensource.github.io/Cool-Chic/results/video/compression_performance.html">Video coding performance</a>
  </p>
</div>

<!-- # What's Cool-chic? -->

Cool-chic (pronounced <span class="ipa">/kul ʃik/</span> as in French 🥖🧀🍷) is
a low-complexity neural image and video codec based on overfitting. It achieves state-of-the-art image compression performance, while requiring only 1000 multiplications per decoded pixel.

<br>
<!-- <div align="center"> -->

### 🚀 Cool-chic 5.0: Better and faster overfitted image compression! 🚀

<!-- </div> -->

Cool-chic 5.0 focuses on image coding, implementing the contributions presented in the following paper: [_Cool-chic 5.0: Faster Encoding and Inter-Feature Entropy Modeling for Overfitted Image Compression_, Ladune et al.](https://arxiv.org/abs/2605.02726).

- **10x encoding speed-up** vs. Cool-chic 4.2
- Better image compression: **Rate decrease of -10%** versus Cool-chic 4.2 for identical quality
- Low decoding complexity of **2000 MAC / pixel**

Check-out the [release history](https://github.com/Orange-OpenSource/Cool-Chic/releases) to see previous versions of Cool-chic.

# Setup

See the [Cool-chic setup documentation](https://orange-opensource.github.io/Cool-Chic/getting_started/quickstart.html) for additional details

```bash
# Clone the Cool-chic source code
git clone https://github.com/Orange-OpenSource/Cool-Chic.git && cd Cool-Chic

# Install create and activate virtual env
sudo apt install -y pip
python3 -m pip install virtualenv
python3 -m virtualenv venv && source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Sanity check
python -m test.sanity_check
```

# Encoding

See the [example page](https://orange-opensource.github.io/Cool-Chic/getting_started/example.html) for more encoding examples. A comprehensive description of the encoder parameters are available for [image](https://orange-opensource.github.io/Cool-Chic/image_compression/preset.html) and [video](https://orange-opensource.github.io/Cool-Chic/video_compression/encoding.html) compression.


```bash
# Encode an image using the default configuration
python cc_encode.py -i image.png -o ./bitstream.cool

# Video requires to successively encode multiples frames.
python samples/encode.py -i myTestVideo_1920x1080_24p_yuv420_8b.yuv -o bitstream.cool
```

# Decoding

```bash
# Decoder outputs either PNG (image) or YUV (video) files
python cc_decode.py -i samples/bitstreams/kodim14.cool -o kodim14.png
```

<br>

# Image compression performance

  <table class="tg"><thead>
    <tr>
      <th class="tg-86ol" rowspan="2"></th>
      <th class="tg-86ol" colspan="6">BD-rate of Cool-chic 5.0.1 vs. [%]</th>
    </tr>
    <tr>
      <th class="tg-86ol"><a href="https://arxiv.org/abs/2001.01568" target="_blank" rel="noopener noreferrer">Cheng</a></th>
      <th class="tg-86ol"><a href="https://arxiv.org/abs/2203.10886" target="_blank" rel="noopener noreferrer">ELIC</a></th>
      <th class="tg-86ol"><a href="https://arxiv.org/abs/2307.15421" target="_blank" rel="noopener noreferrer">MLIC++</a></th>
      <th class="tg-86ol">Cool-chic 4.2 </th>
      <th class="tg-86ol">Cool-chic 5.0.0 </th>
      <th class="tg-86ol">VVC (VTM 28.3)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-86ol">kodak (RGB)</td>
      <td class="tg-qch7">-10.4 %</td>
      <td class="tg-xd3r">+0.1 %</td>
      <td class="tg-xd3r">+10.2%</td>
      <td class="tg-qch7">-6.1 %</td>
      <td class="tg-qch7">-1.2 %</td>
      <td class="tg-qch7">-3.9 % </td>
    </tr>
    <tr>
      <td class="tg-86ol">clic20-pro-valid (RGB)</td>
      <td class="tg-qch7">-21.3 %</td>
      <td class="tg-qch7">-9.5 %</td>
      <td class="tg-xd3r">+0.5 %</td>
      <td class="tg-qch7">-9.9 %</td>
      <td class="tg-qch7">-0.6 %</td>
      <td class="tg-qch7">-11.6%</td>
    </tr>
    <tr>
      <td class="tg-x9uu">jvet A (YUV420)</td>
      <td class="tg-1keu">/</td>
      <td class="tg-1keu">/</td>
      <td class="tg-1keu">/</td>
      <td class="tg-1keu">/</td>
      <td class="tg-qch7">-XX %</td>
      <td class="tg-qch7">-2.8 %</td>
    </tr>
    <tr>
      <td class="tg-x9uu">jvet B (YUV420) </td>
      <td class="tg-1keu">/</td>
      <td class="tg-1keu">/</td>
      <td class="tg-1keu">/</td>
      <td class="tg-qch7">-13.7%</td>
      <td class="tg-qch7">-1.5 %</td>
      <td class="tg-xd3r">+5.0 %</td>
    </tr>
  </tbody></table>


<br>

# Citation

If you use this project, please cite the Cool-chic 5.0 paper in your work.

```bibtex
@article{ladune2026-coolchic5-0-fasterencoding,
      title={Cool-chic 5.0: Faster Encoding and Inter-Feature Entropy Modeling for Overfitted Image Compression},
      author={Théo Ladune and Pierrick Philippe and Pierre Jaffuer and Théophile Blard and Sylvain Kervadec and Félix Henry and Gordon Clare},
      year={2026},
      eprint={2605.02726},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2605.02726},
}
```
<br>

# Acknowledgements

We rely on Robert Bamler's [constriction package](https://github.com/bamler-lab/constriction) for range coding. Thanks to the developers for their work!



<br>
<br>

<div align="center">
  <!-- User has no color preference: -->
  <img src="docs/source/assets/logo_orange.png" alt="Cool-chic Logo" height="150"/>
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
