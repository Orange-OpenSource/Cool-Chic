Cool-chic literature
====================

There are several published papers describing and enhancing Cool-chic. Reading
some of these papers might help understanding how Cool-chic works.

Used in this repo
"""""""""""""""""

This repository is based on several papers successively enhancing Cool-chic. Check-out the `release history <https://github.com/Orange-OpenSource/Cool-Chic/releases>`_ to see previous versions of Cool-chic.


2024
****

* November: `Upsampling Improvement for Overfitted Neural Coding, Philippe et al <https://arxiv.org/abs/2411.19249>`_

  * Replace the upsampling with separable and symmetrical filters for adaptive, lighter and more efficient latent upsampling. Corresponds to Cool-chic 3.4

  * ``git clone https://github.com/Orange-OpenSource/Cool-Chic.git``


* May: `Overfitted image coding at reduced complexity, Blard et al <https://arxiv.org/abs/2403.11651>`_

  * Study the complexity-performance trade-off of Cool-chic & introduce a near real-time CPU only decoder. Corresponds to Cool-chic 3.2 & Cool-chic 3.3

  * ``git clone --branch v3.3 https://github.com/Orange-OpenSource/Cool-Chic.git``

* February: `Cool-chic video: Learned video coding with 800 parameters, Leguay et al <https://arxiv.org/abs/2402.03179>`_

  * Corresponds to Cool-chic 3.1 inter-frame coding scheme.

  * ``git clone --branch v3.1 https://github.com/Orange-OpenSource/Cool-Chic.git``

2023
****

* December: `C3: High-performance and low-complexity neural compression from a single image or video, Kim et al <https://arxiv.org/abs/2312.02753>`_

  * Some improvements related to image coding have been integrated in Cool-chic 3.0

  * ``git clone --branch v3.0 https://github.com/Orange-OpenSource/Cool-Chic.git``


* July:  `Low-complexity Overfitted Neural Image Codec, Leguay et al. <https://arxiv.org/abs/2307.12706>`_

  * This constitutes Cool-chic 2

  * ``git clone --branch v2.0 https://github.com/Orange-OpenSource/Cool-Chic.git``

2022
****

* December: `COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec, Ladune et al. <https://arxiv.org/abs/2212.05458>`_

  * This constitutes Cool-chic 1

  * ``git clone --branch v1.0 https://github.com/Orange-OpenSource/Cool-Chic.git``


Other papers
""""""""""""

Some other papers have enhanced or adapted Cool-chic to particular use-cases but
have not (yet?) been integrated  in this repository.

2024
****

* December: `Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion, Ballé et al. <https://arxiv.org/abs/2412.00505>`_

  * Train Cool-chic with a Wasserstein Distance-based distortion function for better subjective results.

* October: `Redefining Visual Quality: The Impact of Loss Functions on INR-Based Image Compression, Catania et al. <https://ieeexplore.ieee.org/abstract/document/10647328>`_

  * Encode images with Cool-chic using different training loss more aligned with the perceptive metrics.

* April: `Human–Machine Collaborative Image Compression Method Based on Implicit Neural Representations, Li et al. <https://ieeexplore.ieee.org/document/10323534>`_

  * Propose to adapt Cool-chic to accommodate simultaneous coding for human and machine.

* January: `Fast Implicit Neural Representation Image Codec in Resource-limited Devices, Liu et al. <https://arxiv.org/abs/2401.12587>`_

  * Faster Cool-chic decoder

* January: `Cool-Chic: Perceptually Tuned Low Complexity Overfitted Image Coder, Ladune et al <https://arxiv.org/abs/2401.02156>`_

  * Cool-chic-based submission to the `CLIC24 challenge <https://compression.cc/>`_

2023
****

* December: `Hybrid Implicit Neural Image Compression with Subpixel Context Model and Iterative Pruner, Li et al. <https://ieeexplore.ieee.org/abstract/document/10402791>`_

  * Speed-up cool-chic decoding by removing the auto-regressive probability model

* November: `Enhanced Quantified Local Implicit Neural Representation for Image Compression, Zhang et al. <https://ieeexplore.ieee.org/document/10323534>`_

  * Propose a CNN-based post-filter (similar to Cool-chic 2.0) & a SGA-based quantization close to the softround mechanism introduced by C3

* September: `Implicit Neural Multiple Description for DNA-based data storage, Le et al. <https://arxiv.org/abs/2309.06956>`_

  * Adapt Cool-chic to encode/decode image on DNA instead of binary files

* June: `INR-MDSQC: Implicit Neural Representation Multiple Description Scalar Quantization for robust image Coding, Le et al. <https://arxiv.org/abs/2306.13919>`_

  * Add multiple description into Cool-chic to cope with noisy transmission channels
