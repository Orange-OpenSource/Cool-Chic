Cool-chic literature
====================

There are several published papers describing and enhancing Cool-chic. Reading
some of these papers might help understanding how Cool-chic works.

Used in this repo
"""""""""""""""""

This repository is based on several papers successively enhancing Cool-chic. Check-out the `release history <https://github.com/Orange-OpenSource/Cool-Chic/releases>`_ to see previous versions of Cool-chic.


2024
****


* May: `Overfitted image coding at reduced complexity, Blard et al <https://arxiv.org/abs/2403.11651>`_

  * Study the complexity-performance trade-off of Cool-chic & introduce a near real-time CPU only decoder

  * ``git clone https://github.com/Orange-OpenSource/Cool-Chic.git``

* February: `Cool-chic video: Learned video coding with 800 parameters, Leguay et al <https://arxiv.org/abs/2402.03179>`_

  * The inter-frame coding scheme used in the current main branch

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

* January: `Fast Implicit Neural Representation Image Codec in Resource-limited Devices, Liu et al. <https://arxiv.org/abs/2401.12587>`_

  * Faster Cool-chic decoder

* January: `Cool-Chic: Perceptually Tuned Low Complexity Overfitted Image Coder, Ladune et al <https://arxiv.org/abs/2401.02156>`_

  * Cool-chic-based submission to the `CLIC24 challenge <https://compression.cc/>`_

2023
****

* December: `Hybrid Implicit Neural Image Compression with Subpixel Context Model and Iterative Pruner, Li et al. <https://ieeexplore.ieee.org/abstract/document/10402791>`_

  * Speed-up cool-chic decoding by removing the auto-regressive probability model

* September: `Implicit Neural Multiple Description for DNA-based data storage, Le et al. <https://arxiv.org/abs/2309.06956>`_

  * Adapt Cool-chic to encode/decode image on DNA instead of binary files

* June: `INR-MDSQC: Implicit Neural Representation Multiple Description Scalar Quantization for robust image Coding, Le et al. <https://arxiv.org/abs/2306.13919>`_

  * Add multiple description into Cool-chic to cope with noisy transmission channels
