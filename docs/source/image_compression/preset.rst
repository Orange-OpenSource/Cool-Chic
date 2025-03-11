Encoder configuration
=====================

As with conventional codecs, there is a trade-off between Cool-chic encoding
time and compression performance. The encoding settings of Cool-chic are set in
a configuration file. Examples of such configuration files are located in ``cfg/enc/intra/``.
They include the following parameters:

.. list-table:: Parameters relative to the encoder configuration
   :widths: 25 50 25
   :header-rows: 1

   * - Parameter
     - Role
     - Example value
   * - ``preset``
     - Training preset
     - ``c3x``
   * - ``start_lr``
     - Initial learning rate
     - ``1e-2``
   * - ``n_itr``
     - Number of training iterations
     - ``1e4```
   * - ``n_train_loops``
     - Number of independent encodings
     - ``1```

.. tip::

    Each parameter listed in the configuration file can be overridden through a
    command line argument:

    .. code:: bash

        (venv) ~/Cool-Chic python coolchic/encode.py \
          --enc_cfg=enc/cfg/intra_fast_10k.cfg   # fast_10k.cfg has start_lr=1e-2
          --start_lr=1e-3                        # This override the value present in fast_10k.cfg

.. _encoder_cfg_files:
Some existing configuration files
"""""""""""""""""""""""""""""""""

Some configuration files are proposed in ``cfg/enc/intra/``. Longer encoding gives
slightly better compression results. We provide comprehensive results for all
encoding configurations on :doc:`the encoding complexity page <../results/image/encoding_complexity>`.

.. list-table:: Existing encoder configuration files.
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - ``fast_10k.cfg``
     - Reasonable compression performance & fast training
   * - ``medium_30k.cfg``
     - Balance compression performance & training duration
   * - ``slow_100k.cfg``
     - Best performance at the cost of a longer training

Presets
"""""""

Cool-chic encoding works with tweakable presets *i.e.* different training
parameters. Currently available presets are:

* ``c3x_intra`` Inspired by `C3: High-performance and low-complexity neural
  compression from a single image or video, Kim et al
  <https://arxiv.org/abs/2312.02753>`_. It is composed of two main phases:

    1. additive noise model and softround for the latent quantization

    2. Actual quantization with softround in the backward

* ``debug`` Extremely fast preset with very bad performance only for debugging purposes.

All presets feature a decreasing learning rate starting from ``start_lr``.

The number of iterations in the first (and longest) phase of the ``c3x`` preset is
set using ``n_itr``.

In order to circumvent some training irregularities, it is possible to perform
several independent encoding, keeping only the best one. We call that a training
loop. The number of training loops is set by ``n_train_loops``.


