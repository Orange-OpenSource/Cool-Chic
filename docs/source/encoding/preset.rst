Encoder configuration
=====================

As with conventional codecs, there is a trade-off between Cool-chic encoding
time and compression performance. The encoding settings of Cool-chic are set in
a configuration file. Examples of such configuration files are located in ``cfg/enc/``.
They include the following parameters:

.. list-table:: Parameters relative to the encoder configuration
   :widths: 25 50 25
   :header-rows: 1

   * - Parameter
     - Role
     - Example value
   * - ``recipe``
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
          --enc_cfg=example.cfg   # example.cfg has start_lr=1e-2
          --start_lr=1e-3         # This override the value present in example.cfg

Some existing configuration files
"""""""""""""""""""""""""""""""""

Some configuration files are proposed in ``cfg/enc/``:

.. list-table:: Existing encoder configuration files.
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - ``fast.cfg``
     - Reasonable compression performance & fast training
   * - ``medium.cfg``
     - Balance compression performance & training duration
   * - ``slow.cfg``
     - Best performance at the cost of a longer training

Recipes
"""""""

Cool-chic encoding works with tweakable recipes *i.e.* different training
parameters. Currently available recipes are:

* ``c3x`` Inspired by `C3: High-performance and low-complexity neural
  compression from a single image or video, Kim et al
  <https://arxiv.org/abs/2312.02753>`_. It is composed of two main phases:

    1. additive noise model and softround for the latent quantization

    2. Actual quantization with softround in the backward

* ``debug`` Extremely fast preset with very bad performance only for debugging purposes.

All recipes feature a decreasing learning rate starting from ``start_lr``.

The number of iterations in the first (and longest) phase of the ``c3x`` recipe is
set using ``n_itr``.

In order to circumvent some training irregularities, it is possible to perform
several independent encoding, keeping only the best one. We call that a training
loop. The number of training loops is set by ``n_train_loops``.


