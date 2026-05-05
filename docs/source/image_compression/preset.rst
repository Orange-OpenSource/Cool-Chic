Encoder configuration
=====================

As with conventional codecs, there is a trade-off between Cool-chic encoding
time and compression performance. The encoding duration, initial learning rate
and distortion metrics can be changed to accommodate your needs.

.. list-table:: Parameters relative to the encoder configuration
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Role
     - Example value
   * - ``start_lr``
     - Initial learning rate
     - ``1e-2``
   * - ``n_itr``
     - Number of training iterations
     - ``1e4``
   * - ``tune``
     - Optimize the MSE (``mse``) or the Wasserstein Distance (``wasserstein``)
     - ``mse``


.. _encoder_cfg_files:

Tuning
""""""

The tuning parameters ``--tune`` allows selecting the distortion metric(s) to be
optimized. When the mode ``--tune=mse`` is selected, the Mean Squared Error is
optimized. When ``--tune=wasserstein`` the distortion becomes a combination of
MSE and Wasserstein Distance, as proposed in `Good, Cheap, and Fast: Overfitted
Image Compression with Wasserstein Distortion, Ballé et al
<https://arxiv.org/abs/2412.00505>`_.

.. attention::

    Using ``--tune=wasserstein`` also introduces additional common randomness
    features, as described in the aforementioned Ballé's paper.
