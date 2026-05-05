.. _image_compression_overview:

Image compression with Cool-chic
================================

Encoding your own image is achieved by using the script ``cc_encode.py``.

.. code:: bash

    (venv) ~/Cool-Chic$ python cc_encode.py \
        --input=path_to_my_example                \
        --output=bitstream.cool                   \
        --workdir=./my_temporary_workdir/         \
        --n_itr=10000                             \
        --dec_cfg_residue=cfg/dec/intra/mop.cfg   \
        --lmbda=0.001 # Typical range is 1e-2 (low rate) to 1e-4 (high rate)

Unlike the decoding script which only takes input and output arguments, the
encoder has many arguments allowing to tune Cool-chic for your need.

* :ref:`Decoder configuration <decoder_cfg_files>` parametrizes the decoder
  architecture and complexity. This is set through the argument ``--dec_cfg``.
  Several encoder configuration files are available in ``cfg/dec/``.

Working directory
"""""""""""""""""

The ``--workdir`` argument is used to specify a folder where all necessary data
will be stored. This includes the encoder logs and the PyTorch model
(``workdir/0000-frame_encoder.pt``).

.. attention::

  If present, the PyTorch model inside workdir ``workdir/0000-frame_encoder.pt``
  is reloaded by the ``cc_encode.py`` script. In order to encode a new
  image using the same workdir, you must first clean out the workdir.

I/O format
""""""""""

Cool-chic is able to encode PPM, PNG, YUV420 & YUV 444 files. The naming of YUV
files must comply with the following convention.

.. code:: bash

    --input=<videoname>_<Width>x<Height>_<framerate>p_yuv<chromasampling>_<bitdepth>b.yuv

Rate constraint
"""""""""""""""

The rate constraint ``--lmbda`` is used to balance the rate and the distortion when encoding an image.
Indeed, Cool-chic parameters are optimized through gradient descent according to the following rate-distortion objective:

.. math::

    \mathcal{L} = ||\mathbf{x} - \hat{\mathbf{x}}||^2 + \lambda
    (\mathrm{R}(\hat{\mathbf{x}})), \text{ with }
    \begin{cases}
        \mathbf{x} & \text{the original image}\\ \hat{\mathbf{x}} &
        \text{the coded image}\\ \mathrm{R}(\hat{\mathbf{x}}) &
        \text{a measure of the rate of } \hat{\mathbf{x}}\\
        \lambda & \text{the rate constraint }\texttt{--lmbda} 
    \end{cases}