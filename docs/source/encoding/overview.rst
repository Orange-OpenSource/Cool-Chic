Overview
========

Encoding your own image or video is achieved by using the script ``coolchic/encode.py``.

.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/encode.py       \
        --input=path_to_my_example                      \
        --output=bitstream.bin                          \
        --workdir=./my_temporary_workdir/               \
        --enc_cfg=cfg/enc/fast_10k.cfg                  \
        --dec_cfg=cfg/dec/mop.cfg                       \
        --lmbda=0.001 # Typical range is 1e-2 (low rate) to 1e-4 (high rate)

Unlike the decoding script which only takes input and output arguments, the
encoder has many arguments allowing to tune Cool-chic for your need.

* :ref:`Encoder configuration <encoder_cfg_files>` affects the encoding duration by
  changing the training parameters. This is set through the argument
  ``--enc_cfg``. Several encoder configuration files are available in ``cfg/enc/``.

* :ref:`Decoder configuration <decoder_cfg_files>` parametrizes the decoder
  architecture and complexity. This is set through the argument ``--dec_cfg``.
  Several encoder configuration files are available in ``cfg/dec/``.

Working directory
"""""""""""""""""

The ``--workdir`` argument is used to specify a folder where all necessary data will be stored.
This includes the encoder logs and the PyTorch model (``workdir/video_encoder.pt``).

.. attention::

  If present, the PyTorch model inside workdir ``workdir/video_encoder.pt`` is reloaded
  by the ``coolchic/encode.py`` script. In order to encode
  a new image using the same workdir, you must first clean out the workdir.

I/O format
""""""""""

Cool-chic is able to encode PPM, PNG, YUV420 & YUV 444 files. The naming of YUV files
must comply with the following convention

.. code:: bash

    --input=<videoname>_<Width>x<Height>_<framerate>p_yuv<chromasampling>_<bitdepth>b.yuv

Note that Cool-Chic outputs either `PPM
<https://en.wikipedia.org/wiki/Portable_pixmap>`_ (and not PNG!) or `YUV
<https://en.wikipedia.org/wiki/Y%E2%80%B2UV>`_ files.

Rate constraint
"""""""""""""""

The rate constraint ``--lmbda`` is used to balance the rate and the distortion when encoding an image.
Indeed, Cool-chic parameters are optimized through gradient descent according to the following rate-distortion objective:

.. math::

    \mathcal{L} = ||\mathbf{x} - \hat{\mathbf{x}}||^2 + \lambda
    (\mathrm{R}(\hat{\mathbf{x}})), \text{ with }
    \\begin{cases}
        \mathbf{x} & \text{the original image}\\ \hat{\mathbf{x}} &
        \text{the coded image}\\ \mathrm{R}(\hat{\mathbf{x}}) &
        \text{A measure of the rate of } \hat{\mathbf{x}}\\
        \lmbda & \text{the rate constraint }\texttt{--lmbda} 
    \\end{cases}
