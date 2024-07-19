Overview
========

Encoding your own image or video is achieved by using the script ``coolchic/encode.py``.

.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/encode.py             \
        --input=path_to_my_example                      \
        --output=bitstream.bin                          \
        --workdir=./my_temporary_workdir/               \
        --enc_cfg=cfg/enc/fast.cfg                      \
        --dec_cfg=cfg/dec/mop.cfg

Unlike the decoding script which only takes input and output arguments, the
encoder has many arguments allowing to tune Cool-chic for your need.

* :doc:`Encoder configuration  <./preset>` affects the encoding duration by
  changing the training parameters. This is set through the argument
  ``--enc_cfg``. Several encoder configuration files are available in ``cfg/enc/``.

* :doc:`Decoder configuration <./architecture>` parametrizes the decoder
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

Cool-chic is able to encode PNG files and YUV 420 files. The naming of YUV files
must comply with the following convention

.. code:: bash

    --input=<videoname>_<Width>x<Height>_<framerate>p_yuv420_<bitdepth>b.yuv

Note that Cool-chic support both 8-bit and 10-bit YUV file. Support for YUV 444
is not yet implemented.

Note that Cool-Chic outputs either `PPM
<https://en.wikipedia.org/wiki/Portable_pixmap>`_ (and not PNG!) or `YUV
<https://en.wikipedia.org/wiki/Y%E2%80%B2UV>`_ files.


