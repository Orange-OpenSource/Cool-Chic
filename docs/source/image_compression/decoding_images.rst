Decoding images with Cool-chic
===============================

In order to offer fast decoding, Cool-chic relies on a C API re-implementing
efficiently all the components of the decoder *i.e* ARM, Upsampling, Synthesis.

The following script is used to decode image with Cool-chic:

.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/decode.py -i=bitstream.cool -o=decoded_image.ppm

.. attention::

    By default, Cool-chic decoder leverages AVX2 CPU instructions for faster
    decoding. If these instructions not available on your device, use the
    ``--no_avx2`` argument.

Note that Cool-Chic outputs either `PPM
<https://en.wikipedia.org/wiki/Portable_pixmap>`_ (and not PNG!) or `YUV
<https://en.wikipedia.org/wiki/Y%E2%80%B2UV>`_ files.
