Decoding images with Cool-chic
===============================

In order to offer fast decoding, Cool-chic relies on a C API re-implementing
efficiently all the components of the decoder *i.e* ARM, Upsampling, Synthesis.

The following script is used to decode image with Cool-chic. We provide a few
already encoded bitstreams, ready to be decoded in ``samples/bitstreams/``.


.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/decode.py \
        -i=samples/bitstream/kodim14.cool \
        -o=decoded-kodim14.ppm \
        --verbosity=1

Note that Cool-Chic outputs either `PPM
<https://en.wikipedia.org/wiki/Portable_pixmap>`_ (and not PNG!) or `YUV
<https://en.wikipedia.org/wiki/Y%E2%80%B2UV>`_ files.
