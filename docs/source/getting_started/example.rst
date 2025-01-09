Example
=======

Encoding an image into a ``.cool`` bitstream
""""""""""""""""""""""""""""""""""""""""""""

Encoding an image with CoolChic requires to specify

* The input image path ``-i`` and output bitstream path ``-o``

* A working directory ``--workdir`` where logs are written.

* The encoder configuration ``--enc_cfg`` for the training options.

* The decoder configuration ``--dec_cfg`` for the neural networks architecture.

* The rate constraint ``--lmbda`` setting the compressed file size.

.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/encode.py \
        -i=image.png -o=./bitstream.cool --workdir=./dummy_workdir \
        --enc_cfg=cfg/enc/fast_10k.cfg --dec_cfg=cfg/dec/lop.cfg \
        --lmbda=0.001    # Typical range is 1e-2 (low rate) to 1e-4 (high rate)

More details on encoding images with Cool-chic is available in the :doc:`encoder documentation <./../encoding/overview>`.


Decoding a ``.cool`` bitstream
""""""""""""""""""""""""""""""


Decoding an image with CoolChic requires to specify

* The input bitstream path ``-i`` and the decoded image path ``-o``

.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/decode.py -i=bitstream.cool -o=decoded_image.ppm

Note that Cool-Chic outputs either `PPM
<https://en.wikipedia.org/wiki/Portable_pixmap>`_ or `YUV
<https://en.wikipedia.org/wiki/Y%E2%80%B2UV>`_ files.

More details on decoding images with Cool-chic is available in the :doc:`decoder documentation <./../decoding/decoding_images>`.
