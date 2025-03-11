Example
=======

Cool-chic encodes each video frames successively through the
``coolchic/encode.py`` script. An image is simply a video with a single frame.


Encoding an image into a .cool bitstream
""""""""""""""""""""""""""""""""""""""""""""

Encoding an image or a video frame with CoolChic requires to specify

* The input image path ``-i`` and output bitstream path ``-o``

* A working directory ``--workdir`` where logs are written.

* The encoder configuration ``--enc_cfg`` for the training options.

* The decoder configuration ``--dec_cfg_residue`` for the neural networks architecture.

* The rate constraint ``--lmbda`` setting the compressed file size.

.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/encode.py \
        -i=image.png \
        -o=./bitstream.cool \
        --workdir=./dummy_workdir \
        --enc_cfg=cfg/enc/intra/fast_10k.cfg \
        --dec_cfg_residue=cfg/dec/intra_residue/hop.cfg \
        --lmbda=0.001    # Typical range is 1e-2 (low rate) to 1e-4 (high rate)

More details on encoding images with Cool-chic is available in the :doc:`encoder documentation <./../encoding/overview>`.

.. _video_coding_example:
Encoding a video into a .cool bitstream
"""""""""""""""""""""""""""""""""""""""""""

Encoding a video requires to encode successively each video frames through the
``coolchic/encode.py`` script. We provide a ``samples/encode.py`` script
allowing to easily do video encoding.

.. code:: bash

    # Encode the first 33 frames of a video, with an intra at the beginning (0)
    # and at the end (-1)
    (venv) ~/Cool-Chic$ python samples/encode.py \
        -i myTestVideo_1920x1080_24p_yuv420_8b.yuv \
        -o myTestVideo_1920x1080_24p_yuv420_8b.cool \
        --workdir=./dummy_workdir \
        --lambda=0.001 \
        --n_frames=33 \
        --intra_pos=0,-1


Decoding a .cool bitstream
""""""""""""""""""""""""""""""


Decoding an image or a video with CoolChic requires to specify the input and output paths.
We provide a few already encoded bitstreams, ready to be decoded in ``samples/bitstreams/``.

* The input bitstream path ``-i`` and the decoded image path ``-o``

.. code:: bash

    (venv) ~/Cool-Chic$ python coolchic/decode.py \
        -i=samples/bitstream/kodim14.cool \
        -o=decoded-kodim14.ppm

Note that Cool-Chic outputs either `PPM
<https://en.wikipedia.org/wiki/Portable_pixmap>`_ or `YUV
<https://en.wikipedia.org/wiki/Y%E2%80%B2UV>`_ files.

More details on decoding images with Cool-chic is available in the :doc:`decoder documentation <./../decoding/decoding_images>`.