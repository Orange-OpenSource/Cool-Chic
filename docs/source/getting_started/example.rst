Example
=======

Cool-chic encodes each video frames successively through the
``cc_encode.py`` script.


Encoding an image into a .cool bitstream
""""""""""""""""""""""""""""""""""""""""

Encoding an image or a video frame with CoolChic requires to specify

* The input image path ``-i`` and output bitstream path ``-o``

* A working directory ``--workdir`` where logs are written.

* The number of training (encoding) iterations ``--n_itr``.

* The decoder configuration ``--dec_cfg_residue`` for the neural networks architecture.

* The rate constraint ``--lmbda`` setting the compressed file size.

.. code:: bash

    (venv) ~/Cool-Chic$ python cc_encode.py \
        -i=image.png \
        -o=./bitstream.cool \
        --workdir=./dummy_workdir \
        --dec_cfg_residue=cfg/dec/intra/hop.cfg \
        --n_itr=10000 \
        --lmbda=0.001    # Typical range is 1e-2 (low rate) to 1e-4 (high rate)

More details on encoding images with Cool-chic is available in the :doc:`encoder documentation <./../image_compression/overview>`.

.. _video_coding_example:

Encoding a video into a .cool bitstream
"""""""""""""""""""""""""""""""""""""""

Encoding a video requires encoding successively each video frames through the
``cc_encode.py`` script. We provide a ``samples/encode.py`` script
allowing to easily do video encoding.

.. code:: bash

    # Encode the first 33 frames of a video, with an intra at the beginning (0)
    # and at the end (-1)
    (venv) ~/Cool-Chic$ python samples/encode.py \
        -i myTestVideo_1920x1080_24p_yuv420_8b.yuv \
        -o myTestVideo_1920x1080_24p_yuv420_8b.cool \
        --workdir=./dummy_workdir \
        --lmbda=0.001 \
        --n_frames=33 \
        --intra_pos=0,-1


.. _decoding_example:

Decoding a .cool bitstream
""""""""""""""""""""""""""


Decoding an image or a video with CoolChic requires specifying the input and output paths.
We provide a few already encoded bitstreams, ready to be decoded in ``samples/bitstreams/``.

* The input bitstream path ``-i`` and the decoded image path ``-o``

.. code:: bash

    (venv) ~/Cool-Chic$ python cc_decode.py -i=samples/bitstream/kodim14.cool -o=decoded-kodim14.png

More details on decoding images with Cool-chic is available in the :doc:`decoder documentation <./../image_compression/decoding_images>`.