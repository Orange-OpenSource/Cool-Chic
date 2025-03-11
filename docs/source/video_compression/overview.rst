Video compression with Cool-chic
================================



Cool-chic encodes each video frames successively through the
``coolchic/encode.py`` script. Each frame has either 0 reference (intra / plain
image) frame, 1 reference (P frame) or 2 references (B-frames).


Coding structure
""""""""""""""""

Cool-chic allows to perform all the usual coding configurations (hierarchical
random access, low-delay P). It is achieved by specifying:

  * ``--n_frames``: The number of frames to code

  * ``--intra_pos``: Where to place the intra frames

  * ``--p_pos``: Where to place the P frames

All the other frames are hierarchical B-frames with equidistant (past and
future) references. Below are a few examples. See :doc:`coding structure doc <./../../code_documentation/encoder/utils/codingstructure>` for more details.

.. code-block::

    # A low-delay P configuration
    # I0
    # \------> P1
    #             \-------> P2
    #                         \------> P3
    #                                     \-------> P4
    --n_frames=5 --intra_pos=0 --p_pos=1-4

    # A hierarchical Random Access configuration, with a closed GOP
    # I0
    # \-------------------------------------------------------------------------------------> P8
    # \----------------------------------------> B4 <----------------------------------------/
    # \-----------------> B2 <------------------/  \------------------> B6 <-----------------/
    # \------> B1 <------/  \-------> B3 <------/  \------> B5 <-------/  \------> B7 <------/
    --n_frames=8 --intra_pos=0 --p_pos=-1

    # A hierarchical Random Access configuration, with an open GOP
    # I0                                                                                      I8
    # \----------------------------------------> B4 <----------------------------------------/
    # \-----------------> B2 <------------------/  \------------------> B6 <-----------------/
    # \------> B1 <------/  \-------> B3 <------/  \------> B5 <-------/  \------> B7 <------/
    --n_frames=8 --intra_pos=0,-1

    # Or some very peculiar structures...
    # I0
    #   \---------------------------------------------------------------> P6
    #   \-----------------------------> B3 <-----------------------------/  \-----------------> P8
    #   \------> B1 <------------------/  \------> B4 <------------------/  \------> B7 <------/
    #              \------> B2 <-------/             \------> B5 <-------/
    --n_frames=8 --intra_pos=0 --p_pos=6,8


Encoding a video frame
""""""""""""""""""""""

To encode ``N`` frames of a video, it is required to successively call the
``coolchic/encode.py`` script ``N`` times. Always with the same coding structure
parameters while only varying the ``--coding_idx`` parameters to indicate which
frame is currently being encoded. All the references management logic is handle
by Cool-chic, based on the the coding index and the coding structure parameters.


.. code:: bash

    # In the hierarchical Random Access Configuration with open gop listed above
    # encode the 3rd (--coding_idx=2) frames in **coding** order: the B4 frame
    (venv) ~/Cool-Chic$ python coolchic/encode.py       \
        --input=path_to_my_example                      \
        --output=bitstream.bin                          \
        --workdir=./my_temporary_workdir/               \
        --enc_cfg=cfg/enc/inter/tunable .cfg            \
        --dec_cfg_residue=cfg/dec/intra_residue/mop.cfg \
        --dec_cfg_motion=cfg/dec/motion/lop.cfg         \
        --n_frames=8                                    \
        --intra_pos=0,-1                                \
        --coding_idx=2                                  \
        --n_itr=5000                                    \
        --lmbda=0.01 # Typical range is 1e-2 (low rate) to 1e-4 (high rate)

.. tips::

  Check the :ref:`video coding example script. <video_coding_example>`
  ``samples/encode.py`` to see how successive frames are encoded into frame-wise
  bitstream, which is subsequently concatenated into a single bitstream for the video.

File naming
"""""""""""

All the outputs of Cool-chic encoder (logs, compressed frames, trained models,
...) in the ``--workdir`` are prefixed by the frame **display index** *i.e.*
``0016-frame_encoder.pt``.
