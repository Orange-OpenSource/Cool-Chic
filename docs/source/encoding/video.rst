Video coding configuration
==========================

.. attention::

   🛑 Cool-chic 3.2 and 3.3 temporarily disable video coding. If you really want to
   compress videos you can

   * Go back to 3.1: ``git clone --branch v3.1
     https://github.com/Orange-OpenSource/Cool-Chic.git``

   * Wait for Cool-chic 4.0 for better and faster video coding 😉.


.. The video coding configuration is set through the following arguments

.. * ``--intra_period``

.. * ``--p_period``


.. Cool-chic is a video codec implementing 3 different type of frames

.. * I (intra) frames have no reference. They can be used for image coding

.. * P (inter) frames have a single reference.

.. * B (inter) frames have two references


.. Cool-chic always encodes a Group Of Pictures (GOP), consisting in 1 intra
.. frame followed by 0 to 255 inter frame(s).

.. .. tip::

..     Coding an image is achieved by using a single-frame GOP with only an intra frame.


.. Intra period
.. """"""""""""

.. A GOP starts with an intra frame and followed by an arbitrary number of inter (P
.. or B) frames. The number of frames in the GOP and the number of frames encoded
.. by Cool-chic is thus: intra period + 1:

.. .. code:: bash

..     # This codes 9 frames from the video
..     # number_of_coded_frames = number_of_frames_in_gop = intra_period + 1
..     (venv) python3 src/encode.py --intra_period=8 --input=xxx.yuv


.. P-period
.. """"""""

.. Both GOPs below have the same ``--intra_period=8`` but have a different ``--p_period`` 

.. .. code:: bash

..     # A low-delay P configuration
..     # I0 ---> P1 ---> P2 ---> P3 ---> P4 ---> P5 ---> P6 ---> P7 ---> P8
..     python3 src/encode.py --intra_period=8 --p_period=1 --input=xxx.yuv

..     # A hierarchichal Random Access configuration
..     # I0 -----------------------------------------------------> P8
..     # \-------------------------> B4 <-------------------------/
..     #  \----------> B2 <---------/ \----------> B6 <----------/
..     #   \--> B1 <--/ \--> B3 <--/   \--> B5 <--/  \--> B7 <--/
..     python3 src/encode.py --intra_period=8 --p_period=8 --input=xxx.yuv

.. Note the introduction of the ``--p_period`` argument in the command line. This
.. sets the distance between the ``IO`` frame and the successive ``P`` frames.
.. Reducing the temporal distance alleviates the motion compensation process, which
.. can be useful. It can also enables more complex GOP structure such as:

.. .. code:: python

..     # There is no more prediction from I0 to P8. Instead the GOP in split in
..     # half so that there is no inter frame with reference further than --p_period

..     # I0 -----------------------> P4 ------------------------> P8
..     #  \----------> B2 <---------/ \----------> B6 <----------/
..     #   \--> B1 <--/ \--> B3 <--/   \--> B5 <--/  \--> B7 <--/
..     python3 src/encode.py --intra_period=8 --p_period=4 --input=xxx.yuv


Image coding
""""""""""""

To achieve plain image coding *i.e.* compression a single standalone intra
frame simply set the intra period and P period to zero.

.. code:: python

    # Image coding, input could also be a .yuv file in such case, it
    # would only encode the first video frame.
    python3 src/encode.py --intra_period=0 --p_period=0 --input=xxx.png