Decoder configuration
=====================

.. image:: ../assets/frame-decoding-2.png
  :alt: Cool-chic inter-frame decoder overview

Inter frames (P or B) decoders involves two Cool-chic decoders, one for the
residue and one for the motion while intra frames only requires a residue
decoder. Each Cool-chic decoder has the same architecture as the one described
in the  :ref:`image coding section. <image_compression_overview>`

Since the residue and motion Cool-chic decoders have the same design, they are
parameters through similar arguments. For instance ``--arm_residue`` sets up the
the ARM architecture for the residue decoder, while ``--arm_motion`` achieves
the same thing for the motion decoder.

Similarly, it is possible to provide a decoder configuration file for the motion
Cool-chic decoder through the ``--dec_cfg_motion`` arguments. Example
configuration files are listed in ``cfg/dec/motion/``.

Frame-wise decoder configuration
""""""""""""""""""""""""""""""""

In a hierarchical coding structure, not all frames are equally important. As
such, it is often interesting to use lighter architectures for not-so-important
frames. The :ref:`video coding example script <video_coding_example>`
provides an example of frame-wise decoder configuration automatically generated.
