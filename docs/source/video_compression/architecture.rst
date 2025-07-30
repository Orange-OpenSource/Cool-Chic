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

Motion compensation options
"""""""""""""""""""""""""""

Motion compensation requires to interpolate in between pixels e.g., to achieve a
shift of 7.25 pixels. To do so, an interpolation filter is used. The size of the
interpolation filter is parameterized by ``--warp_filter_size``. See paper `Efficient Sub-pixel Motion
Compensation in Learned Video Codecs, Ladune et al
<https://arxiv.org/pdf/2507.21926>`_ for more details.

The type of interpolation performed is derived from the filter size. Note that
``--warp_filter_size`` must be even

.. list-table:: Motion compensation interpolation.
   :widths: 30 30
   :header-rows: 1

   * - ``warp_filter_size``
     - Type
   * - 2
     - Bilinear
   * - 4
     - Bicubic
   * - 6 or above
     - Sinc-based

Motion fields downsampling
""""""""""""""""""""""""""

To obtain uniform motion on a block of :math:`B \times B` pixels, we rely on
``--n_ft_per_res_motion``. As explained :ref:`here <no_high_res_latent>`, having
the first :math:`N` latents of the motion Cool-chic with no feature leads to a
decoding of a motion fields with motion values common to blocks of :math:`2^N
\times 2^N` pixels.

.. code-block::

    # Each motion vector is common for a block of 8x8 pixels.
    --n_ft_per_res_motion=0,0,0,1,1

Frame-wise decoder configuration
""""""""""""""""""""""""""""""""

In a hierarchical coding structure, not all frames are equally important. As
such, it is often interesting to use lighter architectures for not-so-important
frames. The :ref:`video coding example script <video_coding_example>`
provides an example of frame-wise decoder configuration automatically generated.
