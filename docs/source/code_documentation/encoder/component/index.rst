Component
=========

Components gather all the modules of the enc.

   * ``video.py`` implements the class ``VideoEncoder`` which is used to compress a
     video *i.e.* one intra frame followed by zero or more inter frames. It
     contains one or more instances of ``FrameEncoder``.

   * ``frame.py`` implements the class ``FrameEncoder`` is used to compress a
     frame. To do so, it relies on one ``CoolChicEncoder``.

   * ``coolchic.py`` implements the class ``CoolChicEncoder``, which is the main
     coding engine of the codec. It is composed of latent grids, an
     auto-regressive module, an upsampling and a synthesis.

   * ``intercoding/`` contains functions required to encode an intra frame such
     RAFT-based motion estimation, warping an image with an optical flow or
     finding a global translation between two frames.


.. toctree::

   frame <frame>
   coolchic <coolchic>
   core <core/index>
   intercoding <intercoding/index>

