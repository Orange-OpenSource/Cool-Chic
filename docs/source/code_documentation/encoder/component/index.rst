Component
=========

Components gather all the modules of the enc.

   * ``VideoEncoder`` is used to compress a video *i.e.* one intra frame
     followed by zero or more inter frames. It contains one or more
     ``FrameEncoder``.

   * ``FrameEncoder`` is used to compress a frame. It contains one
     ``CoolChicEncoder``.

   * ``CoolChicEncoder`` is the main coding engine of the codec. It is composed
     of latent grids, an auto-regressive module, an upsampling and a synthesis.

   * ``Core`` contains all the modules stated above and required by a
     ``CoolChicEncoder``


.. toctree::

   VideoEncoder <video>
   FrameEncoder <frame>
   CoolChicEncoder <coolchic>
   Core <core/index>


