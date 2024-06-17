Training
=========

The ``training`` submodule gather the different function required to encode
(*i.e.* train a model on) an image. The warm-up is about comparing a list of
different random initialization and selecting the best one after a few hundred
training iterations. At the end of the training, the model is quantized so that
it can be efficiently written into the bitstream.

.. toctree::

   Loss function <loss>
   Train <train>
   Test <test>
   Warm-up <warmup>
   Quantize model <quantizemodel>




