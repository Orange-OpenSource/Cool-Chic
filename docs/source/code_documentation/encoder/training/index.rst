Training
=========

The ``training`` submodule gather the different functions required to encode
(*i.e.* train a model on) an image.

   * ``warmup.py`` is about comparing a list of different random initialization and
     selecting the best one after a few hundred training iterations

   * ``train.py`` is the actual training loop used to perform a number of
     optimization iterations.

   * ``loss.py`` measures the performance of the model to enable learning.

   * ``test.py`` measures more quantities than loss does, e.g. the neural networks
     rate

   * ``quantizemodel.py`` is called at the end of the training to quantize the
     neural networks rate

   * ``preset.py`` contains training recipe with hyper-parameters controlling the
     training process.


.. toctree::

   loss <loss>
   train <train>
   test <test>
   warmup <warmup>
   quantizemodel <quantizemodel>
   preset <preset>




