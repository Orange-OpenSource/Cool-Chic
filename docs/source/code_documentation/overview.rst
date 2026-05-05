Cool-chic code at a glance
==========================

Coolchic source code is organized as follows.

.. code:: none

    coolchic/
        |___ component/                     The different modules composing the Encoder
        |          |___ core/               The different NNs, quantizer and Cool-chic module
        |          |___ intercoding/        RAFT, warping function, global translation
        |          |___ frame.py            Encode a frame with a cool-chic
        |          |___ video.py            Encode one frame function, reference management
        |___ io/                            Read & write PNG, PPM and YUV
        |___ training/                      Train, test, quantize the models, loss, presets, warm-up
        |___ nnquant/                       Quantize neural networks at the end of the training
        |___ utils/                         Coding structure and various stuff.
        |___ bitstream/                     Write the bitstream from a trained model.

The component insides ``coolchic/component/core/`` are made to be instantiated
independently allowing for easy experimentation.