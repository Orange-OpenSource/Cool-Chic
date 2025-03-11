Cool-chic code at a glance
==========================

Coolchic source code is organized as follows

.. code:: none

    coolchic/
        |___ enc/
        |     |___ component/                     The different modules composing the Encoder
        |     |          |___ core/               The different NN (ARM, upsampling and synthesis) + quantizer
        |     |          |___ intercoding/        RAFT, warping function, global translation
        |     |          |___ coolchic.py         Entire Cool-chic: Latent + ARM + upsampling + synthesis
        |     |          |___ frame.py            Encode a frame with a cool-chic
        |     |          |___ video.py            Encode one frame function, reference management
        |     |___ io/                            Read & write PNG, PPM and YUV
        |     |___ training/                      Train, test, quantize the models, loss, presets, warm-up
        |     |___ nnquant/                       Quantize neural networks at the end of the training
        |     |___ utils/                         Coding structure and various stuff.
        |     |___ bitstream/                     Write the bitstream from a trained model.
        |
        |___ dec/                                 Python interface with the C API for the decoder
        |
        |___ cpp/                                 Cool-chic C API for fast decoding & entropy coding
