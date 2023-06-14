#!/bin/bash
python3 src/encode.py                               \
    --input=samples/biville.png                     \
    --output=samples/bitstream.bin                  \
    --model_save_path=samples/model.pt              \
    --enc_results_path=samples/encoder_results.txt  \
    --lmbda=0.0002                                  \
    --start_lr=1e-2                                 \
    --recipe=instant                                \
    --layers_synthesis=40-1-linear-relu,3-1-linear-relu,3-3-residual-relu,3-3-residual-none \
    --layers_arm=24,24                              \
    --n_ctx_rowcol=3                                \
    --latent_n_grids=7