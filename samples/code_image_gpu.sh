#!/bin/bash
python3 src/encode.py                               \
    --device=cuda:0                                 \
    -i samples/biville.png                          \
    -o samples/bitstream.bin                        \
    --decoded_img_path=samples/decoded.png          \
    --model_save_path=samples/model.pt              \
    --enc_results_path=samples/encoder_results.txt  \
    --lmbda=0.001                                   \
    --start_lr=1e-2                                 \
    --n_itr=1000                                    \
    --layers_synthesis=12,12                        \
    --layers_arm=12,12                              \
    --n_ctx_rowcol=2                                \
    --latent_n_grids=7
