#!/bin/bash
python3 src/encode.py                               \
    --input=samples/biville.png                     \
    --output=samples/bitstream.bin                  \
    --workdir=./my_temporary_workdir/               \
    --lmbda=0.0002                                  \
    --start_lr=1e-2                                 \
    --layers_synthesis=40-1-linear-relu,3-1-linear-relu,X-3-residual-relu,X-3-residual-none \
    --upsampling_kernel_size=8                      \
    --layers_arm=24,24                              \
    --n_ctx_rowcol=3                                \
    --n_ft_per_res=1,1,1,1,1,1,1                    \
    --n_itr=1000                                    \
    --n_train_loops=1