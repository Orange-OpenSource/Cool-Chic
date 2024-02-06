#!/bin/bash
python3 src/encode.py                               \
    --input=samples/video/fiveframes-D-BasketballPass_416x240_50p_yuv420_8b.yuv \
    --output=samples/video/fiveframes-D-BasketballPass_416x240_50p_yuv420_8b.bin \
    --workdir=./my_video_workdir/               \
    --intra_period=4                                \
    --p_period=4                                    \
    --lmbda=0.0002                                  \
    --start_lr=1e-2                                 \
    --layers_synthesis=9-1-linear-relu,X-1-linear-none \
    --upsampling_kernel_size=8                      \
    --arm=8,2                                       \
    --n_ft_per_res=1,1,1,1,1,1,1                    \
    --n_itr=1000                                    \
    --n_train_loops=1