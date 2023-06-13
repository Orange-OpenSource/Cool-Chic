#!/bin/bash
python3 src/decode.py                               \
    --device=cpu                                    \
    --input=samples/bitstream.bin                   \
    --output=samples/biville_decoded.png