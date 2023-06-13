# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List

from models.quantizable_module import QuantizableModule
from utils.constants import POSSIBLE_Q_STEP_UPS_NN

kernel_bicubic_alignfalse=[
    [0.0012359619 ,0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 ,0.0037078857 ,0.0012359619],
    [0.0037078857 ,0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 ,0.0111236572 ,0.0037078857],
    [-0.0092010498 ,-0.0276031494 ,0.0684967041 ,0.2300262451 ,0.2300262451 ,0.0684967041 ,-0.0276031494 ,-0.0092010498],
    [-0.0308990479 ,-0.0926971436 ,0.2300262451 ,0.7724761963 ,0.7724761963 ,0.2300262451 ,-0.0926971436 ,-0.0308990479],
    [-0.0308990479 ,-0.0926971436 ,0.2300262451 ,0.7724761963 ,0.7724761963 ,0.2300262451 ,-0.0926971436 ,-0.0308990479],
    [-0.0092010498 ,-0.0276031494 ,0.0684967041 ,0.2300262451 ,0.2300262451 ,0.0684967041 ,-0.0276031494 ,-0.0092010498],
    [0.0037078857 ,0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 ,0.0111236572 ,0.0037078857],
    [0.0012359619 ,0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 ,0.0037078857 ,0.0012359619],
]

class Upsampling(QuantizableModule):
    def __init__(self, upsampling_kernel_size: int):
        """Instantiate an upsampling layer.

        Args:
        """
        super().__init__(possible_q_steps=POSSIBLE_Q_STEP_UPS_NN)

        # adapted filter size
        K=upsampling_kernel_size
        self.upsampling_padding = (K//2, K//2, K//2, K//2)
        self.upsampling_crop=(3*K-2)//2

        # pad bicubic filter according to desired optimised kernel size
        K_bicubic=8 # default bicubic upsampling filter
        tmpad=(K-K_bicubic)//2
        kernel_pad=(tmpad, tmpad, tmpad, tmpad)
        upsampling_kernel=F.pad(torch.tensor(kernel_bicubic_alignfalse), kernel_pad) # padded with zeros by default

        # compatible with transpose conv
        upsampling_kernel=torch.unsqueeze(upsampling_kernel,0)
        upsampling_kernel=torch.unsqueeze(upsampling_kernel,0)

        # initialise layer and weights
        self.upsampling_layer = nn.ConvTranspose2d(1,1,upsampling_kernel.shape ,bias=False,stride=2)
        self.upsampling_layer.weight.data = nn.Parameter(upsampling_kernel,requires_grad=True)


    def forward(self, decoder_side_latent: List[Tensor]) -> Tensor:

        """From a list of C [1, 1, H_i, W_i] tensors, where H_i = H / 2 ** i abd
            W_i = W / 2 ** i, upsample each tensor to H * W. Then return the values
            as a 2d tensor [H * W, C]. This is the synthesis input

        Args:
            decoder_side_latent (List[Tensor]): a list of C latent variables
                with resolution [1, 1, H_i, W_i].

        Returns:
            Tensor: The [H * W, C] synthesis input.
        """

        # upsampling kernel size, convolution padding and cropping preparation

        upsampled_latent: Tensor = decoder_side_latent[-1]
        #print(f"{upsampled_latent.shape}")
        for i in range(len(decoder_side_latent) - 1, 0, -1):

            x_pad = F.pad(upsampled_latent, self.upsampling_padding,mode='replicate')
            y_conv= self.upsampling_layer(x_pad)

            # crop to remove padding in convolution
            H=y_conv.shape[-2]
            W=y_conv.shape[-1]
            upsampled_latent = y_conv[:, :, self.upsampling_crop:H-self.upsampling_crop, self.upsampling_crop:W-self.upsampling_crop]

            # crop to comply to higher resolution fm size before concatenation
            target_tensor = decoder_side_latent[i - 1]
            upsampled_latent = upsampled_latent[:, :, 0:target_tensor.shape[-2], 0:target_tensor.shape[-1]]
            upsampled_latent = torch.cat((target_tensor, upsampled_latent), dim=0)

        # Here we return all the latent variables
        return upsampled_latent.permute((1,0,2,3))


    def display_filters(self):
            # save the 2D basis for visualization purpose

            filter=self.latent_filter_0.clone()
            weight=filter.to('cpu')
            weight=torch.squeeze(weight,dim=0)
            weight=torch.squeeze(weight,dim=0)
            weight=weight.detach().numpy()
            np.savetxt("filter0.tsv",weight,delimiter='\t',fmt='%.10f')

            filter=self.latent_filter_1.clone()
            weight=filter.to('cpu')
            weight=torch.squeeze(weight,dim=0)
            weight=torch.squeeze(weight,dim=0)
            weight=weight.detach().numpy()
            np.savetxt("filter1.tsv",weight,delimiter='\t',fmt='%.10f')

    def display_upsampler(self):

            filter=self.upsampling_kernel.clone()
            weight=filter.to('cpu')
            weight=torch.squeeze(weight,dim=0)
            weight=torch.squeeze(weight,dim=0)
            weight=weight.detach().numpy()
            np.savetxt("upsampler.tsv",weight,delimiter='\t',fmt='%.10f')



