# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:42:26 2025

@author: renxi
"""
# %% 
import torch
from nets.MFSegNet import (conv_block, res_conv_block, decoder_block,
                           MFSegNet)

# %%
nchns = 3
img = torch.rand((4, nchns, 256,256))
aux = torch.rand((4, 1, 256,256))

# %%
m = conv_block(nchns, 16)
print(m(img).shape)

# %%
ft = torch.rand((4, 16, 256,256))
m = res_conv_block(16, 8)
print(m(ft).shape)

# %%
ft = torch.rand((4, 16, 256,256))
skip = torch.rand((4, 8, 512,512))
# with skip connection
m1 = decoder_block(16, 32, skip_channels=8)
print(m1(ft, skip).shape)
# without skip connection
m2 = decoder_block(16, 32)
print(m2(ft).shape)


# %%
mfseg = MFSegNet(img_channels=nchns, aux_channels=1)
o = mfseg(img, aux)
print(o.shape)    
