# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:42:00 2024

@author: renxi

Segmentation network for multiple spectral RS images

multiple-modal fusion segmentation network

"""

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from . import resnet_enc 
#import resnet_enc


class MFSegNet(nn.Module):
    def __init__(self, img_channels=3, aux_channels=1, 
                 n_classes=1, pretrained=True):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        
        # image encoder part        
        self.img_encoder = resnet_enc.resnet34_enc(in_channels=img_channels)

        # auxiliary data encoder part
        self.aux_encoder = resnet_enc.resnet34_enc(in_channels=aux_channels)

        
        self.avgConv = nn.Conv2d(in_channels=128, out_channels=1, 
                                 kernel_size=3, padding=1)
                
        
    def forward(self, x, aux):
        nb, _, h, w = x.shape
        # output feats is a list of size of 6, and the first 
        # feature feats[0] is the  original input data           
        img_feats = self.img_encoder(x)
        
        aux_feats = self.aux_encoder(aux)
        #ndwi = (x[:,2]-x[:,7])/(x[:,2]+x[:,7]+0.00001)
        #ndwi = torch.unsqueeze(ndwi, dim=1)
        #ndwi = aux
        
        fused_ft = torch.cat((img_feats[1], aux_feats[1]), dim=1)
        
        seg = self.avgConv(fused_ft)
        
        seg = F.interpolate(seg, scale_factor=2)
        
        return seg
    
    

if __name__ == '__main__':
    nchns = 12
    mfseg = MFSegNet(img_channels=nchns, aux_channels=1)

    img = torch.rand((4, nchns, 256,256))
    aux = torch.rand((4, 1, 256,256))
    o = mfseg(img, aux)
    print(o.shape)    
        
        