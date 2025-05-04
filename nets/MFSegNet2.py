# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:42:00 2024

@author: renxi

Segmentation network for multiple spectral RS images

multiple-modal fusion segmentation network

"""

import numpy as np
import torch
from torch import nn
from torch.nn import ReLU
from torch.nn import functional as F
from torch.nn import ModuleList


# Normal convolution block
# input:  (n, in_channels, H, W)
# output: (n, out_channels, H, W)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        #self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        #out = self.pool(out)
        return out
    
    
# Residual convolution block (adapted from Resnet)
# input:  (n, in_channels, H, W)
# output: (n, out_channels, H, W)
class ResConvBock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels):
        super(ResConvBock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.identity = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=1, padding=0)
        
        
    def forward(self, x):
        identity = self.identity(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #if self.downsample is not None:
        #    identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
    
    
    
# depthwise separable convolution
# input:  (n, in_channels, H, W)
# output: (n, out_channels, H, W)
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



# residual depthwise separable module decoder
# input:  (n, in_channels, H, W)
# output: (n, out_channels, 2*H, 2*W)
# skip_channels is often the half size of in_channels
class DecoderBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()

        self.identity = torch.nn.Sequential(
            #torch.nn.Upsample(2, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )
        
        self.conv1 = DSConv(in_channels+skip_channels, out_channels)        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Do not use relu here. It causes errors.
        #self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.ReLU()
        
        self.conv2 = DSConv(out_channels, out_channels)        
        self.bn2 = nn.BatchNorm2d(out_channels)

    # 
    def forward(self, x, skip=None):
        # upsampling         
        x = F.interpolate(x, scale_factor=2, 
                          mode='bilinear', align_corners=True)
        
        residual = self.identity(x)        
        
        if skip is not None:                
            ft = torch.cat([x, skip], dim=1)
        else:
            ft = x
            
        ft = self.conv1(ft)
        ft = self.bn1(ft)           
        #ft = self.relu(ft)           
        
        ft = self.conv2(ft)
        ft = self.bn2(ft)
        #ft = self.relu(ft)           
        
        ft += residual
                       
        return ft
    


# gated fusion module
class GatedFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = torch.nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)
        PG = x * G
        FG = y * (1 - G)
        return self.conv(torch.cat([FG, PG], dim=1))


# simple fusion module
class SimpleFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fuseConv = nn.Conv2d(2 * in_channels, in_channels, 
                                  kernel_size=1, padding=0)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        return self.fuseConv(out)

    
# ------------------------------------------------------------------------   
# multiple-modal data fusion network for segmentation 
class MFSegNet2(nn.Module):
    # img_channels: channels of the input image 
    # aux_channnels: channels of auxillary data 
    def __init__(self, img_channels=3, aux_channels=1, 
                 n_classes=1, pretrained=True):
        super().__init__()
        
        # class number of segmentation
        self.n_classes = n_classes
        
        # channels of features in the encoders
        self.feature_channels =  [16, 32, 64, 128, 256, 512]
        
        # image encoders
        channels = [img_channels] + self.feature_channels
        self.img_enc_blocks = ModuleList([ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
                        
        # auxiliary data encoders
        channels = [aux_channels] + self.feature_channels
        self.aux_enc_blocks = ModuleList([ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

        self.pool = nn.MaxPool2d(2)
        
        
        # decoder blocks for image features (inchannels, outchannels)
        blocks = []
        for i in range(len(self.feature_channels)-1):
            in_chs = self.feature_channels[i+1] 
            out_chs = self.feature_channels[i] 
            skip_chs = self.feature_channels[i] 
            dec = DecoderBlock(in_chs, out_chs, skip_channels=skip_chs)
            blocks.append(dec)
        self.img_dec_blocks = ModuleList(blocks)
        
        # decoder blocks for aux features (inchannels, outchannels)
        blocks = []
        for i in range(len(self.feature_channels)-1):
            in_chs = self.feature_channels[i+1] 
            out_chs = self.feature_channels[i] 
            skip_chs = self.feature_channels[i] 
            dec = DecoderBlock(in_chs, out_chs, skip_channels=skip_chs)
            blocks.append(dec)
        self.aux_dec_blocks = ModuleList(blocks)
        
        # fusion blocks for each level of featrues
        blocks = []
        for i in range(len(self.feature_channels)-1):
            in_chs = self.feature_channels[i+1] 
            dec =  SimpleFusionBlock(in_chs)
            blocks.append(dec)
        self.fusion_blocks = ModuleList(blocks)
        
        
        # the final fusion block
        inchns = self.feature_channels[0]
        self.final_fuse = SimpleFusionBlock(inchns)
        
        
        # segmentation head
        inchns = self.feature_channels[0]
        self.avgConv = nn.Conv2d(in_channels=inchns, out_channels=n_classes, 
                                 kernel_size=3, padding=1)


        #self.initialize_weights()
        print('[INFO] MFSegNet2 model has been created')
        

                        
    def forward(self, x, aux):
        nb, _, h, w = x.shape
        
        # encoding -----------------------------------------------
        # image features
        img_feats = []
        feat = x
        for enc in self.img_enc_blocks:
            feat = enc(feat)
            img_feats.append(feat)
            #downsampling
            feat = self.pool(feat)
            
        # aux data features    
        aux_feats = []
        feat = aux
        for enc in self.aux_enc_blocks:
            feat = enc(feat)
            aux_feats.append(feat)
            #downsampling
            feat = self.pool(feat)
                
        # ---------------------------------------------------------
        # decoding image features and upsampling level by level
        xx = img_feats[-1]
        yy = aux_feats[-1]
        for i in range(1, len(img_feats)):
            # (H, W)/32 -> (H, W)/16 
            # ...
            # (H, W)/2 -> (H, W)
            xx = self.img_dec_blocks[-i](xx, skip=img_feats[-i-1])                
            yy = self.aux_dec_blocks[-i](yy, skip=aux_feats[-i-1])                
            #fused_ft4 = self.fuse5(xx, yy)
              
        # here, xx, yy have the same width and height as the input.
        # now, fuse the img(xx) and aux(yy) features
        fused_ft = self.final_fuse(xx, yy)
        
        # apply segmentation module        
        seg = self.avgConv(fused_ft)
        
        
        return seg
    
    

if __name__ =='__main__':
    nchns = 3
    img = torch.rand((4, nchns, 256,256))
    aux = torch.rand((4, 1, 256,256)) 
  
    mfseg = MFSegNet2(img_channels=nchns, aux_channels=1)
    o = mfseg(img, aux)
    print(o.shape)    


        
        