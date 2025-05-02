# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:42:00 2024

@author: renxi

Segmentation network for multiple spectral RS images

multiple-modal fusion segmentation network

"""

import torch
from torch import nn
from torch.nn import ReLU
from torch.nn import functional as F

from . import resnet_enc 
#import resnet_enc


# Normal convolution block
class conv_block(nn.Module):
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
        
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        return out
    
    
# Residual convolution block (adapted from Resnet)
class res_conv_block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(res_conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
    
    
# depthwise separable convolution
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



# residual depthwise separable module decoder
class decoder_block(nn.Module):
    def __init__(self,  input_channels, output_channels):
        super(decoder_block, self).__init__()

        self.identity = torch.nn.Sequential(
            #torch.nn.Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        )
        
        
        self.decode = torch.nn.Sequential(            
            nn.BatchNorm2d(input_channels),
            depthwise_separable_conv(input_channels, input_channels),
            nn.BatchNorm2d(input_channels),
            ReLU(inplace=True),
            depthwise_separable_conv(input_channels, output_channels),
            
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        residual = self.identity(x)        
        out = self.decode(x)
        out += residual
        
        out = F.interpolate(out, scale_factor=2, 
                            mode='bilinear', align_corners=True)
                       
        return out
    


# gated fusion module
class gated_fusion_block(nn.Module):
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
class simple_fusion_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fuseConv = nn.Conv2d(2 * in_channels, in_channels, 
                                  kernel_size=1, padding=0)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        return self.fuseConv(out)

    
# ------------------------------------------------------------------------    
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

        
        self.avgConv = nn.Conv2d(in_channels=64, out_channels=1, 
                                 kernel_size=3, padding=1)
        
        # decoder blocks for each level (inchannels, outchannels)
        self.decoder5 = decoder_block(512, 256)
        self.decoder4 = decoder_block(256, 128)
        self.decoder3 = decoder_block(128, 64)
        self.decoder2 = decoder_block(64, 64)
        self.decoder1 = decoder_block(64, 32)
        
        # fusion blocks
        self.fuse5 = simple_fusion_block(256)
        self.fuse4 = simple_fusion_block(128)
        self.fuse3 = simple_fusion_block(64)
        self.fuse2 = simple_fusion_block(64)
        self.fuse1 = simple_fusion_block(32)
        
        
    def forward(self, x, aux):
        nb, _, h, w = x.shape
        
        # encoding -----------------------------------------------
        # output feats is a list of size of 6, and the first 
        # feature feats[0] is the  original input data           
        img_feats = self.img_encoder(x)
        
        aux_feats = self.aux_encoder(aux)
        
        
        # ---------------------------------------------------------
        # decoding image features and upsampling level by level
        dimg_feats5 = self.decoder5(img_feats[5])                
        daux_feats5 = self.decoder5(aux_feats[5])                
        fused_ft5 = self.fuse5(dimg_feats5, daux_feats5)
        
        dimg_feats4 = self.decoder4(dimg_feats5)
        daux_feats4 = self.decoder4(daux_feats5)        
        fused_ft4 = self.fuse4(dimg_feats4, daux_feats4)        
        
        dimg_feats3 = self.decoder3(dimg_feats4)
        daux_feats3 = self.decoder3(daux_feats4)       
        fused_ft3 = self.fuse3(dimg_feats3, daux_feats3)         
        
        dimg_feats2 = self.decoder2(dimg_feats3)
        daux_feats2 = self.decoder2(daux_feats3)       
        fused_ft2 = self.fuse2(dimg_feats2, daux_feats2)         
        
        dimg_feats1 = self.decoder1(dimg_feats2)
        daux_feats1 = self.decoder1(daux_feats2)       
        fused_ft1 = self.fuse1(dimg_feats1, daux_feats1)         
        
        # fuse the img and aux features
        fused_ft = torch.cat((dimg_feats1, daux_feats1), dim=1)
        
        seg = self.avgConv(fused_ft)
        
        #seg = F.interpolate(seg, scale_factor=2)
        
        return seg
    
    

if __name__ == '__main__':
    nchns = 3
    mfseg = MFSegNet(img_channels=nchns, aux_channels=1)

    img = torch.rand((4, nchns, 256,256))
    aux = torch.rand((4, 1, 256,256))
    o = mfseg(img, aux)
    print(o.shape)    
        
        