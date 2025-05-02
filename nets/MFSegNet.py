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
# input:  (n, in_channels, H, W)
# output: (n, out_channels, H, W)
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
class res_conv_block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels):
        super(res_conv_block, self).__init__()
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
# input:  (n, in_channels, H, W)
# output: (n, out_channels, 2*H, 2*W)
# skip_channels is often the half size of in_channels
class decoder_block(nn.Module):
    def __init__(self,  in_channels, out_channels, skip_channels=0):
        super(decoder_block, self).__init__()

        self.identity = torch.nn.Sequential(
            #torch.nn.Upsample(2, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )
        
        self.conv1 = depthwise_separable_conv(in_channels+skip_channels, out_channels)        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Do not use relu here. It causes errors.
        #self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.ReLU()
        
        self.conv2 = depthwise_separable_conv(out_channels, out_channels)        
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
        self.decoder5 = decoder_block(512, 256, skip_channels=256)
        self.decoder4 = decoder_block(256, 128, skip_channels=128)
        self.decoder3 = decoder_block(128, 64, skip_channels=64)
        self.decoder2 = decoder_block(64, 64, skip_channels=64)
        self.decoder1 = decoder_block(64, 32)
        
        # fusion blocks
        self.fuse5 = simple_fusion_block(256)
        self.fuse4 = simple_fusion_block(128)
        self.fuse3 = simple_fusion_block(64)
        self.fuse2 = simple_fusion_block(64)
        self.fuse1 = simple_fusion_block(32)
        
        # conv
        #self.conv4 = nn.Conv2d(256, 32, kernel_size=1)
        #self.conv3 = nn.Conv2d(128, 32, kernel_size=1)
        #self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        #self.conv1 = nn.Conv2d(64, 32, kernel_size=1)
        #self.conv0 = nn.Conv2d(32, 32, kernel_size=1)
        
    def forward(self, x, aux):
        nb, _, h, w = x.shape
        
        # encoding -----------------------------------------------
        # output feats is a list of size of 6, and the first 
        # feature feats[0] is the  original input data           
        img_feats = self.img_encoder(x)
        
        aux_feats = self.aux_encoder(aux)
        
        
        # ---------------------------------------------------------
        # decoding image features and upsampling level by level
        # (H, W)/32 -> (H, W)/16 
        xx = self.decoder5(img_feats[5], skip=img_feats[4])                
        yy = self.decoder5(aux_feats[5], skip=aux_feats[4])                
        #fused_ft4 = self.fuse5(xx, yy)
        
        # (H, W)/16 -> (H, W)/8
        xx = self.decoder4(xx, skip=img_feats[3])
        yy = self.decoder4(yy, skip=aux_feats[3])        
        #fused_ft3 = self.fuse4(xx, yy)        
        
        # (H, W)/8 -> (H, W)/4
        xx = self.decoder3(xx, skip=img_feats[2])
        yy = self.decoder3(yy, skip=aux_feats[2])       
        #fused_ft2 = self.fuse3(xx, yy)         
        
        # (H, W)/4 -> (H, W)/2
        xx = self.decoder2(xx, skip=img_feats[1])
        yy = self.decoder2(yy, skip=aux_feats[1])       
        #fused_ft1 = self.fuse2(xx, yy)         
        
        # (H, W)/2 -> (H, W)
        xx = self.decoder1(xx)
        yy = self.decoder1(yy)       
        #fused_ft0 = self.fuse1(xx, yy)         
        
                
        # fuse the img and aux features
        fused_ft = torch.cat((xx, yy), dim=1)
        #fused_ft = self.conv4(fused_ft4)
        
        seg = self.avgConv(fused_ft)
        
        #seg = F.interpolate(seg, scale_factor=2)
        
        return seg
    
    


        
        