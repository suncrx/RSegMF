#!/usr/bin/env python3
"""
-*- coding:utf-8 -*-
demo : GFBUNet.py
@author : Shuoyi Wang
Date : 2024/6/16 18:10
        The script contains the definition of our GFBUNet model.
"""
# import the necessary packages
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
import torch as nn


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, kernel_size=3, 
                            padding="same", bias=False)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, kernel_size=3, 
                            padding="same", bias=False)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        #return self.conv2(self.relu(self.conv1(x)))
        return self.conv2(self.relu(self.conv1(x)))



class Gated_Fusion(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = torch.nn.Sequential(
            Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        )
        self.conv = Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return self.conv(torch.cat([FG, PG], dim=1))


class depthwise_separable_conv(Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = Conv2d(nin, nin, kernel_size=kernel_size, 
                                padding=padding, groups=nin)
        self.pointwise = Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class decoder_block(Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(decoder_block, self).__init__()

        self.identity = torch.nn.Sequential(
            #torch.nn.Upsample(2, mode="bilinear"),
            Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        )

        self.decode = torch.nn.Sequential(
            #torch.nn.Upsample(2, mode="bilinear"),
            torch.nn.BatchNorm2d(input_channels),
            depthwise_separable_conv(input_channels, input_channels),
            torch.nn.BatchNorm2d(input_channels),
            ReLU(inplace=True),
            depthwise_separable_conv(input_channels, output_channels),
            torch.nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        residual = self.identity(x)

        out = self.decode(x)

        out += residual

        return out


class Encoder(Module):
    def __init__(self, channels=(1, 16, 32, 64, 128, 256)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)


    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []

        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store the outputs,
            # and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)

        # return the list containing the intermediate outputs
        return blockOutputs


class Img_Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64, 128, 256)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):

        # initialize an empty list to store the intermediate outputs
        blockOutputs = []

        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store the outputs,
            # and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)

        # return the list containing the intermediate outputs
        return blockOutputs


class HED(Module):
    def __init__(self, channels=(3, 16, 32, 64, 128, 256)):
        super().__init__()
        #self.dconv_down = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

        self.dconv_down1 = Block(3, 16)
        self.dconv_down2 = Block(16, 32)
        self.dconv_down3 = Block(32, 64)
        self.dconv_down4 = Block(64, 128)
        self.dconv_down5 = Block(128, 256)
        self.maxpool = MaxPool2d(2)

        # HED Block
        #self.dsn = ModuleList([Conv2d(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.dsn1 = Conv2d(16, 1, 1)
        self.dsn2 = Conv2d(32, 1, 1)
        self.dsn3 = Conv2d(64, 1, 1)
        self.dsn4 = Conv2d(128, 1, 1)
        self.dsn5 = Conv2d(256, 1, 1)
        
    def forward(self, x):

        h = x.size(2)
        w = x.size(3)

        conv6 = self.dconv_down1(x)
        x = self.maxpool(conv6)
        conv7 = self.dconv_down2(x)
        x = self.maxpool(conv7)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        conv5 = self.dconv_down5(x)

        # out = F.sigmoid(self.out(x))
        ## side output
        d1 = self.dsn1(conv6)
        d2 = F.upsample_bilinear(self.dsn2(conv7), size=(h, w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h, w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h, w))

        d1_out = F.sigmoid(d1)
        d2_out = F.sigmoid(d2)
        d3_out = F.sigmoid(d3)
        d4_out = F.sigmoid(d4)
        d5_out = F.sigmoid(d5)
        concat = torch.cat((d1_out, d2_out, d3_out, d4_out, d5_out), 1)

        return concat


class Boundary_Enchance(Module):
    def __init__(self, mode='Train'):
        super().__init__()
        self.mode = mode

        # boundary enhancement part
        self.fuse = torch.nn.Sequential(Conv2d(5, 16, 1), ReLU(inplace=True))
        self.SE_mimic = torch.nn.Sequential(
            torch.nn.Linear(16, 16, bias=False),
            ReLU(inplace=True),
            torch.nn.Linear(16, 5, bias=False),
            torch.nn.Sigmoid()
        )
        self.final_boundary = Conv2d(5, 2, 1)
        self.final_conv = torch.nn.Sequential(
            Conv2d(32, 16, 3, padding=1),
            ReLU(inplace=True)
        )
        self.final_mask = Conv2d(16, 2, 1)
        self.relu = ReLU()
        #self.out = Conv2d(16, 1, 1)
        self.conv = Conv2d(1, 16, 1)



    def forward(self, x, y):
        fuse_box = self.fuse(y)
        GAP = F.adaptive_avg_pool2d(fuse_box, (1, 1))
        GAP = GAP.view(-1, 16)
        se_like = self.SE_mimic(GAP)
        se_like = torch.unsqueeze(se_like, 2)
        se_like = torch.unsqueeze(se_like, 3)

        feat_se = y * se_like.expand_as(y)
        boundary = self.final_boundary(feat_se)
        boundary_out = torch.unsqueeze(boundary[:, 1, :, :], 1)
        bd_sftmax = F.softmax(boundary, dim=1)
        boundary_scale = torch.unsqueeze(bd_sftmax[:, 1, :, :], 1)

        feat_concat = torch.cat([x, fuse_box], 1)
        feat_concat_conv = self.final_conv(feat_concat)
        mask = self.final_mask(feat_concat_conv)
        mask_sftmax = F.softmax(mask, dim=1)
        mask_scale = torch.unsqueeze(mask_sftmax[:, 1, :, :], 1)

        #if self.mode == 'Train':
        scalefactor = torch.clamp(mask_scale + boundary_scale, 0, 1)
        scalefactor = self.conv(scalefactor)
        #elif self.mode == 'Infer':
        #scalefactor = torch.clamp(mask_scale + 5 * boundary_scale, 0, 1)

        return scalefactor

        #z = scalefactor





class GDecoder(Module):
    def __init__(self, channels=(256, 128, 64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([decoder_block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.gated = ModuleList([Gated_Fusion(channels[i + 1]) for i in range(len(channels) - 1)])

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)

        # return the cropped features
        return encFeatures

    def forward(self, x, encFeatures, fusionFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)

            # crop the current features from the encoder blocks, concatenate them with the current upsampled features,
            # and pass the concatenated output through the current decoder block
            encFeat = self.crop(encFeatures[i], x)
            fusionFeat = self.crop(fusionFeatures[i], x)

            encFeatGated = self.gated[i](encFeat, fusionFeat)
            x = torch.cat([x, encFeatGated], dim=1)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x


class Decoder(Module):
    def __init__(self, channels=(256, 128, 64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([decoder_block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)

        # return the cropped features
        return encFeatures

    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)

            # crop the current features from the encoder blocks, concatenate them with the current upsampled features,
            # and pass the concatenated output through the current decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x


class GFBUNet(Module):
    def __init__(self, imgChannels=(3, 16, 32, 64, 128, 256), dsmChannels=(1, 16, 32, 64, 128, 256),
                 decChannels=(256, 128, 64, 32, 16), n_classes=2, retainDim=True, 
                 outSize=(512, 512), mode='Train'):
        super().__init__()
        self.mode = mode

        # initialize the encoder and decoder
        self.imgEncoder = Img_Encoder(imgChannels)
        self.dsmEncoder = Encoder(dsmChannels)
        self.imgDecoder = Decoder(decChannels)
        self.dsmDecoder = GDecoder(decChannels)
        self.boundary = Boundary_Enchance(imgChannels)
        self.hed = HED(imgChannels)

        self.finalGate = Gated_Fusion(decChannels[-1])

        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], n_classes, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x, y):
        # grab the features from the encoder
        imgFeatures = self.imgEncoder(x)
        bounFeatures = self.hed(x)
        dsmFeatures = self.dsmEncoder(y)

        # pass the encoder features through decoder making sure that their dimensions are suited for concatenation
        imgdecFeatures = self.imgDecoder(imgFeatures[::-1][0], imgFeatures[::-1][1:])

        dsmdecFeatures = self.dsmDecoder(dsmFeatures[::-1][0], dsmFeatures[::-1][1:], imgFeatures[::-1][1:])
        imgdecFeatures = self.boundary(imgdecFeatures, bounFeatures)

        decFeatures = self.finalGate(imgdecFeatures, dsmdecFeatures)

        # pass the decoder features through the regression head to obtain the segmentation mask
        map = self.head(decFeatures)

        # check to see if we are retaining the original output dimensions and if so,
        # then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)

        # return the segmentation map
        return map


if __name__ == "__main__":
    iH, iW = 128,128
    oH, oW = 256, 256
    a = torch.randn(1, 3, iH, iW)
    b = torch.randn(1, 1, iH, iW)
    
    model = GFBUNet(n_classes=1, outSize=(oH,oW))
    c = model(a, b)
    print(c.shape)
    
    model = GFBUNet(n_classes=1, retainDim=False)
    c = model(a, b)
    print(c.shape)
