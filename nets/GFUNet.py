#!/usr/bin/env python3
"""
-*- coding:utf-8 -*-
demo : GFUNet.py
@author : Wang Shuoyi
Date : 2024/5/20 14:10
        The script contains the definition of our GFUNet model.
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


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, kernel_size=3, padding="same", bias=False)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, kernel_size=3, padding="same", bias=False)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))



class depthwise_separable_conv(Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
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


class Encoder(Module):
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


class GDecoder(Module):
    def __init__(self, channels=(256, 128, 64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([decoder_block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.gated = ModuleList([Gated_Fusion(channels[i + 1]) for i in range(len(channels) - 1)])

    def crop(self, encFeatures, y):
        # grab the dimensions of the inputs, and crop the encoder features to match the dimensions
        (_, _, H, W) = y.shape
        encFeatures = CenterCrop([H, W])(encFeatures)

        # return the cropped features
        return encFeatures

    def forward(self, y, encFeatures, fusionFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            y = self.upconvs[i](y)

            # crop the current features from the encoder blocks, concatenate them with the current upsampled features,
            # and pass the concatenated output through the current decoder block
            encFeat = self.crop(encFeatures[i], y)
            fusionFeat = self.crop(fusionFeatures[i], y)

            encFeatGated = self.gated[i](encFeat, fusionFeat)
            y = torch.cat([y, encFeatGated], dim=1)
            y = self.dec_blocks[i](y)

        # return the final decoder output
        return y


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


class GFUNet(Module):
    def __init__(self, imgChannels=(3, 16, 32, 64, 128, 256), 
                 dsmChannels=(1, 16, 32, 64, 128, 256),
                 decChannels=(256, 128, 64, 32, 16), n_classes=2, 
                 retainDim=True, outSize=(512,512), activation=None):
        super().__init__()

        # initialize the encoder and decoder
        self.imgEncoder = Encoder(imgChannels)
        self.dsmEncoder = Encoder(dsmChannels)
        self.imgDecoder = Decoder(decChannels)
        self.dsmDecoder = GDecoder(decChannels)

        self.finalGate = Gated_Fusion(decChannels[-1])

        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], n_classes, 1)
        self.retainDim = retainDim
        self.outSize = outSize

        if activation is None:
            # don't apply any activation
            self.act = torch.nn.Identity(32)

        else:
            if activation == 'sigmoid':
                # this is for binary segmentation
                self.act = torch.nn.Sigmoid()

            elif activation == 'softmax':
                # this is for multi-class segmentation
                self.act = torch.nn.Softmax(dim=1)

            else:
                raise ValueError('Activation should be sigmoid/softmax.')

    def forward(self, x, y):
        # grab the features from the encoder
        imgFeatures = self.imgEncoder(x)
        dsmFeatures = self.dsmEncoder(y)

        # pass the encoder features through decoder making sure that their dimensions are suited for concatenation
        imgdecFeatures = self.imgDecoder(imgFeatures[::-1][0], imgFeatures[::-1][1:])
        dsmdecFeatures = self.dsmDecoder(dsmFeatures[::-1][0], dsmFeatures[::-1][1:], imgFeatures[::-1][1:])

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
    model = GFUNet(n_classes=1, retainDim=False) #outSize=(oH,oW))
    a = torch.randn(1, 3, iH, iW)
    b = torch.randn(1, 1, iH, iW)
    c = model(a, b)
    print(c.shape)

