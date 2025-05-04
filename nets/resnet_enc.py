# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:16:39 2024

This is a customed resnet encoder which modified from the pytorch resnet encoder
that can only handle 3-channel inputs, whereas ours can handle any channel number.

e.g. :   m = resnet50_enc(pretrained=True, in_channels=6)

@author: renxi
"""
from typing import Type, List, Union

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

#from torchvision.models.resnet import conv1x1
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import ResNet

resnet_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': "https://download.pytorch.org/models/resnet34-b627a593.pth",
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

class ResNetEnc(ResNet):
    def __init__(self, 
                block: Type[Union[BasicBlock, Bottleneck]],
                layers: List[int],
                in_channels: int=3, 
                **kwargs):

        super().__init__(block, layers, **kwargs)
                       
        #remove the avgpool and fc layers
        del self.avgpool
        del self.fc

        # input channels 
        self.in_channels = in_channels        
        #modify the first conv layer: the input channels is 3 in the original net,
        #and our input channels can be any number. 
        self.conv1 = nn.Conv2d(in_channels, 64, 
                               kernel_size=7, stride=2, padding=3, bias=False)



    def load_state_dict(self, state_dict, **kwargs):
        #fc layer is removed from the resnet, and the weights need to be
        #removed from the dict correspondingly.
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        
        #The default in_channels of conv1 layer is 3. 
        #If the in_channels is changed to 1,2,or others, we need to
        #modify the structure of the new conv1 layer in the state_dict.
        if self.in_channels < 3:
            new_weight = state_dict['conv1.weight']
            new_weight = new_weight[:,0:self.in_channels]
            state_dict['conv1.weight'] = nn.parameter.Parameter(new_weight)
            
        elif self.in_channels > 3:
            weight = state_dict['conv1.weight']
            out_channels, ks1, ks2 = weight.shape[0],weight.shape[2],weight.shape[3]
            
            new_weight = torch.Tensor(out_channels, self.in_channels, ks1, ks2)
            for i in range(self.in_channels):
                new_weight[:, i] = weight[:, i % 3]

            new_weight = new_weight * (3 / self.in_channels)
            state_dict['conv1.weight'] = nn.parameter.Parameter(new_weight)
            
        super().load_state_dict(state_dict, **kwargs)


    def forward(self, x):
        features = [x]
                
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        
        x = self.maxpool(x)                
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)

        return features
    

def resnet18_enc(in_channels=3, pretrained=True, **kwargs):
    model = ResNetEnc(BasicBlock, [2, 2, 2, 2], 
                      in_channels=in_channels, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet18'])
        model.load_state_dict(state_dict) #, strict=False)
    return model

def resnet34_enc(in_channels=3, pretrained=True, **kwargs):
    model = ResNetEnc(BasicBlock, [3, 4, 6, 3], 
                      in_channels=in_channels, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet34'])
        model.load_state_dict(state_dict)
    return model


def resnet50_enc(in_channels=3, pretrained=True, **kwargs):
    model = ResNetEnc(Bottleneck, [3, 4, 6, 3], 
                      in_channels=in_channels, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet50'])
        model.load_state_dict(state_dict)
    return model


def resnet101_enc(in_channels=3, pretrained=True, **kwargs):
    ''' ResNet-101 Model'''
    model = ResNetEnc(Bottleneck, [3, 4, 23, 3], 
                      in_channels=in_channels, **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet101'])
        #model.load_state_dict(state_dict)
        pass
    return model


def resnet152_enc(in_channels=3, pretrained=True, **kwargs):
    model = ResNetEnc(Bottleneck, [3, 8, 36, 3], 
                      in_channels=in_channels, **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet152'])
        #model.load_state_dict(state_dict)
        pass
    return model


if __name__ == '__main__':    
    m = resnet50_enc(pretrained=True, in_channels=6)
    print(m)
    print(m.conv1.weight[0,0])
    d = torch.rand(4,6,256,256)
    fts = m(d)
    for ft in fts:
        print(ft.shape)
        