# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:46:44 2024

@author: renxi
"""

import torch

from .FuseNet import FuseNet
from .vFuseNet import v_Fusenet
from .CMFNet import CMFNet
from .CMGFNet import CMGFNet
from .GFUNet import GFUNet
from .GFBUNet import GFBUNet
from .MFSegNet import MFSegNet
from .MFSegNet2 import MFSegNet2

# create the model from the name 'arct' and pretrained_weight_path 
def create_model(arct, n_classes=1, n_channels=3, pretrained_weight_path=None):
    model = None
    if arct.lower()=='fusenet':
        model = FuseNet(num_labels=n_classes)
    elif arct.lower()=='cmgfnet':
        model = CMGFNet(n_classes=n_classes)
    elif arct.lower()=='gfunet':
        model = GFUNet(n_classes=n_classes, retainDim=False)        
    elif arct.lower()=='gfbunet':
        model = GFBUNet(n_classes=n_classes, retainDim=False)        
    elif arct.lower()=='mfsegnet':
        model = MFSegNet(n_classes=n_classes, img_channels=n_channels, 
                          aux_channels=1)
    elif arct.lower()=='mfsegnet2':
        model = MFSegNet2(n_classes=n_classes, img_channels=n_channels, 
                          aux_channels=1)    
    #model_class = globals()[opt.arct]
    #model = model_class()
        
    
    if model is None:
        raise Exception(f'ERROR: There is no model named {arct}.')        
    # Download VGG-16 weights from PyTorch
    # vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'     
    # if not os.path.isfile('../vgg16_bn-6c64b313.pth'):
    # weights = URLopener().retrieve(vgg_url, '../vgg16_bn-6c64b313.pth')
    if pretrained_weight_path is not None:
        vgg16_weights = torch.load(pretrained_weight_path)
        mapped_weights = {}
        for k_vgg, k_segnet in zip(vgg16_weights.keys(), model.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
    
        for it in model.state_dict().keys():
    
            if it == 'conv1_1_d.weight':
                avg = torch.mean(mapped_weights[it.replace('_d', '')].data, dim=1)
                mapped_weights[it] = avg.unsqueeze(1)
            if '_d' in it and it != 'conv1_1_d.weight':
                if it.replace('_d', '') in mapped_weights:
                    mapped_weights[it] = mapped_weights[it.replace('_d', '')]
    
        try:
            model.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass
    
    return model