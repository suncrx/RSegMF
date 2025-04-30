# -*- coding: utf-8 -*-
# @Author  : Renxi Chen
# @File    : train.py
# coding=utf-8

import os
import time
import yaml
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler

from nets.GFBUNet import GFBUNet
from nets.GFUNet import GFUNet
from nets.FuseNet import FuseNet
from nets.vFuseNet import v_Fusenet
from nets.CMGFNet import CMGFNet

from datasetx import SegDatasetX
#from utils.utils import *

'''
from IPython.display import clear_output
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
'''

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="data/water_nj.yaml",
                        help='dataset yaml path')
    
    parser.add_argument('--out_dir', type=str, default='',
                        help='training output path')
     
    parser.add_argument('--arct', type=str, default='FuseNet', #'GFUNet',
                        help='model architecture (options:GFUNet, GFBUNet, FuseNet, v_Fusenet, CMGFNet')

    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=4,
                        help='total batch size for all GPUs, -1 for autobatch')
    
    parser.add_argument('--img_sz', '--img', '--img-size', type=int, 
                        default=256,
                        help='train, val image size (pixels)')

    parser.add_argument('--save_period', type=int, default=5,
                        help='check point saving period')
       
    return parser.parse_args()


# create the model from the name 'arct' and pretrained_weight_path 
def create_model(arct, pretrained_weight_path):
    model_class = globals()[opt.arct]
    model = model_class()
    # Download VGG-16 weights from PyTorch
    # vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'     
    # if not os.path.isfile('../vgg16_bn-6c64b313.pth'):
    # weights = URLopener().retrieve(vgg_url, '../vgg16_bn-6c64b313.pth')
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


# one pass of train and validation
def one_epoch(net, data_loader, opt, loss_fun,   
              is_training=True, device='cuda', 
              scheduler=None, weights=None):
    if is_training:
        net.train()
    else:
        net.eval()
        
    # loop over the dataset
    loss_mean = 0
    losses = []
    with tqdm(data_loader) as iterator:
        for batch in iterator:
            img, target, aux = batch['image'], batch['mask'], batch['aux']
            img = img.to(device)
            target = target.to(device)
            aux = aux.to(device)
            
            if is_training:
                # first, zero out any previously accumulated gradients, then
                # perform backpropagation, and then update model parameters
                opt.zero_grad()
                
            output = net(img, aux)

            loss = loss_fun(output, target, weight=weights)
              
            losses.append(loss.data)
            loss_mean = np.mean(np.array(losses))
            
            if is_training:
                loss.backward()
                opt.step()            
    return loss_mean
  
    
def run(opt):
    # %% parameters
    print('Parameters from command: ')
    print(opt, '\n')
    data_yaml_file = opt.data
    img_sz = opt.img_sz
    batch_size, epochs = opt.batch_size, opt.epochs
    arct = opt.arct
    #checkpoint_file = opt.checkpoint_file
    save_period = opt.save_period
    
    # determine the device to be used for training and evaluation
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    
    # read data information from yaml file          
    assert os.path.exists(data_yaml_file)
    with open(data_yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # cfg is a dictionary
    if 'exp_name' in cfg.keys():
        print(cfg['exp_name'])
    print('Data information:', cfg)
    
    # Number of classes
    n_classes = cfg['nclasses']
    class_names = cfg['names']
    # Weights for class balancing
    class_wrights = torch.ones(n_classes)  
    
    #band indexes 
    bands = cfg['bands']
        
    root_data_dir = cfg['path']
    train_folder = cfg['train']
    val_folder = cfg['val']
    # train and val image path
    train_img_dir = os.path.join(root_data_dir, train_folder)
    val_img_dir = os.path.join(root_data_dir, val_folder)
    
    # make output folder
    out_dir = opt.out_dir
    if out_dir == '':
        out_dir = os.path.join('out/weight')
    os.makedirs(out_dir, exist_ok=True)
    
    
    # prepare Dadaset
    train_dataset = SegDatasetX(train_img_dir, "train",
                                n_classes=n_classes, imgH=img_sz, imgW=img_sz,
                                channel_indice=bands)
    val_dataset = SegDatasetX(val_img_dir, "val",
                              n_classes=n_classes, imgH=img_sz, imgW=img_sz,
                              channel_indice=bands)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # prepare model from pretrained vgg16
    net = create_model(arct, 'pretrained_weight/vgg16_bn-6c64b313.pth')    
    #print(arct)
      
    # learning ratio  
    base_lr = 0.01
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params': [value], 'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params': [value], 'lr': base_lr / 2}]

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


    losses_train = []
    losses_val = []
    # training
    time_start = time.time()
    net = net.to(DEV)
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        
        loss = one_epoch(net, train_loader, optimizer, 
                         device = DEV,
                         is_training = True, 
                         scheduler = scheduler, 
                         weights = class_wrights)
        losses_train.append(loss)
        
        loss = one_epoch(net, train_loader, optimizer,
                         device = DEV,
                         is_training = False, 
                         scheduler = scheduler, 
                         weights = class_wrights)
        losses_val.append(loss)
                
        if iter_ % 100 == 0:
            clear_output()
            rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
            pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
            gt = target.data.cpu().numpy()[0]
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                e, epochs, batch_idx, len(data_loader),
                100. * batch_idx / len(data_loader), loss.data, accuracy(pred, gt)))

        if e >= 15 and e % save_period == 0:
        #if  e % save_period == 0:
            # We validate with the largest possible stride for faster computing
            acc = test(net, test_ids, all=False, stride=STRIDE)
            if acc > acc_best:
                torch.save(net.state_dict(), '{}{}_epoch{}_{}'.format(out_dir, arct, e, acc))
                acc_best = acc
    
    time_end = time.time()
    print('Total Time Cost: ', time_end - time_start)   

    torch.save(net.state_dict(), '{}{}_best .pth'.format(out_dir, arct))    
    print("Save final model!!")
    
    


if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    run(opt)