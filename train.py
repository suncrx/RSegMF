# -*- coding: utf-8 -*-
# @Author  : Renxi Chen
# @File    : train.py
# coding=utf-8

import os
import time
import yaml
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp


#from models.GFBUNet import GFBUNet
#from models.GFUNet import GFUNet
#from models.vFuseNet import v_Fusenet
#from models.CMGFNet import CMGFNet

import nets
from datasetmm import SegDatasetMM
from epoch_run import MyTrainEpoch, MyValidEpoch
from utils import log_csv, plot_train_val_info

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="data/water_nj.yaml",
                        help='dataset yaml path')
    
    parser.add_argument('--out_dir', type=str, default='',
                        help='training output path')
     
    parser.add_argument('--arct', type=str, default='FuseNet'   #'GFUNet', 'MFSegNet'
                        help='model architecture (options:GFUNet, GFBUNet, FuseNet, v_Fusenet, CMGFNet, MFSegNet')

    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=4,
                        help='total batch size for all GPUs, -1 for autobatch')
    
    parser.add_argument('--img_sz', '--img', '--img-size', type=int, 
                        default=128,
                        help='train, val image size (pixels)')
    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    
    parser.add_argument('--opti', type=str, default='adamw',
                        help='optimizer, options (adamw, sgd)')

    parser.add_argument('--save_period', type=int, default=5,
                        help='check point saving period')
       
    return parser.parse_args()



# %% main
def run(opt):
    print('Parameters from command: ')
    print(opt, '\n')
    data_yaml_file = opt.data
    img_sz = opt.img_sz
    batch_size, epochs = opt.batch_size, opt.epochs
    
    arct = opt.arct    
    base_lr = opt.lr
    opti_name = opt.opti
    #checkpoint_file = opt.checkpoint_file
    #save_period = opt.save_period
    
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
    #class_names = cfg['names']
    # Weights for class balancing
    #class_weights = torch.ones(n_classes)  
    
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
        #out_dir = os.path.join('out/weights')
        out_dir = os.path.join(root_data_dir, 'out', arct.lower()) 
    os.makedirs(out_dir, exist_ok=True)
    
    
    # prepare Dadaset
    train_dataset = SegDatasetMM(train_img_dir, "train",
                                n_classes=n_classes, imgH=img_sz, imgW=img_sz,
                                channel_indice=bands)
    val_dataset = SegDatasetMM(val_img_dir, "val",
                              n_classes=n_classes, imgH=img_sz, imgW=img_sz,
                              channel_indice=bands)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # prepare model from pretrained vgg16
    model = nets.create_model(arct, n_classes = n_classes, 
                                pretrained_weight_path='pretrained_weight/vgg16_bn-6c64b313.pth')    
    #print(arct)
    
    
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params': [value], 'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params': [value], 'lr': base_lr / 2}]

    # optimizer--------------------------------------------------------------
    if opti_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, 
                              weight_decay=0.0005)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.999],
                                eps=1e-7, amsgrad=False)   
    print(f'Optimizer: {opti_name}')
    print(optimizer)
    
    # define the learning rate scheduler --------------------------------------
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 35, 45], 
                                               gamma=0.5)
    
    # define loss function --------------------------------------------------
    # mode: 'Binary', 'multiclass', 'multilabel'
    #lossFunc = smp.losses.DiceLoss('multiclass', from_logits=True) # >=version 0.3.2
    if n_classes<2:
        lossFunc = smp.utils.losses.DiceLoss(activation='sigmoid') 
    else:
        # For multi-class, logsoftmax is more stable than softmax; the first 
        # channel is background and is ignored.  
        lossFunc = smp.utils.losses.DiceLoss(activation='logsoftmax', 
                                         ignore_channels=[0])

    # monitering metrics
    metrics = [smp.utils.metrics.IoU(threshold=0.5)] # >=version 0.3.2


    # training  ---------------------------------------------------------
    print("[INFO] training the network: {}", arct)

    train_epoch = MyTrainEpoch(model, loss=lossFunc, metrics=metrics,
                               optimizer=optimizer, device=DEV, 
                               verbose=True)
    
    valid_epoch = MyValidEpoch(model, loss=lossFunc, metrics=metrics,
                               device=DEV, verbose=True )
    
    # model file name 
    mfname = '%s_best.pt' % arct        
    
    startTime = time.time()
    max_score = 0
    train_losses, valid_losses =[], []
    train_scores, valid_scores =[], []
    for i in range(1, epochs+1):
        print('\nEpoch: %d/%d' % (i, epochs))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
    	
        train_losses.append(train_logs['dice_loss'])
        valid_losses.append(valid_logs['dice_loss'])        
        tsc = train_logs['iou_score']
        vsc = valid_logs['iou_score']
        train_scores.append(tsc)	
        valid_scores.append(vsc)	
        
        # log file name
        logfilepath = os.path.join(out_dir, arct+'_log.csv')
        log_csv(train_losses, valid_losses, 
                train_scores, valid_scores, logfilepath)
        
        #plot train and val information
        plot_train_val_info([train_losses, valid_losses], ['train_loss','val_loss'],
                            xlab='epoch', ylab='loss', 
                            title=f'train and val loss - {arct}', 
                            save_path=os.path.join(out_dir, arct + '_loss.png'))
        plot_train_val_info([train_scores, valid_scores], ['train_IoU','val_IoU'],
                            xlab='epoch', ylab='IoU', 
                            title=f'train and val IoU - {arct}', 
                            save_path=os.path.join(out_dir, arct + '_IoU.png'))

        if max_score < vsc:
            max_score = vsc        
            #torch.save(model, mdlpath)
            mdlpath = os.path.join(out_dir, mfname)
            torch.save(model.state_dict(), mdlpath)
            print('Best model saved:', mdlpath)
        # adjust the learning rate
        #scheduler.step()

    endTime = time.time()
    print("[INFO] total time : {:.2f}s".format(endTime - startTime))
    print('Model saved :' + mdlpath) 
    print('Done!')


#%% run here
if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    run(opt)