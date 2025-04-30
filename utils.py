# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:40:51 2024

@author: renxi
"""
import torch
import matplotlib.pylab as plt

#from models.FuseNet import FuseNet
#from models.CMGFNet import CMGFNet
#from models.GFBUNet import GFBUNet
#from models.GFUNet import GFUNet
#from models.vFuseNet import v_Fusenet


def log_csv(train_loss, val_loss, train_score, val_score, fpath):
    with open(fpath, 'w') as fo:
        print('train_loss, val_loss, train_score, val_score', file=fo)
        for l1, l2, s1, s2 in zip(train_loss, val_loss, train_score, val_score):
            print('%.3f, %.3f, %3f, %3f' % (l1,l2,s1,s2), file=fo)
            
            
def plot_train_val_info(data, des=[], xlab='Epoch', ylab='Loss', 
                        title = 'Train and val Loss', save_path = ''):            
    plt.style.use("ggplot")
    plt.figure()
    
    n_curs = len(data)
    for idx in range(n_curs):
        dt = data[idx]
        if idx < len(des):
            label = des[idx]
        else:
            label = 'untitled'    
        plt.plot(dt, label=label)
        
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc="lower left")
    plt.savefig(save_path, dpi=200)            

    
# save model and auxiliary information
def save_seg_model(fpath, model, arct,  
                   n_classes, class_names,
                   in_channels = 3,
                   img_sz = 512):
    torch.save({
            'n_classes': n_classes,
            'class_names': class_names, 
            'in_channels': in_channels,
            'img_sz': img_sz,
            'arct': arct,
            'model_state_dict': model.state_dict(),                        
            },  fpath)


'''
# save model from model filepath
def load_seg_model(fpath, device='cuda'):
    # load the model and the trained weights
    mdict = torch.load(fpath, map_location=device)
    
    n_classes = mdict['n_classes']
    class_names = mdict['class_names']
    in_channels = mdict['in_channels']
    arct = mdict['arct']
    encoder = mdict['encoder']

    if 'img_sz' in mdict.keys():
        img_sz = mdict['img_sz']
    else:
        img_sz = None
    
    
    model.load_state_dict(mdict['model_state_dict'])
    
    return (model, model_name, n_classes, class_names,
            in_channels, img_sz)
'''

# save training check point
def save_checkpoint(fpath, model, arct,  
               optimizer_name, optimizer,
               n_classes, class_names,
               n_channels, img_sz,
               epochs, batch_size,
               lr, momentum, weight_decay,
               train_losses, val_losses, train_scores, val_scores):
    torch.save({
            'n_classes': n_classes,
            'class_names': class_names, 
            'n_channels': n_channels,
            'img_sz': img_sz,
            
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            
            'arct': arct,
            'model_state_dict': model.state_dict(),            
 
            'opti_name': optimizer_name,
            'opti_state_dict': optimizer.state_dict(),            

            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_scores': train_scores,
            'val_scores': val_scores,
            },  fpath)
    

    
