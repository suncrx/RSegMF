# class of multiple-modal dataset

# import the necessary packages
import os, sys
import cv2
import rasterio
#from rasterio.enums import Resampling

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from dataug import get_train_aug, get_val_aug

'''
torch.utils.data.Dataset is an abstract class representing a dataset. 
Your custom dataset should inherit Dataset and override the following 
methods:

    __len__ : so that len(dataset) returns the size of the dataset.

    __getitem__: to support the indexing such that dataset[i] can be 
                 used to get i-th sample.

'''
# "mean": [0.485, 0.456, 0.406],
# "std":  [0.229, 0.224, 0.225],

#image mean and std from ImageNet
data_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
data_std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(3,1,1)


# Dataset for segmentation
# return numpy image (C, H, W)
# and numpy mask (H, W)
class SegDatasetMM(Dataset):
    '''
    data_dir: The directory that contains images and masks subdirectories.
    '''
    def __init__(self, img_dir, mode="train", 
                 n_classes=2,                  
                 imgH=None, imgW=None,
                 channel_indice=None,
                 apply_aug=False):
        
        assert mode in {"train", "val", "test"}
        
        self.mode = mode
        self.imgW = imgW
        self.imgH = imgH                
        self.n_classes = n_classes
        self.channel_indice = channel_indice

        if apply_aug:
            if mode == 'train':
                self.aug = get_train_aug(height=imgH, width=imgW)
            else:
                self.aug = get_val_aug(height=imgH, width=imgW)
        else:
            self.aug = None
            
              
        self.imgs_dir = img_dir        
        if not os.path.exists(self.imgs_dir):
            print("ERROR: Cannot find directory " + self.imgs_dir)
            sys.exit()
        # mask-image directory
        self.msks_dir = os.path.join(os.path.dirname(self.imgs_dir), "masks")
        # auxiliary data directory
        self.auxs_dir = os.path.join(os.path.dirname(self.imgs_dir), "ndwi")
        
        # search image and mask files
        print('Scanning files in %s ... ' % self.mode)
        print(' ' + self.imgs_dir)
        print(' ' + self.msks_dir)        
        print(' ' + self.auxs_dir)        
        self.imgPairs = self._list_files()
        print(" #image pairs: ", len(self.imgPairs))



    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imgPairs)


    # return a tuple  (image, mask, aux)
    # image: tensor image with shape (C, H, W), and data range (0 ~ 1.0)
    # mask: binary mask image of size (H, W), with value 0 and 1.0.
    # aux: auxiallary data: tensor of size (H, W), with value 0 and 1.0.
    def __getitem__(self, idx):
        # grab the image and mask path from the current index
        imagePath = self.imgPairs[idx]['image']
        maskPath = self.imgPairs[idx]['mask']
        auxiPath = self.imgPairs[idx]['aux']
                
        # 1)Reading image using rasterio package        
        with rasterio.open(imagePath) as imgd:
            # returned image in format [C, H, W]
            ori_image = imgd.read()
            nbands = imgd.count
            # extract the selected bands    
            if self.channel_indice is not None:
                ori_image = ori_image[self.channel_indice]
                nbands = len(self.channel_indice)
            # transpose to [H,W,C]    
            #ori_image = ori_image.transpose(1,2,0)                
            ori_image = np.moveaxis(ori_image, 0, 2)                
            
                            
        # 2)Reading the associated mask from disk in grayscale mode
        if os.path.exists(maskPath):
            with rasterio.open(maskPath) as mskd:
                ori_mask = mskd.read()
                ori_mask = ori_mask[0]                        
        else:
            ori_mask = np.zeros(ori_image.shape[0:2], dtype=np.uint8)
        
        # 3)Reading the auxiliary data
        if os.path.exists(auxiPath):            
             with rasterio.open(auxiPath) as auxd:
                 ori_aux = auxd.read()                        
                 ori_aux = ori_aux[0]    
        else:
             ori_aux = np.ones(ori_image.shape[0:2], dtype=np.uint8)
            
        
        image, mask, aux = ori_image, ori_mask, ori_aux
        # resize image and mask when necessary
        if self.imgH is not None and self.imgW is not None:
            if (self.imgH, self.imgW) != ori_mask.shape:
                image = cv2.resize(ori_image, (self.imgH, self.imgW), 
                                   cv2.INTER_LINEAR)
                mask = cv2.resize(ori_mask, (self.imgH, self.imgW), 
                                   cv2.INTER_NEAREST)
                aux = cv2.resize(ori_aux, (self.imgH, self.imgW), 
                                   cv2.INTER_LINEAR)
        
        # [1] convert image to tensor of shape [C, H, W], with 
        #     values between(0, 1.0)
        if image.dtype == 'uint8':
            ts_image = transforms.ToTensor()(image)
        else:
            raise 'ERROR: Input images should be in uint8 format.'
            
        # [2] convert auxiliary data to tensor of shape [C, H, W], with 
        #     values between(0, 1.0)    
        if aux.dtype == 'uint8':
            ts_aux = transforms.ToTensor()(aux)
        else:
            raise 'ERROR: Input auxiliary data should be in uint8 format.'
                               
        # [3] transform mask to tensor
        # binary segmentation            
        if self.n_classes < 2:            
            # convert to (0, 1) float 
            ts_mask = transforms.ToTensor()(mask > 0).float()              
        # multi-class segmentation
        else:           
            # convert label mask to one-hot tensor
            ts_mask = torch.squeeze(transforms.ToTensor()(mask).long())
            ts_mask = F.one_hot(ts_mask, num_classes=self.n_classes)
            ts_mask = torch.movedim(ts_mask, 2, 0)
            
        return {'image':ts_image, 'mask':ts_mask, 'aux':ts_aux,
                'ori_image':image, 'ori_mask':mask, 'ori_aux':aux,
                'image_path':imagePath,'mask_path':maskPath,'aux_path':auxiPath}
        
     
    #def get_image_and_mask_path(self, idx):
    #    return (self.imgPairs[idx]['image'], self.imgPairs[idx]['mask'])
    
    
    # get all image filepaths   
    #def get_image_filepaths(self):
    #    return [item['image'] for item in self.imgPairs]
    
    
    def _list_files(self):
        #EXTS = ['.png', '.bmp', '.gif', '.jpg', '.jpeg']
        img_files = os.listdir(self.imgs_dir)
        msk_files = os.listdir(self.msks_dir) if os.path.exists(self.msks_dir) else [] 
        aux_files = os.listdir(self.auxs_dir) if os.path.exists(self.auxs_dir) else [] 
        
        def get_bname_exts(parent_dir, filenames):
            bnames, exts = [], []
            for fn in filenames:
                if os.path.isfile(os.path.join(parent_dir, fn)):
                    fname, ext = os.path.splitext(fn)
                    bnames.append(fname)
                    exts.append(ext)
            return bnames, exts

        msk_bnames, msk_extnames = get_bname_exts(self.msks_dir, msk_files)
        aux_bnames, aux_extnames = get_bname_exts(self.auxs_dir, aux_files)
                   
        # extract image, mask and aux data file pairs        
        imgpaths, mskpaths, auxpaths = [], [], []
        for imgf in img_files:
            path_img = os.path.join(self.imgs_dir, imgf)
            if not os.path.isfile(path_img):
                continue
            
            fname, ext = os.path.splitext(imgf)
            
            # if finding a matched mask file in msk_names
            path_msk = ''
            if fname in msk_bnames:
                idx = msk_bnames.index(fname)
                path_msk = os.path.join(self.msks_dir, fname + msk_extnames[idx])                
            
            # if finding a matched aux file in aux_names
            path_aux = ''
            if fname in aux_bnames:
                idx = aux_bnames.index(fname)
                path_aux = os.path.join(self.auxs_dir, fname + aux_extnames[idx])
                
            imgpaths.append(path_img)
            mskpaths.append(path_msk)                        
            auxpaths.append(path_aux)                        
        
        #make image pairs list
        imgPairs = [{'image':fp1, 'mask':fp2, 'aux':fp3} 
                    for fp1, fp2, fp3 in zip(imgpaths, mskpaths, auxpaths)]    
        
        return imgPairs                                  



    
if __name__ == '__main__':
    import matplotlib.pylab as plt

    img_dir = 'D:\\GeoData\\DLData\\buildings\\train\\images'  
    img_dir = 'D:\\Work\\DLearn\\DLWater\\train_data\\wat_nj_mb\\train\\images'        
    
    ds = SegDatasetMM(img_dir, 'train', n_classes=1, 
                     imgH=256, imgW=256,
                     channel_indice=[0, 1, 2],
                     apply_aug=False, 
                     #apply_zero_score=False,
                     #sub_size=1.0,
                     )        
    for i in range(5):        
        samp = ds[i]
        img, msk, aux = samp['image'], samp['mask'], samp['aux']
        
        #print('original image: ', oimg.shape, omsk.shape)
        print('transformed image: ', img.shape, msk.shape)
        
        #plt.figure()
        
        fig, axs = plt.subplots(1,3)
        
        axs[0].imshow(img.numpy()[0])   
        axs[0].set_title('image')
        
        axs[1].imshow(msk.numpy()[0])   
        axs[1].set_title('mask')
        
        axs[2].imshow(aux.numpy()[0])   
        axs[2].set_title('aux data')

        plt.show()                
    
