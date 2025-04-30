# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:09:45 2024

@author: renxi
"""
import os
import numpy as np
import rasterio
import matplotlib.pylab as plt
import cv2

data_dir = 'D:\\Work\\DLearn\\DLWater\\train_data\\wat_nj_mb\\train'
data_dir = 'D:\\Work\\DLearn\\DLWater\\train_data\\wat_nj_mb\\val'

img_dir = os.path.join(data_dir, "images")

ndwi_dir = os.path.join(data_dir, "ndwi")
os.makedirs(ndwi_dir, exist_ok=True)

files = os.listdir(img_dir)
for fn in files:
     imagePath = os.path.join(img_dir, fn) 
     print('processing : ', fn)
     with rasterio.open(imagePath) as imgd:
         # image stored in format [C, H, W]
         image = imgd.read()
         nbands = imgd.count
         # transpose to [H,W,C]    
         #image = image.transpose(1,2,0)                
         B8 = np.float32(image[7,:,:])
         B3 = np.float32(image[2,:,:])
         ndwi = (B3-B8)/(B8+B3)
         ndwi = np.uint8((ndwi+1)*255/2)
         outpath = os.path.join(ndwi_dir, fn)
         cv2.imwrite(outpath, ndwi)         