'''
Validate the trained model and visualize the results.

#USAGE
# python validate.py --model_file 
# "out/weights/FuseNet_best.pt" 
# --data_dir "../train_data/water_nj_rgb/val/images" --conf 0.5 --img_sz 256
# --plot True
#

conf = 0.75 when CMGFNet
'''

# import the necessary packages
# %% import installed packages
import os, sys
import argparse
import shutil
import tqdm

import cv2
import numpy as np
import torch
import matplotlib.pylab as plt

import segmentation_models_pytorch as smp

#from models.FuseNet import FuseNet
from nets import create_model
from datasetmm import SegDatasetMM


# determine the device to be used for training and evaluation
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print('Device : ', DEV)


#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str,
                        default = 'D:/dlwater/train_data/wat_nj_rgb/out/trained_models/cmgfnet/CMGFNet_best.pt',
                        help='model filepath')

    parser.add_argument('--data_dir', type=str,
                        default='D:/dlwater/train_data/wat_nj_rgb/val/images',
                        help='root directory where the val folder resides.')
    
    parser.add_argument('--arct', type=str,
                        default='CMGFNet',
                        help='Model architecture')
    
    parser.add_argument('--n_classes', type=int,
                        default=1,
                        help='number of classes')
    
    parser.add_argument('--img_sz', type=int,
                        default=256, 
                        help='input image size (pixels)')

    #parser.add_argument('--out_dir', type=str, default=ROOT / 'out', 
    #                    help='training output path')    

    parser.add_argument('--conf', type=float, default=0.75,
                        help='test confidence')

    parser.add_argument('--plot', type=int, default=1,
                        help='Plot the results or not')

    parser.add_argument('--verbose', type=int, default=0,
                        help='print processing informaton or not')
    return parser.parse_args()


# calculate the evaluation metrics between the predicted mask and ground-truth
def eval_metrics(predMask, gtMask, n_classes):
    from segmentation_models_pytorch import utils as smp_utils
    if n_classes > 1:
        pm = [predMask == v for v in range(n_classes)]
        gm = [gtMask == v for v in range(n_classes)]
        pm, gm = np.array(pm), np.array(gm)
    else:
        pm, gm = np.uint8(predMask > 0), np.uint8(gtMask > 0)

    iou = smp_utils.metrics.IoU()  #for smp >= 0.3.2
    #acc = smp_utils.metrics.Accuracy()
    pre = smp_utils.metrics.Precision()
    rec = smp_utils.metrics.Recall()
    fsc = smp_utils.metrics.Fscore()

    iouv = iou.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    #accv = acc.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    prev = pre.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    recv = rec.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    fscv = fsc.forward(torch.as_tensor(pm), torch.as_tensor(gm))

    #return iouv
    return {'iou': iouv, 'pre': prev, 'rec': recv, 'fsc': fscv}


#======================================================================
def make_prediction(model, image, aux, out_H, out_W, binary=False, conf=0.5):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        #padding image size to 32*M
        c, h, w = image.shape
        #image = pad_image_32x(image)

        # apply image transformation. This step turns the image into a tensor.
        # with the shape (1, 3, H, W). See IMG_TRANS in dataset.py
        image = torch.unsqueeze(image, 0)
        aux = torch.unsqueeze(aux, 0)
        
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred = model.forward(image, aux).squeeze()
        # crop prediction size to the original size
        #pred = pred[0:h, 0:w]
        # perform Sigmoid or softmax 
        if binary:
            pred = pred.sigmoid()
            pred = cv2.resize(pred.numpy(), (out_W, out_H))
            pred = np.uint8(pred >= conf)
            #pred = np.uint8(pred*255)            
        else:
            # determine the class by the index with the maximum
            pred = torch.softmax(pred, dim=0)
            pred = np.uint8(torch.argmax(pred, dim=0))
            # resize to the original size
            pred = cv2.resize(pred, (out_W, out_H),
                              interpolation=cv2.INTER_NEAREST)
            # print('Found classes: ', np.unique(pred))
    return pred


#display original image, predicted mask and ground-truth.
def plot_prediction(image, predMask, gtMask=None, 
                    sup_title= 'result', save_path=None, auto_close=False):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3)
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(image)
    ax[1].imshow(predMask)
    if gtMask is not None:
        ax[2].imshow(gtMask)
        
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Ground-truth")
    ax[2].set_title("Prediction")
   
    figure.suptitle(sup_title)#, fontsize=30)
    
    # set the layout of the figure and display it
    #figure.tight_layout()
    #figure.show()
    
    if save_path is not None:
       plt.savefig(save_path, dpi=300)
    
    if auto_close:       
        plt.close()       


def load_val_ids(fpath):
    # read validation file ids        
    if os.path.exists(fpath):
        print('loaded val file name ids.')
        with open(fpath) as fo:
            lines = fo.readlines()
            val_bnames = [ss[0:-1] for ss in lines]
    else:
        val_bnames = []
    return val_bnames

        
#%% main 
def run(opt):
    # get parameters
    print(opt)    
    model_file = opt.model_file
    img_dir = opt.data_dir
    #out_dir = opt.out_dir
    
    arct = opt.arct
    n_classes = opt.n_classes
    img_sz = opt.img_sz       
    conf, plot = opt.conf, opt.plot
    verb = opt.verbose

    if not os.path.exists(model_file):
        raise Exception(f"Can not find model path: {model_file}")
    
        
    # make output folders
    model_basename = os.path.basename(model_file)
    # make output folder for predicting images
    # get the grand-parent folder of '../../images'
    data_dir = os.path.dirname(os.path.dirname(img_dir))
    outpred_dir = os.path.join(data_dir, 'out', 'val_pred', model_basename)
    if os.path.exists(outpred_dir):
        shutil.rmtree(outpred_dir)
    os.makedirs(outpred_dir, exist_ok=True)

    #--------------------------------------------------------------------    
    # load our model from disk and flash it to the current device
    print(f'Loading model: {model_file}')
    '''
    model = None
    if arct=='FuseNet':
        model = FuseNet(num_labels=n_classes)
    '''
    model = create_model(arct, n_classes=n_classes) #, pretrained_weight_path)
    if model is None:
        print(f'There is no model named {arct}.')
        sys.exit(0)
    # load weights from the model file    
    model.load_state_dict(torch.load(model_file, map_location=DEV))
    
        
    #---------------------------------------------------------------------------
    # load the image paths in our testing directory and
    # randomly select 10 image paths
    print("Loading validation images ...")
    testDS = SegDatasetMM(img_dir, mode="test",
                         n_classes=n_classes,
                         imgH=img_sz, imgW=img_sz,
                         apply_aug=False)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    nm = len(testDS)
    #nm = min(len(testDS), 10)
    metrics = []
    imgnames = []
    # read validation file ids    
    val_bnames = load_val_ids(os.path.join(os.path.dirname(img_dir), 'val_ids2.txt'))
    
    for i in tqdm.tqdm(range(nm)):
        #get preprocessed image, mask, and auxiliary data
        iga = testDS[i]
        
        #get the image path
        imgPath = iga['image_path']
        bname = os.path.splitext(os.path.basename(imgPath))[0]
        if len(val_bnames)>0 and not bname in val_bnames:
            continue
        
        img, gt, aux = iga['image'], iga['mask'], iga['aux']
        ori_img = iga['ori_image']
        ori_msk = iga['ori_mask']        
        
        outH, outW = ori_msk.shape[0:2]

        # make predictions and visualize the results
        if verb:
            print('\nPredicting ' + imgPath)
        #for binary segmentation, pred is a uint8-type mask with 0, 1;
        #for multi-class segmentation, pred is a uint8-type mask with
        #class labels: 0, 1, 2, 3, ... , n_class-1
        is_binary = (n_classes < 2)
        pred = make_prediction(model, img, aux, outH, outW,
                               binary=is_binary, conf=conf)
        #evaluation
        res = eval_metrics(pred, ori_msk, n_classes=n_classes)
        iouv, prev = res['iou'], res['pre']
        recv, fscv = res['rec'], res['fsc']
        metrics.append([iouv, prev, recv, fscv])
        if verb:
            print(f'IoU: {iouv:.3f} Prec: {prev:.3f} Rec: {recv:.3f} Fscore: {fscv:.3f}')
        image_basename = os.path.basename(imgPath)
        imgnames.append(image_basename)

        #------------------------------------
        #save and convert results to rgb label for visualization 
        image_basename = os.path.basename(imgPath)
        bname, ext = os.path.splitext(image_basename)
        out_mskPath = os.path.join(outpred_dir, bname + '_msk.png')
        fig_path = os.path.join(outpred_dir, 'plot_' + bname + '.png')
        stitle = f'{image_basename} IoU {iouv:.3f}' 
        #save image
        bgrimg = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(outpred_dir, image_basename), bgrimg)
        if is_binary:
            #save predicted mask (one channel)    
            Mask = np.uint8(pred * 255)
            cv2.imwrite(out_mskPath, Mask)
            #save gt mask
            if ori_msk.max() == 1:
                ori_msk = ori_msk * 255
            cv2.imwrite(os.path.join(outpred_dir, bname + '_gt.png'), ori_msk)
            #plot results
            if plot:
                plot_prediction(ori_img, ori_msk, Mask,
                                sup_title=stitle, save_path=fig_path,
                                auto_close=True)
        else:
            #save predicted mask
            Mask = np.uint8(pred)
            cv2.imwrite(out_mskPath, Mask)
            Mask_rgb = Mask
            out_rgbMskPath = os.path.join(outpred_dir, bname + '_rgb.png')
            cv2.imwrite(out_rgbMskPath, Mask_rgb)
            if plot:
                plot_prediction(ori_img, ori_msk, Mask_rgb,
                                sup_title=stitle, save_path=fig_path,
                                auto_close=True)

            '''
            #convert Mask to rgb label image and save
            RgbMask = label_to_rgbLabel(Mask, label_colors)
            BgrMask = cv2.cvtColor(RgbMask, cv2.COLOR_RGB2BGR)            
            cv2.imwrite(out_rgbMskPath, BgrMask)
            print('Saved: %s' % out_rgbMskPath)
            
            #convert ground-truth mask to rgb label image
            gtRgbMask = label_to_rgbLabel(ori_gtMask, label_colors)            
            '''

    #Evaluation metrics 
    Mm = np.array(metrics)
    maxv = Mm.max(axis=0)
    minv = Mm.min(axis=0)
    meanv = Mm.mean(axis=0)
    print('\nname,  Iou,  Precision, Recall, Fscore')
    print('Max,  %.3f, %.3f, %.3f, %.3f' %
          (maxv[0], maxv[1], maxv[2], maxv[3]))
    print('Min,  %.3f, %.3f, %.3f, %.3f' %
          (minv[0], minv[1], minv[2], minv[3]))
    print('Mean,  %.3f, %.3f, %.3f, %.3f' %
          (meanv[0], meanv[1], meanv[2], meanv[3]))
    #plt.figure()
    #plt.plot(Mm[0], '.')
    #plt.show()        
    print('Done!')
    print('Results saved: %s' % outpred_dir)

    #write metrics to log file
    logfn = os.path.join(outpred_dir, model_basename + '_log.txt')
    with open(logfn, 'w') as fo:
        print('\nname,  Iou,  Precision, Recall, Fscore', file=fo)
        for i in range(Mm.shape[0]):
            print('%s, %.3f, %.3f, %.3f, %.3f' %
                  (imgnames[i], Mm[i][0], Mm[i][1], Mm[i][2], Mm[i][3]),
                  file=fo)
        print('%s, %.3f, %.3f, %.3f, %.3f' %
              ('Max', maxv[0], maxv[1], maxv[2], maxv[3]), file=fo)
        print('%s, %.3f, %.3f, %.3f, %.3f' %
              ('Min', minv[0], minv[1], minv[2], minv[3]), file=fo)
        print('%s, %.3f, %.3f, %.3f, %.3f' %
              ('Mean', meanv[0], meanv[1], meanv[2], meanv[3]), file=fo)



#%% run here
if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
