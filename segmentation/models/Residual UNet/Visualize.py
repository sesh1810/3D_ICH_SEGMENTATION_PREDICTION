from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
#from networks import define_G, define_D, GANLoss, print_network
import torch.nn as nn

#from util import is_image_file, load_img, save_img
from skimage.io import imread, imsave
from skimage import io
from glob import glob
import SimpleITK as sitk
import nibabel as nib
from math import log10
#from model import Noise2NoiseUNet3D
from UNet_Residual import ResidualUNet3D

import torch.nn.functional as F

#System

import math
import sys
import os
import random
import time
from numpy.core.umath_tests import inner1d
import random
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = {
    'num_class': 1,
    'ignore_label': 255,
    'num_gpus': 1,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 1,
    'lr': 0.005,
    'lr_decay': 0.9,
    'dice': 0,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
    'pred_dir':'predicted',
}
ckpt_path = '/ckpt/Residual_Random_Crop/'
gpu_ids = range(args['num_gpus'])

model = ResidualUNet3D(in_channels=1, out_channels=2, final_sigmoid=True)
model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
model = model.cuda()
real_a = torch.FloatTensor(args['batch_size'], 1,136,186,186)
real_a = Variable(real_a).cuda()

for epochs in range(69, 70):
    my_model = 'ckpt/Residual_Random_Crop/epoch_' + str(epochs) + '.pth.tar'
    model.load_state_dict(torch.load(my_model))
    model.eval()
    
    #ND_dir = sorted(glob('/home/mobarak/uu/ICH/Registration_Final/CT_VALID_SS/101/101ct1_resampled.nii.gz'))
    #ND_dir1 = sorted(glob('/home/mobarak/uu/ICH/Registration_Final/CT_VALID_SS/101/101ct2_resampled.nii.gz'))
    ND_dir = glob('1ct1_resampled.nii.gz')
    ND_dir1 = glob('2ct1_resampled.nii.gz')
    path = ND_dir
    path1 = ND_dir1
    print(path)
    print(path1)
    
    #t1 = glob(path[0] + '/*ct1_resampled.nii.gz')
    #real_a_cpu, real_b_cpu = batch[0], batch[1]  
    #print(real_a_cpu.size())
    _img1 = nib.load(path[0])
    _img21 = nib.load(path1[0])

    _img = _img1.get_data()
    _img2 = _img21.get_data()
    print(np.max(_img))

    _img = _img.transpose(2,0,1)
    _img2 = _img2.transpose(2,0,1)

    #_img = _img/255
    #_img2 = _img2/255

    
    #print(np.min(_img))
    
    _img = np.expand_dims(_img, axis=0)
    _img = np.expand_dims(_img, axis=0)

    _img2 = np.expand_dims(_img2, axis=0)
    _img2 = np.expand_dims(_img2, axis=0)

    
    
    with torch.no_grad():
        real_a = _img
        real_b = _img2
        real_a = torch.from_numpy(real_a)
        real_b = torch.from_numpy(real_b)


        #print(torch.min(real_a))
        #print(torch.min(real_b))
        #print(torch.max(real_a))
        #print(torch.max(real_b))
        #real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        
        #real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        np.unique(real_a)
        #fake_b = netG(real_a)
        seg_a = model(real_a)    
        #fake_b[real_b==0] = 0
        img_pred = seg_a.data.max(1)[1].squeeze_(1).cpu().numpy()
        print(np.unique(img_pred))
        img_pred = np.squeeze(img_pred, axis=0).transpose(1,2,0)
        print(img_pred.shape)

        segmented_img = nib.Nifti1Image(img_pred, _img1.affine, _img1.header)
        nib.save(segmented_img,str(epochs)+'UU.nii.gz')
    
        #img_pred= torch.squeeze(fake_b,0).cpu()
        #print(seg_a.size())
        #print(seg_a.size())
        #img_pred= torch.squeeze(seg_a,0).cpu()
        #real_b= torch.squeeze(real_b,0).cpu()
        #print(img_pred.size())
        #output= torch.squeeze(img_pred,0).cpu()
        
        #truth= torch.squeeze(real_b,0).cpu()
        #img_pred= torch.squeeze(seg_a,0).cpu()
        #print('gfhfhg',img_pred.size())
        #img_pred= torch.squeeze(img_pred,0).cpu()
        #img_pred[img_pred<0] = 0
'''
        img = np.zeros((1,138,186,186))
        img_zero = img_pred[0,0:138,0:186,0:186]
        img_one = img_pred[1,0:138,0:186,0:186]'''
        #target = np.zeros((138,186,186))
        #print(torch.max(output))
        #print(torch.min(output))
        #img = img_pred

        # mse = criterionMSE(output,truth)
        # psnr = 10 * log10(1 / mse.item())
        #print(img_pred.size())
'''
        output_zero = img_zero.numpy()
        output_zero = output_zero.transpose(1,2,0)
        #print(np.unique(seg_a))
        #print("LOL")
        #print(output_zero)
        
        #n_img = nib.Nifti1Image(output1, _img1.affine, _img1.header)
        segmented_img = nib.Nifti1Image(output_zero, _img1.affine, _img1.header)
        nib.save(segmented_img,str(epochs)+'ct1_segmented_op_zero.nii.gz')

        output_one = img_one.numpy()
        output_one = output_one.transpose(1,2,0)
        #print("LOL")
        #print(output_one)

        #n_img = nib.Nifti1Image(output1, _img1.affine, _img1.header)
        segmented_img = nib.Nifti1Image(output_one, _img1.affine, _img1.header)
        nib.save(segmented_img,str(epochs)+'ct1_segmented_op_one.nii.gz')
        #print("Done")
        #break'''
        