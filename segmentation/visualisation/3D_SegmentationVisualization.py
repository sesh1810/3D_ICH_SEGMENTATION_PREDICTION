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
#from UNet_Residual import ResidualUNet3D
#from UNet_3D import UNet3D

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
ckpt_path = '/ckpt/UNet3d/'
gpu_ids = range(args['num_gpus'])

#model = UNet3D(in_channels=1, out_channels=2, final_sigmoid=True)
#model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
#model = model.cuda()
real_a = torch.FloatTensor(args['batch_size'], 1,136,186,186)
real_a = Variable(real_a).cuda()

#for epochs in range(81, 82):
    #my_model = 'ckpt/UNet3d_Points/epoch_' + str(epochs) + '.pth.tar'
    #model.load_state_dict(torch.load(my_model))
    #model.eval()




ND_dir = glob('/media/mmlab/data/sesh/Data_ICH/Sesh_Valid/8/101ct1__resampled.nii.gz')
seg_dir = glob('/media/mmlab/data/sesh/Data_ICH/Sesh_Valid/8/101ct1_seg_resampled.nii.gz')
path = ND_dir
path_seg = seg_dir
print(path)
print(path_seg)


_img1 = nib.load(path[0])
_img2 = nib.load(path_seg[0])
_img = _img1.get_data()
_imgg = _img2.get_data()
_img = _img.transpose(2,0,1)
_img = _img/255
_imgg = _imgg.transpose(2,0,1)
#_imgg = _imgg/255
#print(np.min(_img))

_img = np.expand_dims(_img, axis=0)
_img = np.expand_dims(_img, axis=0)

_imgg = np.expand_dims(_imgg, axis=0)
_imgg = np.expand_dims(_imgg, axis=0)
# _img2 = np.expand_dims(_img2, axis=0)
# _img2 = np.expand_dims(_img2, axis=0)

_img[_img>0] = 1 #input
_img[_imgg>0] = 2 #imgg is output

x = _img.shape[2]*1//2
y = _img.shape[3]*1//2
z = _img.shape[4]*1//2

a = _img.shape[2]
b = _img.shape[3]
c = _img.shape[4]
for i in range(_img.shape[2]):
    for j in range(_img.shape[3]):
        for k in range(_img.shape[4]):
            if(_img[:,:,i,j,k]==2):
                if(i<x and j<y and k<z):
                    init_x = 0
                    fin_x = x 
                    init_y = 0
                    fin_y = y
                    init_z = 0
                    fin_z = z
                    break
                elif(i>x and j>y and k>z):
                    init_x = x
                    fin_x =  a
                    init_y = y
                    fin_y = b
                    init_z = z
                    fin_z = c
                    break
                elif(i<x and j<y and k>z):
                    init_x = 0
                    fin_x =  x
                    init_y = 0
                    fin_y = y
                    init_z = 0
                    fin_z = z
                    break
                elif(i<x and j>y and k<z):
                    init_x = 0
                    fin_x =  x
                    init_y = y
                    fin_y = b
                    init_z = 0
                    fin_z = z
                    break
                elif(i>x and j<y and k<z):
                    init_x = x
                    fin_x =  a
                    init_y = 0
                    fin_y = y
                    init_z = 0
                    fin_z = z
                    break
                elif(i>x and j>y and k<z):
                    init_x = x
                    fin_x =  a
                    init_y = y
                    fin_y = b
                    init_z = 0
                    fin_z = z
                    break
                elif(i>x and j<y and k>z):
                    init_x = x
                    fin_x =  a
                    init_y = 0
                    fin_y = y
                    init_z = z
                    fin_z = c
                    break
                elif(i<x and j>y and k>z):
                    init_x = 0
                    fin_x =  x
                    init_y = y
                    fin_y = b
                    init_z = z
                    fin_z = c
                    break

'''
x = np.zeros(3)
y = np.zeros(3)
z = np.zeros(3)
x[0] = _img.shape[2]*1//4
y[0] = _img.shape[3]*1//4
z[0] = _img.shape[4]*1//4

x[1] = _img.shape[2]*2//4
y[1] = _img.shape[3]*2//4
z[1] = _img.shape[4]*2//4

x[2] = _img.shape[2]*3//4
y[2] = _img.shape[3]*3//4
z[2] = _img.shape[4]*3//4
'''


#print(_img.shape[3]//4)
unique, counts = (np.unique(_img,return_counts = True))
#print(unique, counts)
_img[:,:,init_x:fin_x,init_y:fin_y,init_z:fin_z] = 0  #this sets segemneted sliced part to zero (Vanishes basically)
unique, counts = (np.unique(_img,return_counts = True))
#print(unique, counts)
#_img[0:_img.shape(0)/4,0: ]
real_a = torch.from_numpy(_img)
real_a = real_a.squeeze_(1).cpu().numpy()
_img = np.squeeze(real_a, axis=0).transpose(1,2,0)
#print(unique,counts)
segmented_img = nib.Nifti1Image(_img, _img1.affine, _img1.header)
nib.save(segmented_img,'sample101ct1.nii.gz')    
        
