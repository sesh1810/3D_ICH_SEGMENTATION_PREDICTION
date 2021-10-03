import numpy as np
import math
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import time
#from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import directed_hausdorff
from numpy.core.umath_tests import inner1d
from skimage.io import imread, imsave
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as standard_transforms
from torchsummary import summary
#Customs
from UNet_3D import UNet3D
import cv2

#System
import SimpleITK as sitk
#Torch
import torch.nn.functional as F
from torch.autograd import Function
import torch
import torchvision.transforms as standard_transforms
#from torchvision.models import resnet18
import nibabel as nib
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

args = {
    'snapshot': '',
    'num_class': 2,
    'batch_size':1,
    'num_gpus':1,
    'crop_size1': 138,
    'ckpt_path': 'ckpt/sSE',
}

IMG_MEAN = np.array((0.4789, 0.3020, 0.3410), dtype=np.float32)

def HausdorffDist(A,B):
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
    return (dH)

def dice(pred, label):
    dice_val = np.float(np.sum(pred[label == 1] == 1)) * 2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)));
    return dice_val

def specificity(TP, TN, FP, FN):
    return TN / (FP + TN)


def sensitivity(TP, TN, FP, FN):
    return TP / (TP + FN)

def spec_sens(pred, gt):
    # pred[pred>0] = 1
    # gt[gt>0] = 1
    A = np.logical_and(pred, gt)
    TP = float(A[A > 0].shape[0])
    TN = float(A[A == 0].shape[0])
    B = img_pred - labels
    FP = float(B[B > 0].shape[0])
    FN = float(B[B < 0].shape[0])
    specificity = TN / (FP + TN)
    sensitivity = TP / (TP + FN)
    return specificity, sensitivity

class HEMDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_anno_pairs = glob(img_dir)

    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):

        _img = glob(self.img_anno_pairs[index] + '/*__resampled.nii.gz')
        _gt = glob(self.img_anno_pairs[index] + '/*seg_resampled.nii.gz')

        _img = nib.load(_img[0]).get_data()
        _gt = nib.load(_gt[0]).get_data()
        _img = _img.transpose(2,0,1)
        _gt = _gt.transpose(2,0,1)

        a0 = _img.shape[0]  #138
        a1 = _img.shape[2]  #186
        a0 = (a0 - 138)//2
        a1 = (a1 - 186)//2
        
        img = _img[a0:a0+138, a1:a1+186, a1:a1+186]
        target = _gt[a0:a0+138, a1:a1+186, a1:a1+186]

        img = img/255
        img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(np.array(img)).float()
        target = torch.from_numpy(np.array(target)).long()
        

        return img, target


if __name__ == '__main__':
    img_dir = '/media/mmlab/data/sesh/Data_ICH/Sesh_Valid/**'
    dataset = HEMDataset(img_dir=img_dir)
    test_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=False, num_workers=1,drop_last=False)
    model = UNet3D(in_channels=1, out_channels=2, final_sigmoid=True)
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    Best_Dice = 0
    Best_epoch=0
    print(args['ckpt_path'])
    #for epochs in range(1,100):
    for epochs in range(0,1):
        args['snapshot'] = 'Paper_098ct2_sSE.nii.gz'
        model.load_state_dict(torch.load(args['snapshot']))
        model.eval()
        w, h = 0, args['num_class']
        mdice = [[0 for x in range(w)] for y in range(h)]
        mspecificity = [[0 for x in range(w)] for y in range(h)]
        msensitivity = [[0 for x in range(w)] for y in range(h)]
        mhausdorff = [[0 for x in range(w)] for y in range(h)]
        haus = []
        mytime = []
        dice_best = {}
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                inputs, labels = data
                inputs = Variable(inputs).cuda()
                t0 = time.time()
                #print(summary(model, (3, 1024, 1280)))
                outputs = model(inputs)
                #print(inputs.shape)
                #print(outputs.shape)
                #print(labels.shape)
                t1 = time.time()
                mytime.append((t1 - t0))
                outputs = outputs.view(args['batch_size'], args['num_class'], args['crop_size1'], -1)
                labels = labels.view(args['batch_size'], args['crop_size1'], -1)
                img_pred = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
                labels = np.array(labels)
                
                #labels[labels>0] = 1
                #img_pred[img_pred>0] = 1
                #req_img = mpath[0][33] + '_' + mpath[0][53:61]+'.png'
                #cv2.imwrite(pred_path+req_img, img_pred.transpose(1,2,0))
                for dice_idx in range(0, img_pred.shape[0]):
                    if(np.max(labels[dice_idx])==0):
                        continue
                    labs = np.unique(labels[dice_idx])
                    #print(labs)
                    #dataset_no = os.path.basename(os.path.dirname(os.path.dirname(mpath[dice_idx])))[-1:]
                    #cv2.imwrite(os.path.join(str(epochs), dataset_no+os.path.basename(mpath[dice_idx])), img_pred[dice_idx])
                    for seg_idx in range(1, len(labs)):
                        labels_temp = np.zeros(labels.shape[1:])
                        img_pred_temp = np.zeros(labels.shape[1:])
                        labels_temp[labels[dice_idx] == labs[seg_idx]] = 1
                        img_pred_temp[img_pred[dice_idx] == labs[seg_idx]] = 1
                        if (np.max(labels_temp) == 0):# or (np.max(img_pred_temp)==0):
                            continue

                        # d_idx = dataset_no + os.path.basename(mpath[dice_idx])
                        # dice_best[d_idx] = dice(img_pred_temp, labels_temp)
                        #print(sorted(dice_best.items(), key=lambda kv:(kv[1], kv[0]),reverse=True))

                        mdice[labs[seg_idx]].append(dice(img_pred_temp, labels_temp))
                        #print(dice(img_pred_temp, labels_temp))
                        mhausdorff[labs[seg_idx]].append(directed_hausdorff(img_pred_temp, labels_temp)[0])
                        spec, sens = spec_sens(img_pred_temp, labels_temp)
                        mspecificity[labs[seg_idx]].append(spec)
                        msensitivity[labs[seg_idx]].append(sens)

        #print(sorted(dice_best.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        avg_dice = []
        avg_hd = []
        avg_spec = []
        avg_sens = []
        for idx_eval in range(1, args['num_class']):
            avg_dice.append(np.mean(mdice[idx_eval]))
            avg_hd.append(np.mean(mhausdorff[idx_eval]))
            avg_spec.append(np.mean(mspecificity[idx_eval]))
            avg_sens.append(np.mean(msensitivity[idx_eval]))

        if np.mean(avg_dice) > Best_Dice:
            Best_Dice = np.mean(avg_dice)
            Best_epoch = epochs

        print(str(epochs) +' len:'+str(len(avg_dice))+ ' Mean Dice:' + str(np.mean(avg_dice)) +' Each:'+ str(avg_dice) +'   Best='+str(Best_epoch)+':'+str(Best_Dice))
'''
        print(' Mean Dice:', str(np.mean(avg_dice)), ' Each:', str(avg_dice),'\n',
               ' Mean Hausdorff:', str(np.mean(avg_hd)), ' Each:', str(avg_hd), '\n',
               ' Mean Specificity:', str(np.mean(avg_spec)), ' Each:', str(avg_spec), '\n',
               ' Mean Sensitivity:', str(np.mean(avg_sens)), ' Each:', str(avg_sens), '\n',
               'Avg Time(ms):', np.mean(mytime) * 1000, 'fps:', (1.0 / np.mean(mytime)))'''