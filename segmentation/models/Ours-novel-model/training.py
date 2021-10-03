#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
import SimpleITK as sitk
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
import torch
import torchvision.transforms as standard_transforms
#from torchvision.models import resnet18
from UNet_3D import UNet3D
import nibabel as nib


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)
ckpt_path = 'ckpt'
exp_name = 's(c)SE'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(os.path.join(ckpt_path, exp_name)):
    os.makedirs(os.path.join(ckpt_path, exp_name))
args = {
    'num_class': 2,
    'num_gpus': 1,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 1 ,
    'lr': 0.0001,
    'lr_decay': 0.9,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
    'crop_size1': 138,

}


def mat2img(slices):
    tmin = np.amin(slices)
    tmax = np.amax(slices)
    diff = tmax -tmin
    if (diff == 0):
        return slices
    else:
        return np.uint8(255 * (slices - tmin) / (diff))

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
        hflip = random.random() < 0.5
        
        if hflip:
            img = img[:, ::-1, :, :]
            target = target[::-1,:, :]

        img = torch.from_numpy(np.array(img)).float()
        target = torch.from_numpy(np.array(target)).long()

        return img, target


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


if __name__ == '__main__':
    input_transform = standard_transforms.Compose([standard_transforms.ToTensor()])

    img_dir = '/media/mmlab/data/sesh/Data_ICH/Sesh_Segmentation/**'
    print(img_dir)
    dataset = HEMDataset(img_dir=img_dir)
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,drop_last=True)
    in_channels = 1
    model = UNet3D(in_channels=1, out_channels=2, final_sigmoid=True) 
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.0001)

    criterion = CrossEntropyLoss2d(size_average=True).cuda()
    model.train()
    epoch_iters = dataset.__len__() / args['batch_size']
    max_epoch = 100
    print(exp_name)
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs.view(args['batch_size'], args['num_class'], args['crop_size1'], -1)
            labels = labels.view(args['batch_size'], args['crop_size1'], -1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 20 == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
                    epoch, batch_idx + 1, epoch_iters, loss.item(),
                    optimizer.param_groups[0]['lr']))

            cur_iter = batch_idx + epoch * len(train_loader)
            max_iter = len(train_loader) * max_epoch

        snapshot_name = 'epoch_' + str(epoch)
        torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth.tar'))
