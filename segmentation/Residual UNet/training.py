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
from UNet_Residual import ResidualUNet3D
import nibabel as nib
#Customs
#from ptsemseg.models import get_model

#from linknet import LinkNet
#from models import LinkNet34


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#torch.cuda.set_device(1)
ckpt_path = 'ckpt'
exp_name = 'Residual_Points'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(os.path.join(ckpt_path, exp_name)):
    os.makedirs(os.path.join(ckpt_path, exp_name))
args = {
    'num_class': 2,
    #'ignore_label': 255,
    'num_gpus': 1,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 1 ,
    'lr': 0.0001,
    'lr_decay': 0.9,
    #'dice': 0,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
    #'crop_size': 186,
    'crop_size1': 138,

}
#IMG_MEAN = np.array((12.309, 16.46, 10.607, 12.907), dtype=np.float32)


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
        #self.crop_size = args['crop_size']
        #self.crop_size1 = args['crop_size1']

    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        #modalities1 = np.zeros((174, 241, 241))
        #modalities2 = np.zeros((174, 241, 241))
        #print("LOLOLOL, length to print")
        #print(len(self.img_anno_pairs))
        _img = glob(self.img_anno_pairs[index] + '/*__resampled.nii.gz')
        _gt = glob(self.img_anno_pairs[index] + '/*seg_resampled.nii.gz')
        #print(_gt)
        _img = nib.load(_img[0]).get_data()
        _gt = nib.load(_gt[0]).get_data()
        
        _img = _img.transpose(2,0,1)
        _gt = _gt.transpose(2,0,1)


        img = np.zeros((138,186,186))
        target = np.zeros((138,186,186))

        
        a0 = _img.shape[0]  #138
        a1 = _img.shape[2]  #186
        
        a0 = (a0 - 138)//2
        a1 = (a1 - 186)//2
        

        #crop_x = np.random.randint(10, a1 - self.crop_size)
        #crop_y = np.random.randint(10, a1 - self.crop_size)
        #crop_z = np.random.randint(5,  a0 - self.crop_size1)
        
        img = _img[a0:a0+138, a1:a1+186, a1:a1+186]
        target = _gt[a0:a0+138, a1:a1+186, a1:a1+186]
        
        
        # _img1 = _img[crop_z:crop_z + self.crop_size1,crop_x:crop_x + self.crop_size, crop_y:crop_y + self.crop_size]
        # gt1 = gt[crop_z:crop_z + self.crop_size1,crop_x:crop_x + self.crop_size, crop_y:crop_y + self.crop_size]

        
        #target[target==4] = 3
        #print(_img.shape,_target.shape)

        img = img/255

        #print(np.unique(img))
        #print(np.unique(target))

        img = np.expand_dims(img, axis=0)
        #target = np.expand_dims(target, axis=0)

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
    #mean_std = ([0.154, 0.150, 0.019], [0.0118, 0.0218, 0.0022])
    input_transform = standard_transforms.Compose([
        #standard_transforms.RandomHorizontalFlip(),
        standard_transforms.ToTensor()
        #,standard_transforms.Normalize(*mean_std)
    ])

    #img_dir = '/home/mobarak/uu/ICH/Segmentation_ICH3d/Segmentation/**'
    img_dir = '/media/mmlab/data/sesh/Data_ICH/Sesh_Segmentation/**'

    dataset = HEMDataset(img_dir=img_dir)
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,drop_last=True)
    #model = get_model("linknet", n_classes=2).cuda()
    #model = LinkNet34(num_classes=args['num_class'], pretrained=True)
    in_channels = 1
    #n_classes = 2
    #base_n_filter = 16
    model = ResidualUNet3D(in_channels=1, out_channels=2, final_sigmoid=True) 
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.0001)

    criterion = CrossEntropyLoss2d(size_average=True).cuda()
    model.train()
    epoch_iters = dataset.__len__() / args['batch_size']
    #print(dataset.__len__())
    max_epoch = 100
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            #print(inputs.size(), labels.size())
            # print(np.unique(inputs),np.unique(labels),np.unique(labels_aux4),np.unique(labels_aux24))
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            
            #labels_aux = Variable(labels_aux).cuda()
            #print(inputs.size(),labels.size(), labels_aux.size())
            
            #print('inputs: ',inputs.size())
            #print('labels: ', labels.size())
            
            optimizer.zero_grad()
            outputs = model(inputs)
            #outputs = Variable(outputs).cuda()
            #labels = labels.permute(1, 2, 3, 0).contiguous().view(-1, 1)

            #print(outputs.size())
            #print(labels.size())

            #print(labels.size())

            outputs = outputs.view(args['batch_size'], args['num_class'], args['crop_size1'], -1)
            labels = labels.view(args['batch_size'], args['crop_size1'], -1)

            #print(outputs.size())
            #print(labels.size())

            #print('inputs: ',inputs.size())
            #print('outputs: ', outputs.size())
            #print('labels: ', labels.size())

            #print(labels.shape, outputs.shape)
            #loss = DiceLoss(outputs, labels)
            #print(loss)
            #print(outputs.unique(), labels.unique())
            
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 20 == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
                    epoch, batch_idx + 1, epoch_iters, loss.item(),
                    optimizer.param_groups[0]['lr']))

            cur_iter = batch_idx + epoch * len(train_loader)
            max_iter = len(train_loader) * max_epoch
            # adjust_learning_rate(optimizer, cur_iter, max_iter)
            # poly_lr_scheduler(optimizer, init_lr=args['lr'], iter=cur_iter, max_iter=max_iter)
        snapshot_name = 'epoch_' + str(epoch)
        torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth.tar'))


#etG = torch.nn.parallel.DataParallel(netG, device_ids=gpu_ids)
#netG = netG.cuda()
