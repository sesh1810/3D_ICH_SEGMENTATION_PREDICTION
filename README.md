# 3D_ICH_SEGMENTATION_PREDICTION
This repository contains the official implementation of the paper "[Identifying Risk Factors of Intracerebral Hemorrhage stability using Explainable Attention Model](https://link.springer.com/article/10.1007/s11517-021-02459-y)"

A simple 3D_sSe implementation
```python
class SSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return spa_se
```
Train command for 3D-sSE on the dataset 
```
CUDA_VISIBLE_DEVICES=0,1 python training.py --batch_size 2 --data_root /media/mmlab/data/sesh/Data_ICH/Sesh_Segmentation/ 
```
Validation command for 3D-sSE on the dataset
```
CUDA_VISIBLE_DEVICES=0,1 python tester.py --batch_size 2 --data_root /media/mmlab/data/sesh/Data_ICH/Sesh_Segmentation/
```
The model architecture is adopted from this [repository](https://github.com/wolny/pytorch-3dunet)

# Citation
If you use this code for your research, please cite our paper:
```bibtex
@article{rangaraj2021identifying,
  title={Identifying risk factors of intracerebral hemorrhage stability using explainable attention model},
  author={Rangaraj, Seshasayi and Islam, Mobarakol and Vs, Vibashan and Wijethilake, Navodini and Uppal, Utkarsh and See, Angela An Qi and Chan, Jasmine and James, Michael Lucas and King, Nicolas Kon Kam and Ren, Hongliang},
  journal={Medical \& biological engineering \& computing},
  pages={1--12},
  year={2021},
  publisher={Springer}
}
```
