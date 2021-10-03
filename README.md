# 3D_ICH_SEGMENTATION_PREDICTION
This repository contains the official implementation of the paper "Identifying Risk Factors of Intracerebral Hemorrhage stability using Explainable Attention Model"

A simple 3D_sSe implementation
```
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

The model architecture is adopted from this [repository](https://github.com/wolny/pytorch-3dunet)
