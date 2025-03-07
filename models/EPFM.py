import torch
from torch import nn
from models.EDB import EDB
from models.HPB import HPB

class EPFM(nn.Module):
    def __init__(self, in_channels,out_channels,sample,up=True,kernel_list=[3,9]):
        super().__init__()
        
        self.edb=EDB(in_channels,kernel_list=kernel_list)
        self.hpb=HPB(in_channels)
        self.mlp=nn.Sequential(
                nn.BatchNorm2d(in_channels*2),
                nn.Conv2d(in_channels*2,out_channels,1),
                nn.GELU(),
                nn.Conv2d(out_channels,out_channels,1),
                nn.BatchNorm2d(out_channels)
            )
        if sample:
            if up:
                self.sample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.sample=nn.MaxPool2d(2,stride=2)
        else:
            self.sample=None

    def forward(self,x):
        x_edb=self.edb(x)
        x_hpb=self.hpb(x)
        x_cat=torch.cat([x_edb,x_hpb],dim=1)
        x=self.mlp(x_cat)
        if self.sample!=None:
            x=self.sample(x)
        return x
