import torch
from torch import nn
from models.EDM import EDM
from models.GCM import GCM

class EPFM(nn.Module):
    def __init__(self, in_channels,out_channels,sample,up=True,args=None):
        super().__init__()
        self.args=args
        
        self.edm=EDM(in_channels,args=args)
        self.gcm=GCM(in_channels,args=args)
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
        x_edm=self.edm(x)
        x_gcm=self.gcm(x)
        x_cat=torch.cat([x_edm,x_gcm],dim=1)
        x=self.mlp(x_cat)
        if self.sample!=None:
            x=self.sample(x)
        return x