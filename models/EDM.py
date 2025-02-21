import torch

import torch.nn as nn
from models.DWT import DWT
from models.CRB import CRB


class MSFA_Block(nn.Module):
    def __init__(self,in_channels,kernel,sample1=None,sample2=None):
        super().__init__()
        self.sample1=sample1
        self.sample2=sample2
        self.extract=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel,padding=kernel//2,groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self,x):
        if self.sample1!=None:
            x=self.sample1(x)
        x=self.extract(x)
        if self.sample2!=None:
            x=self.sample2(x)
        return x
    
    
class MSFA(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        '''
        extract multi-scale features from different receptive fields (3×3 or 9×9) and different image sizes (original sizes, upsample, or maxpool).
        '''
        self.c=in_channels
        #different image sizes and receptive fields
        self.msfa1=MSFA_Block(in_channels,3)
        self.msfa2=MSFA_Block(in_channels,9)
        self.msfa3=MSFA_Block(in_channels,3,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.msfa4=MSFA_Block(in_channels,9,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.msfa5=MSFA_Block(in_channels,3,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.msfa6=MSFA_Block(in_channels,9,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.extract=nn.Sequential(
            nn.Conv2d(6*in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1),
        )
        
        
    def forward(self,x):
        x1=self.msfa1(x)
        x2=self.msfa2(x)
        x3=self.msfa3(x)
        x4=self.msfa4(x)
        x5=self.msfa5(x)
        x6=self.msfa6(x)
        out=torch.cat([x1,x2,x3,x4,x5,x6],dim=1)
        out=self.extract(out)
        return out



class EDM(nn.Module):
    def __init__(self,in_channels=3,args=None):
        super().__init__()
        '''
        MSFA: multi-scale features
        DWT: multi-frequency features
        CRB: learning deform
        '''
        self.msfa=MSFA(in_channels,args=args)
        self.dwt=DWT(in_channels)
        self.crb=CRB(in_channels)


    def forward(self,x):
        x=self.msfa(x)
        x=self.dwt(x)
        x=self.crb(x)
        return x


