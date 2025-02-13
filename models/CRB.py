import torch

import torch.nn as nn


import torchvision

class DirectionOffsets(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        
        self.offset1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,(1,15),1,(0,7),groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.offset2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,(15,1),padding=(7,0),groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.offset3=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.mlp=nn.Sequential(
            nn.Linear(in_channels,in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels,in_channels),
        )
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels,2 * kernel_size * kernel_size,1),
            nn.BatchNorm2d(2 * kernel_size * kernel_size)
        )
        
    def forward(self, x):
        B,C,H,W=x.shape
        offsets1=self.offset1(x)
        offsets2=self.offset2(x)
        offsets3=self.offset3(x)
        offsets=offsets1+offsets2+offsets3
        offsets=offsets.permute(0,2,3,1).contiguous().view(B,H*W,C)
        offsets=self.mlp(offsets)
        offsets=offsets.permute(0,2,1).contiguous().view(B,C,H,W)
        offsets=self.balance(offsets)
        return offsets

class CRB(nn.Module):
    def __init__(self, in_channels, kernel_size=3,padding=1,dilation=1):
        super().__init__()
        self.offset=DirectionOffsets(in_channels)
        self.deform=torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=in_channels,
                                                        dilation=dilation,
                                                        bias=False)
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        offsets = self.offset(x)
        out = self.deform(x, offsets)
        out = self.balance(out)*x
        return out
