import torch.nn as nn



from models.EPFM import EPFM



class Net(nn.Module):
    def __init__(self,input_channels=3, out_channels:list=None,args=None):
        super().__init__()
        #encoding
        self.en1=EPFM(out_channels[0],out_channels[1],sample=True,up=False,args=args)
        self.en2=EPFM(out_channels[1],out_channels[2],sample=True,up=False,args=args)
        self.en3=EPFM(out_channels[2],out_channels[3],sample=True,up=False,args=args)
        self.en4=EPFM(out_channels[3],out_channels[4],sample=True,up=False,args=args)

        #decoding
        self.de1=EPFM(out_channels[1],out_channels[0],sample=True,up=True,args=args)
        self.de2=EPFM(out_channels[2],out_channels[1],sample=True,up=True,args=args)
        self.de3=EPFM(out_channels[3],out_channels[2],sample=True,up=True,args=args)
        self.de4=EPFM(out_channels[4],out_channels[3],sample=True,up=True,args=args)

        #patch
        self.patch_conv=nn.Sequential(
            nn.Conv2d(input_channels,out_channels[0],3,padding=1),
            nn.BatchNorm2d(out_channels[0])
        )

        #prediction
        self.ph=PH(out_channels)
        
    def forward(self,x):
        #patch
        x=self.patch_conv(x)

        #encoding
        e1=self.en1(x)
        e2=self.en2(e1)
        e3=self.en3(e2)
        e4=self.en4(e3)

        #decoding
        d4=self.de4(e4)
        d3=self.de3(d4+e3)
        d2=self.de2(d3+e2)
        d1=self.de1(d2+e1)
        
        #prediction
        x_pre=self.ph([d4,d3,d2,d1])
        return x_pre



class PH_Block(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Conv2d(in_channels,1,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        x=self.sig(x)
        return x

class PH(nn.Module):
    def __init__(self,in_channels=[12,24,36,48],scale_factor=[1,2,4,8]):
        super().__init__()
        self.ph1=PH_Block(in_channels[0],scale_factor[0])
        self.ph2=PH_Block(in_channels[1],scale_factor[1])
        self.ph3=PH_Block(in_channels[2],scale_factor[2])
        self.ph4=PH_Block(in_channels[3],scale_factor[3])
        
    def forward(self,x):
        x4,x3,x2,x1=x
        x1=self.ph1(x1)
        x2=self.ph2(x2)
        x3=self.ph3(x3)
        x4=self.ph4(x4)
        return [x1,x2,x3,x4]
