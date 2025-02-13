import torch.nn as nn



from models.EPFM import EPFM



class Net(nn.Module):
    def __init__(self,input_channels=3, out_channels:list=[12,24,36,48]):
        super().__init__()
        self.num_layer=len(out_channels)
        out_channels.append(out_channels[-1])
        downsample=[True for i in range(self.num_layer-1)]
        downsample.append(True)
        upsample=[True for i in range(self.num_layer-1)]
        upsample=[True]+upsample        
        self.patch_conv=nn.Sequential(
            nn.Conv2d(input_channels,out_channels[0],3,padding=1),
            nn.BatchNorm2d(out_channels[0])
        )
        self.encoder=nn.ModuleList()
        self.decoder=nn.ModuleList()
        for i in range(self.num_layer):
            self.encoder.append(
                EPFM(out_channels[i],out_channels[i+1],sample=downsample[i],up=False)
            )
            self.decoder.append(
                EPFM(out_channels[i+1],out_channels[i],sample=upsample[i],up=True)
            )
        self.ph=PH(out_channels)
        
        

    def forward(self,x):
        x=self.patch_conv(x)
        x_encoder=[]
        x_decoder=[]
        for i in range(self.num_layer):
            x=self.encoder[i](x)
            x_encoder.append(x)
        x=x_encoder[-1]
        for j in range(self.num_layer):
            x=self.decoder[-j-1](x)
            x_decoder.append(x)
            if j<3:
                x=x+x_encoder[-j-2]
        x_pre=self.ph(x_decoder)
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
        self.final=nn.ModuleList()
        self.ph1=PH_Block(in_channels[0],scale_factor[0])
        self.ph2=PH_Block(in_channels[1],scale_factor[1])
        self.ph3=PH_Block(in_channels[2],scale_factor[2])
        self.ph4=PH_Block(in_channels[3],scale_factor[3])
        
    def forward(self,x):
        x4,x3,x2,x1=x
        x_list=[]
        x1=self.ph1(x1)
        x_list.append(x1.sigmoid())
        x2=self.ph2(x2)
        x_list.append(x2.sigmoid())
        x3=self.ph3(x3)
        x_list.append(x3.sigmoid())
        x4=self.ph4(x4)
        x_list.append(x4.sigmoid())
        return x_list
