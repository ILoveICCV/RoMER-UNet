import torch.nn as nn
import torch


class GCM(nn.Module):
    def __init__(self,input_channels=3,args=None):
        super().__init__()
        self.args=args
        #PGC
        self.linear_o=nn.Linear(input_channels,input_channels)
        self.linear_p=nn.Linear(input_channels,input_channels)
        self.scale=input_channels**-0.5
        self.norm=nn.LayerNorm(input_channels)
        self.soft=nn.Softmax(-1)
        self.dsc=nn.Sequential(
                nn.Conv2d(input_channels,input_channels,3,padding=1,groups=input_channels),
                nn.GELU(),
                nn.Conv2d(input_channels,input_channels,1),
                nn.BatchNorm2d(input_channels)
            )
        self.linear_p=nn.Linear(input_channels,input_channels)

        #LPG
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.Q_proj=nn.Linear(input_channels,input_channels)
        self.K_proj=nn.Linear(input_channels,input_channels)
        self.liner_l=nn.Linear(input_channels,input_channels)
        self.sig=nn.Sigmoid()

        #HPG
        self.sc_h=nn.Sequential(
                nn.Conv2d(input_channels,input_channels,kernel_size=(5,1),padding=(2,0),groups=input_channels),
                nn.GELU(),
                nn.Conv2d(input_channels,input_channels,1),
                nn.BatchNorm2d(input_channels)
            )
        self.sc_v=nn.Sequential(
                nn.Conv2d(input_channels,input_channels,kernel_size=(1,5),padding=(0,2),groups=input_channels),
                nn.GELU(),
                nn.Conv2d(input_channels,input_channels,1),
                nn.BatchNorm2d(input_channels)
            )
        self.conv_h=nn.Conv2d(input_channels,input_channels,1)
        
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        
    def forward_lpg(self,x):
        x_gap=self.gap(x)*x
        B,C,H,W=x_gap.shape
        N=H*W
        x_gap=x_gap.view(B,C,N).permute(0,2,1).contiguous() # B N C
        x_Q=self.Q_proj(x_gap)
        x_K=self.K_proj(x_gap)
        x_V=self.sig(x_Q)*x_K+x_gap
        prompt_l=self.liner_l(x_V)
        prompt_l=prompt_l.permute(0,2,1).contiguous().view(B,C,H,W)
        return prompt_l

    def forward_hpg(self,x):
        x_h=self.sc_h(x)
        x_v=self.sc_v(x)
        prompt_h=self.conv_h(x_h+x_v)
        return prompt_h

    def forward_pgc(self,prompt_l,prompt_h,x_ori):
        B,C,H,W=x_ori.shape
        N=H*W
        x_V=self.linear_o(x_ori.view(B,C,N).permute(0,2,1).contiguous()) # B N C
        x_K=prompt_l.view(B,C,N).permute(0,2,1).contiguous() 
        x_Q=prompt_h.view(B,C,N).permute(0,2,1).contiguous() 
        x_attn=x_Q@x_K.transpose(1,2) # B N N
        prompt=self.soft(x_attn*self.scale)@x_V# B N C
        prompt=self.linear_p(prompt)+x_V
        p_norm=self.norm(prompt)
        p_norm=p_norm.permute(0,2,1).contiguous().view(B,C,H,W)
        p_norm=self.up(p_norm)
        out=self.dsc(p_norm)+p_norm
        return out

    def forward(self,x:torch.Tensor):
        x=self.pool(x)
        prompt_l=self.forward_lpg(x) #generate low-frequency prompts
        prompt_h=self.forward_hpg(x) #generate high-frequency prompts
        out=self.forward_pgc(prompt_l,prompt_h,x) #embed prompts into cross self-attention
        return out