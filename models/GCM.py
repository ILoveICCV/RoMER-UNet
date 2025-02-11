import torch.nn as nn
import torch

class GCM(nn.Module):
    def __init__(self,input_channels=3, gamma=1.5,num_head=1):
        super().__init__()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.sample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sig=nn.Sigmoid()
        self.gamma=gamma
        self.num_head=num_head
        self.conv=nn.Conv2d(input_channels,out_channels=input_channels,kernel_size=1,groups=num_head)

        self.scale=(input_channels//num_head)**-0.5
        self.norm=nn.LayerNorm(input_channels)
        self.Q_proj=nn.Linear(input_channels,input_channels)
        self.K_proj=nn.Linear(input_channels,input_channels)
        self.V_proj=nn.Linear(input_channels,input_channels)
        self.soft=nn.Softmax(-1)

        self.mlp=nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels,input_channels,1),
            nn.GELU(),
            nn.Conv2d(input_channels,input_channels,1),
        )

    def forward(self,x:torch.Tensor):
        x=self.pool(x)
        x_m=self.sig(x)**self.gamma
        x_masks=self.conv(x_m) # B C H W
        B,C,H,W=x.shape
        N=H*W
        x_norm=x.view(B,C,N).permute(0,2,1).contiguous() # B N C
        x_norm=self.norm(x_norm)
        x_Q=self.Q_proj(x_norm)
        x_K=self.K_proj(x_norm)
        x_V=self.V_proj(x_norm)
        x_attn=x_Q@x_K.transpose(1,2) # B N N
        x_attn=self.soft(x_attn*self.scale)@x_V # B N C
        x_attn=x_attn.permute(0,2,1).contiguous().view(B,C,H,W)
        # x_masks=x_masks.view(B,1,H,W)
        x_res=x_attn*x_masks
        x_res=x_res+x_norm.permute(0,2,1).view(B,C,H,W)
        x_res=self.mlp(x_res)+x_res
        x_res=self.sample(x_res)
        return x_res
   


    def forward1(self,x:torch.Tensor):
        x=self.pool(x)
        x_m=self.sig(x)**self.gamma
        x_masks=self.conv(x_m) # B C H W
        B,C,H,W=x.shape
        N=H*W
        x_norm=x.view(B,C,N).permute(0,2,1).contiguous() # B N C
        x_norm=self.norm(x_norm)
        x_Q=self.Q_proj(x_norm).view(B,self.num_head,N,C//self.num_head) # B H N D
        x_K:torch.Tensor=self.K_proj(x_norm).view(B,self.num_head,N,C//self.num_head) # B H N D
        x_V=self.V_proj(x_norm).view(B,self.num_head,N,C//self.num_head) # B H N D
        x_attn=x_Q@x_K.transpose(2,3) # B H N N
        x_attn=self.soft(x_attn**self.scale)@x_V # B H N D
        x_attn=x_attn.permute(0,1,3,2).contiguous().view(B,self.num_head,C//self.num_head,H,W)
        x_masks=x_masks.view(B,self.num_head,1,H,W)
        x_res=x_attn*x_masks
        x_res=x_res.view(B,C,H,W)+x_norm.permute(0,2,1).view(B,C,H,W)
        x_res=self.mlp(x_res)+x_res
        x_res=self.sample(x_res)
        return x_res