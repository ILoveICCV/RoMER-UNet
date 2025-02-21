

import pywt
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from typing import Sequence, Tuple, Union, List


def _outer(v1, v2):
    '''
    transform the demension
    '''
    v1_flat = torch.reshape(v1, [-1])
    v2_flat = torch.reshape(v2, [-1])
    v1_mul = torch.unsqueeze(v1_flat, dim=-1)
    v2_mul = torch.unsqueeze(v2_flat, dim=0)
    return v1_mul * v2_mul

#construct filter
def construct_filt(low, high):
    ll = _outer(low, low)
    lh = _outer(high, low)
    hl = _outer(low, high)
    hh = _outer(high, high)
    filt = torch.stack([ll, lh, hl, hh], 0)
    return filt

#dwt factors
def get_filter_tensors(wavelet,device: Union[torch.device, str] = 'cpu',dtype = torch.float32,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    def _create_vector(filter: Sequence[float]) -> torch.Tensor:
        if isinstance(filter, torch.Tensor):
            return filter.flip(-1).unsqueeze(0)
        else:
            return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)

    lo, hi, _, _ = wavelet.filter_bank
    lo_tensor = _create_vector(lo)
    hi_tensor = _create_vector(hi)
    return lo_tensor, hi_tensor


def _get_pad(data_len, filt_len):
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2
    if data_len % 2 != 0:
        padr += 1
    return padr, padl


def fwt_pad2(data, wavelet, mode):
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))
    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad

class DWT(nn.Module):
    def __init__(self,in_channels=3, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet)
        #construct filters
        lo, hi = get_filter_tensors(self.wavelet)
        #learnable DWT factors
        self.lo = nn.Parameter(lo, requires_grad=True)
        self.hi = nn.Parameter(hi, requires_grad=True)
        self.sample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.level = level
        self.mode = mode
        self.conv=nn.Conv2d(in_channels,in_channels,1)
        self.sig=nn.Sigmoid()
        self.fus=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        l_component = x
        #construct filters
        dwt_kernel = construct_filt(low=self.lo, high=self.hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
        l_component=l_component.to(x.device)
        h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
        res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
        component, lh_component, hl_component, hh_component = res.split(1, 2)
        component, lh_component, hl_component, hh_component = component.squeeze(2),lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)
        component=self.sample(component)
    
        #high-frequency information
        h_component=lh_component + hl_component + hh_component
        h_component=self.conv(h_component)
        h_component=self.sample(h_component)
        x_1=self.sig(h_component)*x

        #low-frequency information
        x_2=self.sig(x-l_component)*x
        
        x=x_1+x_2
        x=self.fus(x)
        return x
