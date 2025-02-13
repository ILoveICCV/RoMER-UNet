

import pywt

import torch

import torch.nn as nn

def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

def construct_2d_filt(lo, hi) -> torch.Tensor:
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    return filt


from einops import rearrange, repeat
import torch.nn.functional as F

from typing import Sequence, Tuple, Union, List


def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int):
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl

def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad

class DWT(nn.Module):
    def __init__(self,in_channels=3, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet)
        
        dec_lo, dec_hi, _, _ = get_filter_tensors(
            self.wavelet, flip=True
        )
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
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
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
        l_component=l_component.to(x.device)
        h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
        res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
        component, lh_component, hl_component, hh_component = res.split(1, 2)
        component, lh_component, hl_component, hh_component = component.squeeze(2),lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)
        component=self.sample(component)

        h_component=lh_component + hl_component + hh_component
        h_component=self.conv(h_component)
        h_component=self.sample(h_component)
        x_1=self.sig(h_component)*x

        x_2=self.sig(x-l_component)*x
        x=x_1+x_2
        x=self.fus(x)
        return x
