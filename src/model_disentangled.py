# -*- coding: utf-8 -*-
"""
SP03 B — physics-disentangled (dual-branch) restorer.

Noise-first decomposition (cf. CDASRN noise-first; SP02 sigma_n estimated before nc_ratio):
  denoise branch : [degraded, blur-prompt, noise-prompt] -> noise-removed image (target = blur-only, noiseless)
  optics branch  : [denoised, blur-prompt]               -> restored image       (target = clean)
The synthetic degradation supplies the intermediate NOISELESS target (blur-only), giving explicit
physical supervision of the noise/optics separation — more than two stacked NAFNets.
"""
import torch
import torch.nn as nn
from model import NAFNetUNet


class DisentangledRestorer(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.denoise = NAFNetUNet(in_ch=3, out_ch=1, width=width)   # noise channel
        self.optics = NAFNetUNet(in_ch=2, out_ch=1, width=width)    # optical channel (deblur)

    def forward(self, x):
        """x: [B,3,H,W] = [degraded, blur-prompt, noise-prompt]. Returns (restored, denoised)."""
        den = self.denoise(x)                                   # ~ noiseless blurred
        deb_in = torch.cat([den, x[:, 1:2]], dim=1)             # [denoised, blur-prompt]
        out = self.optics(deb_in)                               # restored
        return out, den


if __name__ == "__main__":
    m = DisentangledRestorer(width=24)
    n = sum(p.numel() for p in m.parameters())
    o, d = m(torch.randn(2, 3, 256, 256))
    print(f"DisentangledRestorer params {n/1e6:.2f}M  out {tuple(o.shape)} den {tuple(d.shape)}")
