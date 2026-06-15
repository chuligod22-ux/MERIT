# -*- coding: utf-8 -*-
"""
SP03 B (retry) — parallel + fusion physics-disentangled restorer (avoids cascade over-smoothing).

Both branches see the SAME degraded input IN PARALLEL:
  noise branch  : -> denoised image (aux target = noiseless blur-only)   [removes noise, keeps blur]
  optics branch : -> optical/high-freq estimate (operates on raw input, KEEPS high frequency)
  fusion        : [denoised, optics, degraded, blur-prompt] -> final = denoised + learned optical detail
Unlike the cascade (denoise->deblur), the optics branch never sees the smoothed denoised output,
so high-frequency content is not destroyed before deblurring. Param budget < single (4.95M).
"""
import torch
import torch.nn as nn
from model import NAFNetUNet, NAFBlock


class ParallelFusionRestorer(nn.Module):
    def __init__(self, width=20, fuse_width=24):
        super().__init__()
        self.noise = NAFNetUNet(in_ch=3, out_ch=1, width=width)
        self.optics = NAFNetUNet(in_ch=3, out_ch=1, width=width)
        self.fuse = nn.Sequential(
            nn.Conv2d(4, fuse_width, 3, padding=1),
            NAFBlock(fuse_width), NAFBlock(fuse_width),
            nn.Conv2d(fuse_width, 1, 3, padding=1))

    def forward(self, x):
        den = self.noise(x)                                   # denoised (~noiseless blurred)
        opt = self.optics(x)                                  # optical high-freq estimate
        f = torch.cat([den, opt, x[:, 0:1], x[:, 1:2]], dim=1)
        fin = torch.clamp(den + self.fuse(f), 0.0, 1.0)       # denoised + learned optical detail
        return fin, den


if __name__ == "__main__":
    m = ParallelFusionRestorer(width=20)
    n = sum(p.numel() for p in m.parameters())
    o, d = m(torch.randn(2, 3, 256, 256))
    print(f"ParallelFusionRestorer params {n/1e6:.2f}M  out {tuple(o.shape)} den {tuple(d.shape)}")
