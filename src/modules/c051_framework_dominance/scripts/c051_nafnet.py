# -*- coding: utf-8 -*-
"""
Compact NAFNet restorer (pure PyTorch, no custom CUDA — runs on Blackwell/torch2.11).

Input  : 3ch = [degraded grayscale, blur-prompt map, noise-prompt map]  (measured degradation prompt)
Output : 1ch = restored grayscale
A flat stack of NAFBlocks at full resolution; global residual adds the degraded channel back.
NAFBlock = Simplified-channel-attention + SimpleGate (Chen et al., ECCV'22 "Simple Baselines").
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.w[None, :, None, None] + self.b[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1)
        )
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv2(self.conv1(y))
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class NAFNetRestorer(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, width=32, n_blocks=8):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(n_blocks)])
        self.out = nn.Conv2d(width, out_ch, 3, padding=1)

    def forward(self, x):
        feat = self.intro(x)
        feat = self.body(feat)
        res = self.out(feat)
        return torch.clamp(
            x[:, :1] + res, 0.0, 1.0
        )  # global residual on the degraded channel


class NAFNetUNet(nn.Module):
    """Multi-scale encoder-decoder NAFNet (large receptive field + capacity) for strong blur."""

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        width=32,
        enc_blks=(2, 2, 4),
        middle_blks=8,
        dec_blks=(2, 2, 2),
    ):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encoders, self.downs, self.decoders, self.ups = (
            nn.ModuleList() for _ in range(4)
        )
        c = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blks)])
        for n in dec_blks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.out = nn.Conv2d(width, out_ch, 3, padding=1)

    def forward(self, x):
        f = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            f = enc(f)
            skips.append(f)
            f = down(f)
        f = self.middle(f)
        for dec, up, sk in zip(self.decoders, self.ups, reversed(skips)):
            f = up(f) + sk
            f = dec(f)
        return torch.clamp(x[:, :1] + self.out(f), 0.0, 1.0)


if __name__ == "__main__":
    for name, m in [("flat", NAFNetRestorer()), ("unet", NAFNetUNet(width=32))]:
        n = sum(p.numel() for p in m.parameters())
        out = m(torch.randn(2, 3, 256, 256))
        print(f"{name:6s} params {n / 1e6:.2f}M  out {tuple(out.shape)}")
