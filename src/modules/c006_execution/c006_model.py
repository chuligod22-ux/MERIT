"""Frozen C006 NAFNetUNet architecture.

Copied from the audited legacy ``03_restoration/03_src/model.py`` architecture,
but kept inside the C006 execution package so the exact implementation can be
hashed with each run.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(1, keepdim=True)
        variance = (tensor - mean).pow(2).mean(1, keepdim=True)
        normalized = (tensor - mean) / torch.sqrt(variance + self.eps)
        return (
            normalized * self.weight[None, :, None, None]
            + self.bias[None, :, None, None]
        )


class SimpleGate(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(
        self, channels: int, dw_expand: int = 2, ffn_expand: int = 2
    ) -> None:
        super().__init__()
        depthwise_channels = channels * dw_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, depthwise_channels, 1)
        self.conv2 = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            3,
            padding=1,
            groups=depthwise_channels,
        )
        self.simple_gate = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(depthwise_channels // 2, depthwise_channels // 2, 1),
        )
        self.conv3 = nn.Conv2d(depthwise_channels // 2, channels, 1)
        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = tensor
        transformed = self.norm1(tensor)
        transformed = self.conv2(self.conv1(transformed))
        transformed = self.simple_gate(transformed)
        transformed = transformed * self.sca(transformed)
        transformed = self.conv3(transformed)
        tensor = residual + transformed * self.beta
        transformed = self.conv5(
            self.simple_gate(self.conv4(self.norm2(tensor)))
        )
        return tensor + transformed * self.gamma


class NAFNetUNet(nn.Module):
    """Multi-scale encoder-decoder NAFNet used in the C006 comparison."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 32,
        encoder_blocks: tuple[int, ...] = (2, 2, 4),
        middle_blocks: int = 8,
        decoder_blocks: tuple[int, ...] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = width
        for block_count in encoder_blocks:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(channels) for _ in range(block_count)]
                )
            )
            self.downs.append(
                nn.Conv2d(channels, 2 * channels, 2, stride=2)
            )
            channels *= 2
        self.middle = nn.Sequential(
            *[NAFBlock(channels) for _ in range(middle_blocks)]
        )
        for block_count in decoder_blocks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, 2 * channels, 1),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(channels) for _ in range(block_count)]
                )
            )
        self.out = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.intro(tensor)
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = downsample(features)
        features = self.middle(features)
        for decoder, upsample, skip in zip(
            self.decoders, self.ups, reversed(skips)
        ):
            features = upsample(features) + skip
            features = decoder(features)
        return torch.clamp(
            tensor[:, :1] + self.out(features), min=0.0, max=1.0
        )
