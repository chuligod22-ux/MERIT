"""Prompt-conditioned residual wrapper used by the exploratory MERIT v2 pilot."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.scale * self.body(value)


class PromptResidualCorrector(nn.Module):
    """Predict a small local correction and a prompt-conditioned base blend."""

    def __init__(
        self,
        hidden_channels: int = 24,
        blocks: int = 3,
        max_delta: float = 0.08,
        blend_span: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden_channels < 4 or blocks < 1:
            raise ValueError("The corrector requires hidden_channels >= 4 and blocks >= 1")
        if max_delta <= 0.0 or not 0.0 < blend_span < 1.0:
            raise ValueError("Invalid correction bounds")
        self.max_delta = float(max_delta)
        self.blend_span = float(blend_span)
        self.stem = nn.Sequential(
            nn.Conv2d(5, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualBlock(hidden_channels) for _ in range(blocks))
        )
        self.head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.prompt_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.prompt_mlp[-1].weight)
        nn.init.zeros_(self.prompt_mlp[-1].bias)

    def forward(
        self,
        degraded: torch.Tensor,
        base: torch.Tensor,
        blur_prompt: torch.Tensor,
        noise_prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [degraded, base, base - degraded, blur_prompt, noise_prompt], dim=1
        )
        residual = self.max_delta * torch.tanh(
            self.head(self.blocks(self.stem(features)))
        )
        prompt_values = torch.cat(
            [
                blur_prompt.mean(dim=(2, 3)),
                noise_prompt.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        blend = 1.0 + self.blend_span * torch.tanh(self.prompt_mlp(prompt_values))
        return residual, blend[:, :, None, None]


class MeritV2(nn.Module):
    """Freeze a C009 base restorer and learn only a bounded prompt adapter.

    A zero prompt returns the source exactly. At initialization every positive
    prompt returns the frozen base output exactly.
    """

    def __init__(self, base: nn.Module, corrector: PromptResidualCorrector) -> None:
        super().__init__()
        self.base = base
        self.corrector = corrector
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True) -> "MeritV2":
        super().train(mode)
        self.base.eval()
        return self

    def _validate(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                "MERIT v2 input must have shape [batch,3,height,width]; "
                f"received {tuple(inputs.shape)}"
            )

    def forward_components(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate(inputs)
        degraded = inputs[:, :1]
        blur_prompt = inputs[:, 1:2]
        noise_prompt = inputs[:, 2:3]
        zero_prompt = torch.zeros_like(degraded)
        base_inputs = torch.cat([degraded, zero_prompt, zero_prompt], dim=1)
        with torch.no_grad():
            base = self.base.forward_preclip(base_inputs)
        residual, blend = self.corrector(
            degraded, base, blur_prompt, noise_prompt
        )
        active = (
            torch.amax(torch.abs(inputs[:, 1:3]), dim=(1, 2, 3), keepdim=True) > 0
        ).to(degraded.dtype)
        restored = degraded + active * (blend * (base - degraded) + residual)
        return restored, base, residual, blend

    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_components(inputs)[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.forward_preclip(inputs), 0.0, 1.0)


class CalibratedPromptBlend(nn.Module):
    """Blend a frozen base through a bounded piecewise-linear prompt curve."""

    def __init__(
        self,
        base: nn.Module,
        severity_anchors: list[float],
        blend_anchors: list[float],
    ) -> None:
        super().__init__()
        if len(severity_anchors) != len(blend_anchors) or len(severity_anchors) < 2:
            raise ValueError("Prompt calibration requires matching anchor lists")
        if any(
            right <= left
            for left, right in zip(severity_anchors, severity_anchors[1:])
        ):
            raise ValueError("Severity anchors must be strictly increasing")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.register_buffer(
            "severity_anchors", torch.tensor(severity_anchors, dtype=torch.float32)
        )
        self.register_buffer(
            "blend_anchors", torch.tensor(blend_anchors, dtype=torch.float32)
        )

    def train(self, mode: bool = True) -> "CalibratedPromptBlend":
        super().train(mode)
        self.base.eval()
        return self

    def _interpolate(self, severity: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(
            severity,
            float(self.severity_anchors[0]),
            float(self.severity_anchors[-1]),
        )
        indices = torch.searchsorted(self.severity_anchors, clipped, right=True) - 1
        indices = torch.clamp(indices, 0, len(self.severity_anchors) - 2)
        x0 = self.severity_anchors[indices]
        x1 = self.severity_anchors[indices + 1]
        y0 = self.blend_anchors[indices]
        y1 = self.blend_anchors[indices + 1]
        weight = (clipped - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0)

    def forward_components(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                "MERIT v2 input must have shape [batch,3,height,width]; "
                f"received {tuple(inputs.shape)}"
            )
        source = inputs[:, :1]
        prompts = inputs[:, 1:3]
        zeros = torch.zeros_like(source)
        with torch.no_grad():
            base = self.base.forward_preclip(torch.cat([source, zeros, zeros], dim=1))
        severity = prompts.float().mean(dim=(1, 2, 3))
        blend = self._interpolate(severity)[:, None, None, None].to(source.dtype)
        active = (torch.amax(torch.abs(prompts), dim=(1, 2, 3), keepdim=True) > 0).to(
            source.dtype
        )
        output = source + active * blend * (base - source)
        return output, base, torch.zeros_like(output), blend

    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_components(inputs)[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.forward_preclip(inputs), 0.0, 1.0)
