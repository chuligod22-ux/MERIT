"""C017 non-CNN detector models with one-logit 512 px output contracts."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Mamba2Config, Mamba2Model, SegformerForSemanticSegmentation

MODEL_NAMES = ("segformer_b0", "mamba2_unet")


class C017ModelError(RuntimeError):
    """Raised when a C017 architecture or checkpoint violates its contract."""


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.body = ConvBlock(out_channels, out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.body(self.down(inputs))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(torch.cat((inputs, skip), dim=1))


class SegformerBinary(nn.Module):
    """NVIDIA SegFormer-B0 encoder/decoder with a new one-logit crack head."""

    def __init__(self, pretrained_path: Path) -> None:
        super().__init__()
        if not pretrained_path.is_dir():
            raise C017ModelError(f"Missing SegFormer asset directory: {pretrained_path}")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            str(pretrained_path),
            local_files_only=True,
            num_labels=1,
            id2label={0: "crack"},
            label2id={"crack": 0},
            ignore_mismatched_sizes=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        logits = self.model(pixel_values=inputs).logits
        return F.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)


class Mamba2UNet(nn.Module):
    """U-Net local encoder/decoder with an official Mamba2 global bottleneck.

    The 512 px input is reduced to 32 x 32 before the Mamba2 block. This keeps
    the official PyTorch fallback tractable on Windows while retaining local
    high-resolution skip features for thin cracks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBlock(3, 32)
        self.down1 = DownBlock(32, 48)
        self.down2 = DownBlock(48, 64)
        self.down3 = DownBlock(64, 96)
        self.down4 = DownBlock(96, 128)
        config = Mamba2Config(
            hidden_size=128,
            state_size=16,
            num_hidden_layers=2,
            expand=2,
            conv_kernel=4,
            n_groups=8,
            head_dim=32,
            num_heads=8,
            use_cache=False,
            residual_in_fp32=True,
        )
        self.mamba = Mamba2Model(config)
        self.up3 = UpBlock(128, 96, 96)
        self.up2 = UpBlock(96, 64, 64)
        self.up1 = UpBlock(64, 48, 48)
        self.up0 = UpBlock(48, 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        x0 = self.stem(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        batch, channels, height, width = x4.shape
        tokens = x4.flatten(2).transpose(1, 2).contiguous()
        tokens = self.mamba(inputs_embeds=tokens, use_cache=False).last_hidden_state
        x4 = tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
        output = self.up3(x4, x3)
        output = self.up2(output, x2)
        output = self.up1(output, x1)
        output = self.up0(output, x0)
        return self.head(output)


def build_model(model_name: str, segformer_asset: Path) -> nn.Module:
    if model_name == "segformer_b0":
        return SegformerBinary(segformer_asset)
    if model_name == "mamba2_unet":
        return Mamba2UNet()
    raise C017ModelError(f"Unknown model: {model_name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def validate_output(model: nn.Module, device: torch.device, size: int = 512) -> dict[str, object]:
    model = model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(torch.zeros(1, 3, size, size, device=device))
    if tuple(output.shape) != (1, 1, size, size):
        raise C017ModelError(f"Unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise C017ModelError("Non-finite model output")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "parameters": parameter_count(model),
    }
