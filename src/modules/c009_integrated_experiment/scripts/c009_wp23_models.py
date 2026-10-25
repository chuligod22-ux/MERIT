"""Frozen parameter-matched NAFNet structures for C009 WP3."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
NAFNET_ROOT = C009_ROOT / "third_party" / "NAFNet"
BRANCH_KWARGS = {
    "img_channel": 3,
    "width": 22,
    "middle_blk_num": 1,
    "enc_blk_nums": [1, 1, 1, 28],
    "dec_blk_nums": [1, 1, 1, 1],
}
EXPECTED_BRANCH_PARAMETERS = 8_148_077
EXPECTED_SINGLE_PARAMETERS = 17_111_907
EXPECTED_PARAMETERS = {
    "cascade": 16_296_154,
    "parallel": 16_296_158,
}


class ModelContractError(RuntimeError):
    """Raised when the frozen WP3 model contract is violated."""


def official_nafnet_class() -> type[nn.Module]:
    root = str(NAFNET_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("basicsr.models.archs.NAFNet_arch")
    model_class = getattr(module, "NAFNetLocal")
    return model_class


def build_branch() -> nn.Module:
    branch = official_nafnet_class()(**BRANCH_KWARGS)
    count = sum(parameter.numel() for parameter in branch.parameters())
    if count != EXPECTED_BRANCH_PARAMETERS:
        raise ModelContractError(
            f"NAFNet branch parameter mismatch: {count} != {EXPECTED_BRANCH_PARAMETERS}"
        )
    return branch


class CascadeRestorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blur_stage = build_branch()
        self.noise_stage = build_branch()

    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        image = inputs[:, 0:1]
        blur_prompt = inputs[:, 1:2]
        noise_prompt = inputs[:, 2:3]
        zeros = torch.zeros_like(image)
        blur_output = self.blur_stage(
            torch.cat([image, blur_prompt, zeros], dim=1)
        )[:, 0:1]
        return self.noise_stage(
            torch.cat([blur_output, zeros, noise_prompt], dim=1)
        )[:, 0:1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.forward_preclip(inputs), 0.0, 1.0)


class ParallelRestorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blur_branch = build_branch()
        self.noise_branch = build_branch()
        self.fusion = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[0, 0, 0, 0] = 0.5
            self.fusion.weight[0, 1, 0, 0] = 0.5
            self.fusion.bias.zero_()

    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        image = inputs[:, 0:1]
        blur_prompt = inputs[:, 1:2]
        noise_prompt = inputs[:, 2:3]
        zeros = torch.zeros_like(image)
        blur_output = self.blur_branch(
            torch.cat([image, blur_prompt, zeros], dim=1)
        )[:, 0:1]
        noise_output = self.noise_branch(
            torch.cat([image, zeros, noise_prompt], dim=1)
        )[:, 0:1]
        return self.fusion(torch.cat([blur_output, noise_output, image], dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.forward_preclip(inputs), 0.0, 1.0)


def build_structure(name: str) -> tuple[nn.Module, dict[str, object]]:
    if name == "cascade":
        model: nn.Module = CascadeRestorer()
    elif name == "parallel":
        model = ParallelRestorer()
    else:
        raise ValueError(f"Unknown WP3 structure: {name}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected = EXPECTED_PARAMETERS[name]
    if parameters != expected or trainable != expected:
        raise ModelContractError(
            f"{name} parameter mismatch: {parameters}/{trainable} != {expected}"
        )
    relative = parameters / EXPECTED_SINGLE_PARAMETERS - 1.0
    if abs(relative) > 0.05:
        raise ModelContractError(f"{name} exceeds the frozen ±5% parameter gate")
    return model, {
        "structure": name,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "single_parameters": EXPECTED_SINGLE_PARAMETERS,
        "relative_to_single": relative,
        "branch_class": "basicsr.models.archs.NAFNet_arch.NAFNetLocal",
        "branch_kwargs": BRANCH_KWARGS,
    }
