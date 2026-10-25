"""Frozen candidate transforms for the C012 MERIT v2.3 field package."""

from __future__ import annotations

import torch
import torch.nn.functional as F


DETECTOR_PROFILE = {
    "d2_resnet34_unet": "identity",
    "d2_vgg16_unet": "inverse_base_residual_0p25",
    "d2_deeplabv3_resnet50": "unsharp_0p10",
}
WIDTH_PROFILE = "inverse_base_residual_0p50"


def gaussian_blur(source: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    radius = 3
    axis = torch.arange(-radius, radius + 1, device=source.device, dtype=torch.float32)
    kernel_1d = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).to(source.dtype)[None, None]
    padded = F.pad(source, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel)


@torch.inference_mode()
def detector_candidate(
    inputs: torch.Tensor,
    base_model: torch.nn.Module,
    detector_name: str,
) -> torch.Tensor:
    """Return the active detector-conditioned route.

    Exact identity is handled by the public router when the prompt is zero.
    This function implements only the explicitly active field-detection route.
    """
    if detector_name not in DETECTOR_PROFILE:
        raise ValueError(f"Unsupported detector profile: {detector_name}")
    source = inputs[:, :1]
    profile = DETECTOR_PROFILE[detector_name]
    if profile == "identity":
        output = source
    elif profile == "unsharp_0p10":
        output = source + 0.10 * (source - gaussian_blur(source))
    elif profile == "inverse_base_residual_0p25":
        base = base_model.forward_preclip(inputs)
        output = source - 0.25 * (base - source)
    else:
        raise AssertionError(profile)
    return torch.clamp(output, 0.0, 1.0)


@torch.inference_mode()
def width_candidate(inputs: torch.Tensor, base_model: torch.nn.Module) -> torch.Tensor:
    source = inputs[:, :1]
    base = base_model.forward_preclip(inputs)
    return torch.clamp(source - 0.50 * (base - source), 0.0, 1.0)


@torch.inference_mode()
def route(
    inputs: torch.Tensor,
    base_model: torch.nn.Module,
    *,
    active: bool,
    task: str,
    detector_name: str | None = None,
) -> torch.Tensor:
    """Public prompt router with an exact zero-prompt identity path."""
    source = inputs[:, :1]
    if not active:
        return source
    if task == "crack_detection":
        if detector_name is None:
            raise ValueError("crack_detection requires detector_name")
        return detector_candidate(inputs, base_model, detector_name)
    if task == "printed_width":
        return width_candidate(inputs, base_model)
    raise ValueError(f"Unsupported C012 task: {task}")
