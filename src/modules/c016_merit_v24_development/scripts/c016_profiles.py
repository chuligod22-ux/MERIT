"""Bounded MERIT v2.4 development profiles.

Every active route is a deterministic perturbation of the source image.  The
restoration residual is reversed because the frozen v2.3 experiments showed
that the base restorer can attenuate thin cracks.  The unsharp term is kept
small and bounded.  An exact identity route is always available.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F


IDENTITY = "identity"

# Frozen before the v2.4 all-tunnel candidate screen.  Positive
# residual_gain means source + gain * (source - base_restoration).
CANDIDATES: dict[str, dict[str, float | str]] = {
    "inv0125": {"family": "inverse_base_residual", "residual_gain": 0.125, "sharp_gain": 0.0},
    "inv0250": {"family": "inverse_base_residual", "residual_gain": 0.250, "sharp_gain": 0.0},
    "inv0375": {"family": "inverse_base_residual", "residual_gain": 0.375, "sharp_gain": 0.0},
    "inv0500": {"family": "inverse_base_residual", "residual_gain": 0.500, "sharp_gain": 0.0},
    "inv0750": {"family": "inverse_base_residual", "residual_gain": 0.750, "sharp_gain": 0.0},
    "inv1000": {"family": "inverse_base_residual", "residual_gain": 1.000, "sharp_gain": 0.0},
    "inv1250": {"family": "inverse_base_residual", "residual_gain": 1.250, "sharp_gain": 0.0},
    "sharp0025": {"family": "unsharp", "residual_gain": 0.0, "sharp_gain": 0.025},
    "sharp0050": {"family": "unsharp", "residual_gain": 0.0, "sharp_gain": 0.050},
    "sharp0075": {"family": "unsharp", "residual_gain": 0.0, "sharp_gain": 0.075},
    "sharp0100": {"family": "unsharp", "residual_gain": 0.0, "sharp_gain": 0.100},
    "combo_a050_g005": {"family": "combined", "residual_gain": 0.500, "sharp_gain": 0.050},
    "combo_a075_g005": {"family": "combined", "residual_gain": 0.750, "sharp_gain": 0.050},
}


def gaussian_blur(source: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply the frozen 7x7 separable-equivalent Gaussian kernel."""

    radius = 3
    axis = torch.arange(-radius, radius + 1, device=source.device, dtype=torch.float32)
    kernel_1d = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).to(source.dtype)[None, None]
    padded = F.pad(source, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel)


@torch.inference_mode()
def make_candidates(
    inputs: torch.Tensor,
    base_model: torch.nn.Module,
    names: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Generate requested candidate tensors, including exact identity."""

    requested = tuple(names) if names is not None else tuple(CANDIDATES)
    unknown = set(requested) - ({IDENTITY} | set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown v2.4 profiles: {sorted(unknown)}")
    source = inputs[:, :1]
    outputs: dict[str, torch.Tensor] = {}
    if IDENTITY in requested:
        outputs[IDENTITY] = source
    active = [name for name in requested if name != IDENTITY]
    if not active:
        return outputs
    requires_base = any(float(CANDIDATES[name]["residual_gain"]) != 0.0 for name in active)
    requires_sharp = any(float(CANDIDATES[name]["sharp_gain"]) != 0.0 for name in active)
    base = base_model.forward_preclip(inputs) if requires_base else source
    highpass = source - gaussian_blur(source) if requires_sharp else torch.zeros_like(source)
    for name in active:
        specification = CANDIDATES[name]
        residual_gain = float(specification["residual_gain"])
        sharp_gain = float(specification["sharp_gain"])
        outputs[name] = torch.clamp(
            source + residual_gain * (source - base) + sharp_gain * highpass,
            0.0,
            1.0,
        )
    return outputs


def profile_magnitude(name: str) -> float:
    """Return a deterministic tie-break magnitude; identity is smallest."""

    if name == IDENTITY:
        return 0.0
    specification: dict[str, Any] = CANDIDATES[name]
    return float(specification["residual_gain"]) + float(specification["sharp_gain"])


def safe_width_measurement(
    raw: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Use raw width when the active transform loses an estimable raw ROI.

    This routing rule has no access to the reference width.  It only uses the
    estimator's predeclared status and therefore guarantees that an estimable
    raw ROI is not dropped by the active profile.
    """

    fallback = raw.get("status") == "ESTIMABLE" and candidate.get("status") != "ESTIMABLE"
    if fallback:
        return dict(raw), True
    return dict(candidate), False
