"""Quantization-safe optical-fidelity adapter used by MERIT C058 v3h.

The v1 adapter bounded the restoration residual but clipped only after applying
it.  That allowed bright pixels to cross the uint8 boundary, especially for
the negative-alpha route.  This module projects every proposed change into the
available per-pixel headroom before uint8 quantization and reports both the
unprojected and returned-image safety diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SafeAdapterCandidate:
    route_id: str
    alpha: float
    sigma: float = 0.75
    sharp_gain: float = 0.0
    cap_dn: float = 8.0
    maximum_headroom_limited_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.route_id:
            raise ValueError("route_id must be non-empty")
        if self.sigma < 0 or self.sharp_gain < 0 or self.cap_dn <= 0:
            raise ValueError("invalid adapter parameters")
        if not 0 <= self.maximum_headroom_limited_fraction <= 1:
            raise ValueError("maximum_headroom_limited_fraction must be in [0, 1]")


@dataclass(frozen=True)
class SafeAdapterDiagnostics:
    exact_identity: bool
    safety_bypass: bool
    residual_cap_fraction: float
    uncapped_residual_mean_abs_dn: float
    capped_residual_mean_abs_dn: float
    unprojected_preclip_overshoot_fraction: float
    unprojected_below_zero_fraction: float
    unprojected_above_255_fraction: float
    headroom_limited_fraction: float
    introduced_clipping_fraction: float
    introduced_low_clipping_fraction: float
    introduced_high_clipping_fraction: float
    mean_abs_change_dn: float
    max_abs_change_dn: float

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return image
    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )


def _identity(*, safety_bypass: bool = False) -> SafeAdapterDiagnostics:
    return SafeAdapterDiagnostics(
        exact_identity=True,
        safety_bypass=safety_bypass,
        residual_cap_fraction=0.0,
        uncapped_residual_mean_abs_dn=0.0,
        capped_residual_mean_abs_dn=0.0,
        unprojected_preclip_overshoot_fraction=0.0,
        unprojected_below_zero_fraction=0.0,
        unprojected_above_255_fraction=0.0,
        headroom_limited_fraction=0.0,
        introduced_clipping_fraction=0.0,
        introduced_low_clipping_fraction=0.0,
        introduced_high_clipping_fraction=0.0,
        mean_abs_change_dn=0.0,
        max_abs_change_dn=0.0,
    )


def apply_safe_adapter(
    raw: np.ndarray,
    task_on: np.ndarray,
    candidate: SafeAdapterCandidate,
) -> tuple[np.ndarray, SafeAdapterDiagnostics]:
    """Apply a bounded residual while preventing new uint8 0/255 clipping.

    Pixels that were originally inside ``[1, 254]`` remain in that interval.
    Existing 0/255 pixels may move inward, while outward changes are projected
    to zero.  A candidate can request exact RAW bypass when projection would be
    needed for too large a fraction of the crop.
    """

    if raw.ndim != 2 or task_on.ndim != 2 or raw.shape != task_on.shape:
        raise ValueError("raw and task_on must be equally shaped 2D images")
    if raw.dtype != np.uint8 or task_on.dtype != np.uint8:
        raise ValueError("v3h safe adapter requires uint8 images")
    if candidate.alpha == 0 and candidate.sharp_gain == 0:
        return raw.copy(), _identity()

    raw_dn = raw.astype(np.float32)
    on_dn = task_on.astype(np.float32)
    residual = on_dn - raw_dn
    capped = np.clip(residual, -candidate.cap_dn, candidate.cap_dn)
    filtered = _gaussian(capped, candidate.sigma)
    if candidate.sharp_gain:
        sharp = raw_dn - _gaussian(raw_dn, 1.0)
    else:
        sharp = np.zeros_like(raw_dn)
    delta = np.float32(candidate.alpha) * filtered + np.float32(candidate.sharp_gain) * sharp
    proposed = raw_dn + delta
    below = proposed < 0.0
    above = proposed > 255.0

    lower = np.where(raw == 0, 0.0, 1.0).astype(np.float32)
    upper = np.where(raw == 255, 255.0, 254.0).astype(np.float32)
    projected = np.minimum(np.maximum(proposed, lower), upper)
    limited = np.abs(projected - proposed) > np.float32(1e-6)
    limited_fraction = float(np.mean(limited))
    if limited_fraction > candidate.maximum_headroom_limited_fraction:
        diagnostics = _identity(safety_bypass=True)
        return raw.copy(), SafeAdapterDiagnostics(
            **{
                **diagnostics.to_dict(),
                "residual_cap_fraction": float(np.mean(np.abs(residual) > candidate.cap_dn)),
                "uncapped_residual_mean_abs_dn": float(np.mean(np.abs(residual))),
                "capped_residual_mean_abs_dn": float(np.mean(np.abs(capped))),
                "unprojected_preclip_overshoot_fraction": float(np.mean(below | above)),
                "unprojected_below_zero_fraction": float(np.mean(below)),
                "unprojected_above_255_fraction": float(np.mean(above)),
                "headroom_limited_fraction": limited_fraction,
            }
        )

    output = np.rint(projected).astype(np.uint8)
    introduced_low = (output == 0) & (raw != 0)
    introduced_high = (output == 255) & (raw != 255)
    change = np.abs(output.astype(np.int16) - raw.astype(np.int16)).astype(np.float32)
    diagnostics = SafeAdapterDiagnostics(
        exact_identity=bool(np.array_equal(output, raw)),
        safety_bypass=False,
        residual_cap_fraction=float(np.mean(np.abs(residual) > candidate.cap_dn)),
        uncapped_residual_mean_abs_dn=float(np.mean(np.abs(residual))),
        capped_residual_mean_abs_dn=float(np.mean(np.abs(capped))),
        unprojected_preclip_overshoot_fraction=float(np.mean(below | above)),
        unprojected_below_zero_fraction=float(np.mean(below)),
        unprojected_above_255_fraction=float(np.mean(above)),
        headroom_limited_fraction=limited_fraction,
        introduced_clipping_fraction=float(np.mean(introduced_low | introduced_high)),
        introduced_low_clipping_fraction=float(np.mean(introduced_low)),
        introduced_high_clipping_fraction=float(np.mean(introduced_high)),
        mean_abs_change_dn=float(np.mean(change)),
        max_abs_change_dn=float(np.max(change)),
    )
    return output, diagnostics
