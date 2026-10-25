"""Deterministic bounded optical-fidelity adapter for MERIT C058.

The adapter is deliberately small and auditable.  It consumes the frozen RAW
crop and the corresponding C006 task-ON restoration, caps the ON residual in
digital-number (DN) units, and optionally applies a low-pass residual and a
small RAW-domain unsharp term.  A negative ``alpha`` is the predeclared
inverse-residual route ``RAW + |alpha| * (RAW - ON)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AdapterCandidate:
    """One member of the finite, predeclared C058 candidate family."""

    route_id: str
    alpha: float
    sigma: float
    sharp_gain: float
    cap_dn: float

    def __post_init__(self) -> None:
        if not self.route_id:
            raise ValueError("route_id must be non-empty")
        if self.sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        if self.sharp_gain < 0.0:
            raise ValueError("sharp_gain must be nonnegative")
        if self.cap_dn <= 0.0:
            raise ValueError("cap_dn must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdapterCandidate":
        return cls(
            route_id=str(payload["route_id"]),
            alpha=float(payload["alpha"]),
            sigma=float(payload["sigma"]),
            sharp_gain=float(payload["sharp_gain"]),
            cap_dn=float(payload["cap_dn"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterDiagnostics:
    """Pixel-domain diagnostics computed before and after clipping."""

    exact_identity: bool
    reference_bypass: bool
    residual_cap_fraction: float
    uncapped_residual_mean_abs_dn: float
    capped_residual_mean_abs_dn: float
    preclip_overshoot_fraction: float
    preclip_below_zero_fraction: float
    preclip_above_one_fraction: float
    introduced_clipping_fraction: float
    introduced_low_clipping_fraction: float
    introduced_high_clipping_fraction: float
    mean_abs_change_dn: float
    max_abs_change_dn: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(image: np.ndarray, label: str) -> tuple[np.ndarray, np.dtype]:
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional NumPy array")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / np.float32(255.0), image.dtype
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"{label} must use uint8 or floating dtype")
    if not np.isfinite(image).all():
        raise ValueError(f"{label} contains non-finite pixels")
    if float(np.min(image)) < 0.0 or float(np.max(image)) > 1.0:
        raise ValueError(f"{label} floating pixels must be in [0, 1]")
    return image.astype(np.float32, copy=False), image.dtype


def _restore_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.dtype(np.uint8):
        return np.rint(image * np.float32(255.0)).astype(np.uint8)
    return image.astype(dtype, copy=False)


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return image
    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )


def _identity_diagnostics(reference_bypass: bool) -> AdapterDiagnostics:
    return AdapterDiagnostics(
        exact_identity=True,
        reference_bypass=reference_bypass,
        residual_cap_fraction=0.0,
        uncapped_residual_mean_abs_dn=0.0,
        capped_residual_mean_abs_dn=0.0,
        preclip_overshoot_fraction=0.0,
        preclip_below_zero_fraction=0.0,
        preclip_above_one_fraction=0.0,
        introduced_clipping_fraction=0.0,
        introduced_low_clipping_fraction=0.0,
        introduced_high_clipping_fraction=0.0,
        mean_abs_change_dn=0.0,
        max_abs_change_dn=0.0,
    )


def apply_adapter(
    raw: np.ndarray,
    task_on: np.ndarray,
    candidate: AdapterCandidate,
    *,
    reference_bypass: bool = False,
) -> tuple[np.ndarray, AdapterDiagnostics]:
    """Apply one bounded candidate and preserve the RAW input dtype.

    ``reference_bypass`` is mandatory for the 60 km/h ISO100 safety-reference
    cells.  It returns a bit-exact RAW copy regardless of candidate values.
    """

    raw_float, raw_dtype = _normalise(raw, "raw")
    on_float, _ = _normalise(task_on, "task_on")
    if raw.shape != task_on.shape:
        raise ValueError("raw and task_on shapes must match")
    identity = candidate.alpha == 0.0 and candidate.sharp_gain == 0.0
    if reference_bypass or identity:
        return raw.copy(), _identity_diagnostics(reference_bypass)

    residual = on_float - raw_float
    cap = np.float32(candidate.cap_dn / 255.0)
    capped = np.clip(residual, -cap, cap)
    filtered_residual = _gaussian(capped, candidate.sigma)
    raw_lowpass = _gaussian(raw_float, 1.0) if candidate.sharp_gain else raw_float
    sharp_component = raw_float - raw_lowpass
    preclip = (
        raw_float
        + np.float32(candidate.alpha) * filtered_residual
        + np.float32(candidate.sharp_gain) * sharp_component
    )
    below = preclip < 0.0
    above = preclip > 1.0
    clipped = np.clip(preclip, 0.0, 1.0)
    introduced_low = (clipped <= 0.0) & (raw_float > 0.0)
    introduced_high = (clipped >= 1.0) & (raw_float < 1.0)
    preclip_change = np.abs(preclip - raw_float) * np.float32(255.0)
    diagnostics = AdapterDiagnostics(
        exact_identity=bool(np.array_equal(clipped, raw_float)),
        reference_bypass=False,
        residual_cap_fraction=float(np.mean(np.abs(residual) > cap)),
        uncapped_residual_mean_abs_dn=float(np.mean(np.abs(residual)) * 255.0),
        capped_residual_mean_abs_dn=float(np.mean(np.abs(capped)) * 255.0),
        preclip_overshoot_fraction=float(np.mean(below | above)),
        preclip_below_zero_fraction=float(np.mean(below)),
        preclip_above_one_fraction=float(np.mean(above)),
        introduced_clipping_fraction=float(np.mean(introduced_low | introduced_high)),
        introduced_low_clipping_fraction=float(np.mean(introduced_low)),
        introduced_high_clipping_fraction=float(np.mean(introduced_high)),
        mean_abs_change_dn=float(np.mean(preclip_change)),
        max_abs_change_dn=float(np.max(preclip_change)),
    )
    return _restore_dtype(clipped, raw_dtype), diagnostics


def build_candidate_grid() -> list[AdapterCandidate]:
    """Return the fixed 24-member development grid in deterministic order."""

    candidates = [AdapterCandidate("identity", 0.0, 0.0, 0.0, 8.0)]
    cap_for_alpha = {0.125: 2.0, 0.25: 4.0, 0.5: 8.0}
    for alpha in (-0.5, -0.25, -0.125, 0.125, 0.25, 0.5):
        cap = cap_for_alpha[abs(alpha)]
        sign = "inverse" if alpha < 0 else "forward"
        candidates.append(
            AdapterCandidate(
                f"plain_{sign}_a{abs(alpha):.3f}_cap{int(cap)}",
                alpha,
                0.0,
                0.0,
                cap,
            )
        )
    for alpha in (-0.5, -0.25, -0.125):
        for sigma in (0.75, 1.5, 3.0):
            cap = cap_for_alpha[abs(alpha)]
            candidates.append(
                AdapterCandidate(
                    f"band_inverse_a{abs(alpha):.3f}_s{sigma:.2f}_cap{int(cap)}",
                    alpha,
                    sigma,
                    0.0,
                    cap,
                )
            )
    for alpha in (-0.5, -0.25, -0.125):
        for sharp_gain in (0.025, 0.05):
            cap = cap_for_alpha[abs(alpha)]
            candidates.append(
                AdapterCandidate(
                    f"lowsharp_inverse_a{abs(alpha):.3f}_s1.50_g{sharp_gain:.3f}_cap{int(cap)}",
                    alpha,
                    1.5,
                    sharp_gain,
                    cap,
                )
            )
    for sharp_gain in (0.025, 0.05):
        candidates.append(
            AdapterCandidate(
                f"raw_only_unsharp_g{sharp_gain:.3f}",
                0.0,
                0.0,
                sharp_gain,
                8.0,
            )
        )
    if len({candidate.route_id for candidate in candidates}) != len(candidates):
        raise RuntimeError("C058 candidate route IDs are not unique")
    return candidates
