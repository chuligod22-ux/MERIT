"""Released MERIT v2.4 endpoint routes.

Detection uses the cycle-5 detector-conditioned image profiles followed by a
fixed linear fusion of raw and profile probability maps.  Natural apparent-
width measurement uses the v2.4 inverse-residual profile and reference-blind
safe fallback validated in C016.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
MODIFIED = HERE.parent.parent
C014_SCRIPTS = MODIFIED / "c014_existing_four_tunnel_validation" / "scripts"
for directory in (HERE, C014_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import c014_common as width_common  # noqa: E402
import c016_profiles as width_v24  # noqa: E402


VERSION = "MERIT-v2.4"
DETECTION_VERSION = "MERIT-v2.4-cycle5"
DETECTION_PROFILES = {
    "d2_resnet34_unet": "sharp0025",
    "d2_vgg16_unet": "inv1250",
    "d2_deeplabv3_resnet50": "inv0125",
}
DETECTION_FUSION_FAMILY = "linear_probability"
DETECTION_FUSION_WEIGHTS = {
    "d2_resnet34_unet": 1.0,
    "d2_vgg16_unet": 0.75,
    "d2_deeplabv3_resnet50": 0.75,
}
WIDTH_PROFILE = "inv1250"
WIDTH_SAFE_FALLBACK = "raw_when_raw_estimable_and_profile_nonestimable"


@torch.inference_mode()
def detection_candidate(
    inputs: torch.Tensor,
    base_model: torch.nn.Module,
    detector_name: str,
) -> torch.Tensor:
    """Create the released detector-conditioned v2.4 profile image."""

    if detector_name not in DETECTION_PROFILES:
        raise ValueError(f"Unsupported detector: {detector_name}")
    profile = DETECTION_PROFILES[detector_name]
    return width_v24.make_candidates(inputs, base_model, [profile])[profile]


def fuse_detection_probabilities(
    raw_probability: np.ndarray,
    profile_probability: np.ndarray,
    detector_name: str,
) -> np.ndarray:
    """Fuse raw/profile detector probabilities with the released fixed weight."""

    if detector_name not in DETECTION_FUSION_WEIGHTS:
        raise ValueError(f"Unsupported detector: {detector_name}")
    raw = np.asarray(raw_probability, dtype=np.float32)
    profile = np.asarray(profile_probability, dtype=np.float32)
    if raw.shape != profile.shape:
        raise ValueError(
            f"Probability shape mismatch: raw={raw.shape}, profile={profile.shape}"
        )
    if not np.isfinite(raw).all() or not np.isfinite(profile).all():
        raise ValueError("Detection probabilities must be finite")
    weight = np.float32(DETECTION_FUSION_WEIGHTS[detector_name])
    return np.clip(raw + weight * (profile - raw), 0.0, 1.0).astype(
        np.float32, copy=False
    )


@torch.inference_mode()
def width_candidate(
    inputs: torch.Tensor,
    base_model: torch.nn.Module,
) -> torch.Tensor:
    """Apply the released v2.4 natural apparent-width image route."""

    return width_v24.make_candidates(inputs, base_model, [WIDTH_PROFILE])[WIDTH_PROFILE]


def measure_apparent_width(
    raw_image: np.ndarray,
    profile_image: np.ndarray,
    coordinates: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Measure apparent width in pixels with the v2.4 safe fallback.

    Both images must be normalized float32 grayscale arrays.  The routing
    decision uses estimator status only and never accesses a reference width.
    """

    raw = width_common.estimate_dark_width(raw_image, coordinates)
    candidate = width_common.estimate_dark_width(profile_image, coordinates)
    effective, fallback = width_v24.safe_width_measurement(raw, candidate)
    return {
        "version": VERSION,
        "unit": "pixel",
        "claim": "expert-mask-compatible natural crack apparent width",
        "raw": raw,
        "candidate": candidate,
        "effective": effective,
        "safe_fallback": fallback,
    }


def route(
    inputs: torch.Tensor,
    base_model: torch.nn.Module,
    *,
    task: str,
    detector_name: str | None = None,
) -> torch.Tensor:
    """Route a batch to the released endpoint-specific v2.4 image stage.

    Detection callers must run their detector on both the source image and
    this returned profile image, then call ``fuse_detection_probabilities``.
    """

    if task == "crack_detection":
        if detector_name is None:
            raise ValueError("crack_detection requires detector_name")
        return detection_candidate(inputs, base_model, detector_name)
    if task == "natural_apparent_width_px":
        return width_candidate(inputs, base_model)
    raise ValueError(f"Unsupported MERIT v2.4 task: {task}")
