"""Release-interface tests for MERIT v2.4 cycle 5."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import c016_release as release


class DummyBase(torch.nn.Module):
    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1] * 0.5


def test_detection_profiles_are_bounded_and_detector_conditioned() -> None:
    inputs = torch.linspace(0.0, 1.0, 49).reshape(1, 1, 7, 7).repeat(1, 3, 1, 1)
    outputs = {
        detector: release.detection_candidate(inputs, DummyBase(), detector)
        for detector in release.DETECTION_PROFILES
    }
    assert all(output.shape == (1, 1, 7, 7) for output in outputs.values())
    assert all(torch.all((0.0 <= output) & (output <= 1.0)) for output in outputs.values())
    assert not torch.equal(outputs["d2_vgg16_unet"], outputs["d2_deeplabv3_resnet50"])


@pytest.mark.parametrize(
    ("detector", "weight"),
    tuple(release.DETECTION_FUSION_WEIGHTS.items()),
)
def test_detection_probability_fusion_matches_released_weight(
    detector: str, weight: float
) -> None:
    raw = np.array([0.0, 0.2, 1.0], dtype=np.float32)
    profile = np.array([1.0, 0.6, 0.0], dtype=np.float32)
    actual = release.fuse_detection_probabilities(raw, profile, detector)
    expected = np.clip(raw + np.float32(weight) * (profile - raw), 0.0, 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    assert actual.dtype == np.float32


def test_detection_probability_fusion_fails_closed() -> None:
    raw = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported detector"):
        release.fuse_detection_probabilities(raw, raw, "unknown")
    with pytest.raises(ValueError, match="shape mismatch"):
        release.fuse_detection_probabilities(
            raw, np.zeros((2, 1), dtype=np.float32), "d2_resnet34_unet"
        )
    invalid = raw.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        release.fuse_detection_probabilities(invalid, raw, "d2_resnet34_unet")


def test_route_returns_detection_image_stage() -> None:
    inputs = torch.linspace(0.0, 1.0, 49).reshape(1, 1, 7, 7).repeat(1, 3, 1, 1)
    expected = release.detection_candidate(inputs, DummyBase(), "d2_vgg16_unet")
    actual = release.route(
        inputs,
        DummyBase(),
        task="crack_detection",
        detector_name="d2_vgg16_unet",
    )
    assert torch.equal(actual, expected)
