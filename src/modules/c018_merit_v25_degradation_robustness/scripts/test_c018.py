"""Focused checks for C018 degradation-conditioned restoration branches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402


def test_identity_is_exact() -> None:
    image = np.random.RandomState(1).rand(32, 48).astype(np.float32)
    restored = common.restore_physics(image, "identity", {}, 0.0)
    assert np.array_equal(restored, image)


def test_illumination_inverse_recovers_clean() -> None:
    clean = np.random.RandomState(2).uniform(0.05, 0.9, (32, 48)).astype(np.float32)
    parameters = {"edge_gain": 0.65, "vertical_gain": 0.85}
    y = np.linspace(0.0, 1.0, clean.shape[0], dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, clean.shape[1], dtype=np.float32)[None, :]
    field = (1.0 - (1.0 - parameters["edge_gain"]) * x**2) * (
        1.0 - (1.0 - parameters["vertical_gain"]) * y
    )
    degraded = clean * field
    restored = common.restore_physics(
        degraded, "illumination_gradient_vignetting", parameters
    )
    assert float(np.max(np.abs(restored - clean))) < 2e-7


def test_veiling_inverse_recovers_clean() -> None:
    clean = np.random.RandomState(3).uniform(0.05, 0.9, (32, 48)).astype(np.float32)
    parameters = {"contrast": 0.75, "veil": 0.1}
    degraded = clean * parameters["contrast"] + parameters["veil"]
    restored = common.restore_physics(degraded, "veiling_scattering", parameters)
    assert float(np.max(np.abs(restored - clean))) < 2e-7


def test_saturation_inverse_is_exact_below_plateau() -> None:
    clean = np.random.RandomState(4).uniform(0.0, 0.5, (32, 48)).astype(np.float32)
    parameters = {"gain": 1.2, "clip_high": 0.94}
    degraded = np.minimum(clean * parameters["gain"], parameters["clip_high"])
    degraded /= parameters["clip_high"]
    restored = common.restore_physics(degraded, "saturation_clipping", parameters)
    assert float(np.max(np.abs(restored - clean))) < 2e-7


def test_motion_kernel_is_normalized() -> None:
    kernel = common.motion_kernel(7, 5.0)
    assert kernel.shape == (7, 7)
    assert np.isclose(float(kernel.sum()), 1.0, atol=1e-6)
    assert np.all(kernel >= 0.0)


def test_development_protocol_is_explicitly_nonconfirmatory() -> None:
    protocol = common.load_json(common.DEVELOPMENT_PROTOCOL)
    assert protocol["status"] == "EXPLORATORY_DEVELOPMENT_ONLY"
    assert "confirmation" in protocol["scientific_boundary"]
