"""Tests for the C055 matched factorial estimand."""

from __future__ import annotations

import numpy as np

import c055_common as analysis
import c055_confirm as confirmation


def test_factorial_main_effect_uses_marginal_backbone_means() -> None:
    rows = []
    gains = {"nafnet": 0.10, "plainunet": 0.12, "restormer": 0.14, "mamba": 0.16}
    for backbone, gain in gains.items():
        for tunnel in analysis.TUNNELS:
            rows.append(
                {
                    "backbone": backbone,
                    "tunnel": tunnel,
                    "parent": "p1",
                    "detector": "d1",
                    "raw": 0.20,
                    "framework": 0.20 + gain,
                }
            )
    result = analysis.effects(rows)
    assert np.isclose(result["framework_main"], 0.13)
    assert np.isclose(result["max_backbone_conditional"], 0.06)
    assert np.isclose(result["max_backbone_main"], 0.03)
    assert np.isclose(result["factorial_dominance"], 0.10)


def test_frozen_confirmation_fusions() -> None:
    raw = np.asarray([[[0.1, 0.7]]], dtype=np.float32)
    off = np.asarray([[[0.2, 0.6]]], dtype=np.float32)
    on = np.asarray([[[0.3, 0.5]]], dtype=np.float32)
    np.testing.assert_array_equal(
        confirmation.selected_fusion(raw, off, on, "max3"),
        np.asarray([[[0.3, 0.7]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        confirmation.selected_fusion(raw, off, on, "max_raw_off"),
        np.asarray([[[0.2, 0.7]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        confirmation.selected_fusion(raw, off, on, "mean3"),
        np.asarray([[[0.2, 0.6]]], dtype=np.float32),
    )
