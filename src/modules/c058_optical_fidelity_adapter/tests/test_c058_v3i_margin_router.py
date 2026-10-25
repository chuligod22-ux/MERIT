from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_v3i_margin_router.py"
SPEC = importlib.util.spec_from_file_location("c058_v3i_margin_router_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reference_condition_is_always_bypassed():
    assert not MODULE.route_is_active(reference_condition=True, optical_margin_cy_px=1.0)


def test_margin_gate_is_strictly_greater_than_frozen_threshold():
    assert not MODULE.route_is_active(reference_condition=False, optical_margin_cy_px=0.001)
    assert MODULE.route_is_active(reference_condition=False, optical_margin_cy_px=0.0010000001)


def test_inactive_route_is_byte_exact_and_does_not_call_model():
    raw = np.arange(64, dtype=np.uint8).reshape(8, 8)

    def forbidden(_raw):
        raise AssertionError("model must not be called for exact bypass")

    output, diagnostics = MODULE.route_native_crop(
        raw,
        active=False,
        model_callback=forbidden,
        candidate=None,
    )

    assert np.array_equal(output, raw)
    assert output is not raw
    assert diagnostics["exact_identity"] is True
    assert diagnostics["max_abs_change_dn"] == 0.0


def test_bootstrap_is_deterministic_and_order_independent():
    values = np.array([-0.2, -0.1, 0.0, 0.0, 0.0], dtype=float)
    first = MODULE.bootstrap_ci(values, count=2000, seed=41)
    second = MODULE.bootstrap_ci(values[::-1], count=2000, seed=41)
    assert first == second
