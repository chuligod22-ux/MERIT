from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from c058_adapter import AdapterCandidate, apply_adapter, build_candidate_grid


def test_identity_is_bit_exact_for_uint8_and_float() -> None:
    raw_u8 = np.arange(64, dtype=np.uint8).reshape(8, 8)
    on_u8 = np.flipud(raw_u8)
    identity = AdapterCandidate("identity", 0.0, 0.0, 0.0, 8.0)
    out_u8, diagnostics_u8 = apply_adapter(raw_u8, on_u8, identity)
    assert out_u8.dtype == np.uint8
    assert np.array_equal(out_u8, raw_u8)
    assert diagnostics_u8.exact_identity

    raw_f32 = raw_u8.astype(np.float32) / 255.0
    on_f32 = on_u8.astype(np.float32) / 255.0
    out_f32, diagnostics_f32 = apply_adapter(raw_f32, on_f32, identity)
    assert out_f32.dtype == np.float32
    assert np.array_equal(out_f32, raw_f32)
    assert diagnostics_f32.exact_identity


def test_reference_bypass_is_exact_for_nonidentity_candidate() -> None:
    raw = np.arange(100, dtype=np.uint8).reshape(10, 10)
    on = np.full((10, 10), 200, dtype=np.uint8)
    candidate = AdapterCandidate("inverse", -0.25, 1.5, 0.05, 4.0)
    output, diagnostics = apply_adapter(raw, on, candidate, reference_bypass=True)
    assert np.array_equal(output, raw)
    assert diagnostics.reference_bypass
    assert diagnostics.exact_identity


def test_inverse_residual_and_cap_have_declared_sign() -> None:
    raw = np.full((5, 5), 100, dtype=np.uint8)
    on = np.full((5, 5), 140, dtype=np.uint8)
    candidate = AdapterCandidate("inverse", -0.5, 0.0, 0.0, 8.0)
    output, diagnostics = apply_adapter(raw, on, candidate)
    # residual is capped to +8 DN; alpha=-0.5 therefore moves 4 DN away from ON.
    assert np.all(output == 96)
    assert diagnostics.residual_cap_fraction == pytest.approx(1.0)
    assert diagnostics.max_abs_change_dn == pytest.approx(4.0)


def test_overshoot_and_introduced_clipping_are_reported() -> None:
    raw = np.full((8, 8), 2, dtype=np.uint8)
    on = np.full((8, 8), 255, dtype=np.uint8)
    candidate = AdapterCandidate("inverse", -0.5, 0.0, 0.0, 8.0)
    output, diagnostics = apply_adapter(raw, on, candidate)
    assert np.all(output == 0)
    assert diagnostics.preclip_overshoot_fraction == pytest.approx(1.0)
    assert diagnostics.introduced_clipping_fraction == pytest.approx(1.0)


def test_candidate_grid_is_finite_unique_and_contains_required_ablation_axes() -> None:
    grid = build_candidate_grid()
    assert len(grid) <= 32
    assert len({candidate.route_id for candidate in grid}) == len(grid)
    assert {candidate.alpha for candidate in grid} >= {
        -0.5,
        -0.25,
        -0.125,
        0.0,
        0.125,
        0.25,
        0.5,
    }
    assert {candidate.sigma for candidate in grid} >= {0.0, 0.75, 1.5, 3.0}
    assert {candidate.sharp_gain for candidate in grid} >= {0.0, 0.025, 0.05}
    assert {candidate.cap_dn for candidate in grid} >= {2.0, 4.0, 8.0}
    assert any(c.alpha == 0.0 and c.sharp_gain == 0.025 for c in grid)
    assert any(c.alpha < 0.0 and c.sharp_gain > 0.0 for c in grid)


def test_invalid_float_range_is_rejected() -> None:
    raw = np.ones((8, 8), dtype=np.float32) * 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_adapter(raw, np.zeros_like(raw), build_candidate_grid()[0])
