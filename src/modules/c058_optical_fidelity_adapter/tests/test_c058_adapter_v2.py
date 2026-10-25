from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_adapter_v2.py"
SPEC = importlib.util.spec_from_file_location("c058_adapter_v2_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def candidate(**changes):
    values = {
        "route_id": "safe",
        "alpha": -1.5,
        "sigma": 0.0,
        "cap_dn": 8.0,
        "maximum_headroom_limited_fraction": 1.0,
    }
    values.update(changes)
    return MODULE.SafeAdapterCandidate(**values)


def test_quantized_output_introduces_no_new_low_or_high_clipping():
    raw = np.array([[0, 1, 2, 253, 254, 255]], dtype=np.uint8)
    task_on = np.array([[8, 9, 10, 245, 246, 247]], dtype=np.uint8)
    output, diagnostics = MODULE.apply_safe_adapter(raw, task_on, candidate())

    assert output.tolist() == [[0, 1, 1, 254, 254, 255]]
    assert diagnostics.headroom_limited_fraction > 0
    assert diagnostics.introduced_clipping_fraction == 0
    assert not np.any((output == 0) & (raw != 0))
    assert not np.any((output == 255) & (raw != 255))


def test_existing_boundary_pixels_can_move_inward():
    raw = np.array([[0, 255]], dtype=np.uint8)
    task_on = np.array([[8, 247]], dtype=np.uint8)
    output, diagnostics = MODULE.apply_safe_adapter(
        raw, task_on, candidate(alpha=1.0)
    )

    assert output.tolist() == [[8, 247]]
    assert diagnostics.introduced_clipping_fraction == 0


def test_large_projection_fraction_triggers_exact_raw_bypass():
    raw = np.full((8, 8), 254, dtype=np.uint8)
    task_on = np.full((8, 8), 246, dtype=np.uint8)
    output, diagnostics = MODULE.apply_safe_adapter(
        raw,
        task_on,
        candidate(maximum_headroom_limited_fraction=0.10),
    )

    assert np.array_equal(output, raw)
    assert diagnostics.safety_bypass is True
    assert diagnostics.headroom_limited_fraction == 1.0


def test_returned_change_respects_alpha_times_cap_bound():
    raw = np.full((16, 16), 128, dtype=np.uint8)
    task_on = np.zeros((16, 16), dtype=np.uint8)
    output, diagnostics = MODULE.apply_safe_adapter(raw, task_on, candidate())

    assert diagnostics.max_abs_change_dn <= 12.0
    assert int(np.max(np.abs(output.astype(np.int16) - raw.astype(np.int16)))) <= 12
