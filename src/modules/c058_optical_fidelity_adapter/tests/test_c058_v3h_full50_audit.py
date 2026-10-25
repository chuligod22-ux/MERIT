from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_v3h_full50_audit.py"
SPEC = importlib.util.spec_from_file_location("c058_v3h_full50_audit_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bootstrap_is_deterministic_and_order_independent():
    values = np.array([-0.006, -0.004, -0.003, 0.0, 0.0, 0.0], dtype=float)
    forward = MODULE.bootstrap_ci(values, count=2000, seed=17)
    reverse = MODULE.bootstrap_ci(values[::-1], count=2000, seed=17)

    assert forward == reverse
    assert forward[1] <= 0


def test_frozen_route_covers_complete_50_condition_grid_without_worsening():
    table, summary = MODULE.build_condition_table()

    assert len(table) == 50
    assert int(table["available_frames"].sum()) == 479
    assert summary["active_condition_cells"] == 6
    assert summary["bypass_condition_cells"] == 44
    assert summary["improved_condition_cells"] == 6
    assert summary["preserved_exact_bypass_condition_cells"] == 44
    assert summary["worsened_condition_cells"] == 0
    assert summary["nonworse_condition_cells"] == 50
    assert all(summary["gate_checks"].values())
