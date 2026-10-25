from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_v3h_all50_recalculate.py"
SPEC = importlib.util.spec_from_file_location("c058_v3h_all50_recalculate_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bootstrap_is_deterministic_and_order_independent():
    values = np.array([-0.4, -0.1, 0.0, 0.2], dtype=float)
    first = MODULE.bootstrap_ci(values, count=2000, seed=31)
    second = MODULE.bootstrap_ci(values[::-1], count=2000, seed=31)
    assert first == second
