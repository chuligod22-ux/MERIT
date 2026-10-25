from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_v3h_all50_efficacy.py"
SPEC = importlib.util.spec_from_file_location("c058_v3h_all50_efficacy_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bootstrap_is_deterministic_and_order_independent():
    values = np.array([-0.2, -0.1, 0.0, 0.05, 0.1], dtype=float)
    first = MODULE.bootstrap_ci(values, count=2000, seed=19)
    reversed_values = MODULE.bootstrap_ci(values[::-1], count=2000, seed=19)
    assert first == reversed_values


def test_classification_preserves_all_directions():
    assert MODULE.classify(-1e-9) == "improved"
    assert MODULE.classify(0.0) == "tied"
    assert MODULE.classify(1e-9) == "worsened"


def test_selected_route_is_the_existing_frozen_v3h_route():
    route = MODULE.selected_route()
    assert route["run_key"] == "safe_s1_am0.75"
    assert route["model_seed"] == 1
    assert route["alpha"] == -0.75
    assert route["sigma_px"] == 0.75
    assert route["residual_cap_dn"] == 8.0
