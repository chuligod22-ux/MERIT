from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "d0_cam1_eval.py"
SPEC = importlib.util.spec_from_file_location("d0_cam1_eval_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_vertical_edge(height: int = 256, width: int = 192) -> np.ndarray:
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    boundary = width / 2 + 0.08 * (rows - height / 2)
    signed = cols[None, :] - boundary[:, None]
    edge = 40.0 + 170.0 / (1.0 + np.exp(-signed / 1.8))
    return np.clip(edge, 0, 255).astype(np.uint8)


def test_parse_condition() -> None:
    assert MODULE.parse_condition("80km_6.5m_ISO1600") == (80, 6.5, 1600)


def test_inhouse_mtf50_orientation_consistency() -> None:
    vertical = synthetic_vertical_edge()
    horizontal = vertical.T
    first = MODULE.inhouse_mtf50(vertical, "vertical_edge")
    second = MODULE.inhouse_mtf50(horizontal, "horizontal_edge")
    assert np.isfinite(first)
    assert abs(first - second) < 1e-12


def test_bootstrap_is_deterministic() -> None:
    values = np.asarray([-0.1, -0.05, 0.02, 0.03], dtype=float)
    first = MODULE.bootstrap_mean_ci(values, 500, 17)
    second = MODULE.bootstrap_mean_ci(values, 500, 17)
    assert first == second
