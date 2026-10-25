from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_mtf_measure_v3h.py"
SPEC = importlib.util.spec_from_file_location("c058_mtf_measure_v3h_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Locator:
    @staticmethod
    def normalize_chart(image, located):
        rois = [
            {"roi_id": 1, "orientation": "vertical_edge", "x1": 0, "y1": 0, "x2": 8, "y2": 8, "source_coverage": 0.90},
            {"roi_id": 2, "orientation": "vertical_edge", "x1": 8, "y1": 0, "x2": 16, "y2": 8, "source_coverage": 1.0},
        ]
        return np.zeros((16, 16), dtype=np.uint8), rois


class Estimator:
    calls = 0

    @classmethod
    def estimate_mtf50_v2(cls, patch, orientation, localization_sigma_px):
        cls.calls += 1
        return SimpleNamespace(valid=True, reason="ok", mtf50=0.2)


def test_partial_source_roi_is_not_measured_or_filled_by_border_replication():
    Estimator.calls = 0
    results = MODULE.measure_native(
        np.zeros((16, 16), dtype=np.uint8),
        SimpleNamespace(radius=320.0),
        Locator,
        Estimator,
    )
    assert results[0]["reason"] == "partial_roi_source"
    assert results[0]["mtf50_native_cy_px"] is None
    assert results[1]["valid"] is True
    assert Estimator.calls == 1
