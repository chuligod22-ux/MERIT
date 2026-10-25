from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "d0_cam2_response_eval.py"
SPEC = importlib.util.spec_from_file_location("d0_cam2_response_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_condition() -> None:
    assert MODULE.parse_condition("crack_d25_ISO100_V60") == (60, 2.5, 100)
    assert MODULE.parse_condition("crack_d65_ISO1600_V80") == (80, 6.5, 1600)
    with pytest.raises(MODULE.ProtocolError):
        MODULE.parse_condition("60km_2.5m_ISO100")


def test_lower_median_selection_is_content_blind() -> None:
    assert MODULE.lower_median_name(["f4.png", "f1.png", "f3.png", "f2.png"]) == "f2.png"
    assert MODULE.lower_median_name(["f3.png", "f1.png", "f2.png"]) == "f2.png"
    with pytest.raises(MODULE.ProtocolError):
        MODULE.lower_median_name([])


def test_probability_metrics() -> None:
    probability = np.asarray([[0.0, 0.54], [0.55, 1.0]], dtype=np.float32)
    result = MODULE.probability_metrics(probability, 0.55)
    assert result["mean_probability"] == pytest.approx(0.5225)
    assert result["maximum_probability"] == pytest.approx(1.0)
    assert result["area_fraction_at_0.55"] == pytest.approx(0.5)


def test_probability_metrics_rejects_nonfinite() -> None:
    with pytest.raises(MODULE.ProtocolError):
        MODULE.probability_metrics(np.asarray([[np.nan]], dtype=np.float32), 0.55)
