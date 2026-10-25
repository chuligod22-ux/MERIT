from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c009_summarize_results_schema_repair_v1 as repair


def test_crack_visible_uses_frame_roi_hit_fraction() -> None:
    visible, negative_any, burden = repair.d3_endpoint_values(
        {
            "label": "crack_visible",
            "frame_roi_hit_fraction": 0.625,
            "roi_count": 8,
            "roi_hit_count": 5,
        },
        "crack_visible",
    )
    assert visible == 0.625
    assert math.isnan(negative_any)
    assert math.isnan(burden)


def test_no_visible_uses_component_burden_fields() -> None:
    visible, negative_any, burden = repair.d3_endpoint_values(
        {
            "label": "no_visible_crack",
            "any_eligible_component": True,
            "components_per_megapixel": 1.25,
        },
        "no_visible_crack",
    )
    assert math.isnan(visible)
    assert negative_any == 1.0
    assert burden == 1.25


def test_endpoint_label_and_schema_are_fail_closed() -> None:
    with pytest.raises(repair.frozen.common.EvaluationError, match="label mismatch"):
        repair.d3_endpoint_values(
            {"label": "no_visible_crack"}, "crack_visible"
        )
    with pytest.raises(repair.frozen.common.EvaluationError, match="lacks"):
        repair.d3_endpoint_values({"label": "crack_visible"}, "crack_visible")


def test_completed_d3_artifacts_have_full_repair_coverage() -> None:
    contract = repair.frozen.common.load_contract()
    result = repair.summarize_d3(contract)
    assert result["frames_per_model"] == 974
    assert len(result["models"]) == 18
    assert result["visible_metric"] == "frame_equal_roi_hit_fraction"
    raw_visible = {
        round(value["visible_sensitivity_raw"], 15)
        for value in result["models"].values()
    }
    raw_negative = {
        round(value["no_visible_positive_rate_raw"], 15)
        for value in result["models"].values()
    }
    assert len(raw_visible) == 1
    assert len(raw_negative) == 1
    for value in result["models"].values():
        assert all(math.isfinite(float(metric)) for metric in value.values())
