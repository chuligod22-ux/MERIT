from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c009_summarize_results as summary
import c009_d2_external_eval as external_eval


def test_cluster_bootstrap_is_cluster_weighted() -> None:
    frame = pd.DataFrame(
        {
            "cluster": ["a", "a", "b"],
            "value": [0.0, 0.0, 2.0],
        }
    )
    result = summary.cluster_bootstrap(
        frame, "value", "cluster", seed=1, replicates=1000
    )
    assert result["clusters"] == 2
    assert result["mean"] == 1.0
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]


def test_stratified_bootstrap_equal_weights_tunnels() -> None:
    frame = pd.DataFrame(
        {
            "tunnel": ["A", "A", "B"],
            "parent": ["a1", "a2", "b1"],
            "value": [0.0, 0.0, 2.0],
        }
    )
    result = summary.cluster_bootstrap(
        frame,
        "value",
        "parent",
        strata="tunnel",
        seed=1,
        replicates=1000,
    )
    assert result["clusters"] == 3
    assert result["mean"] == 1.0
    assert np.isfinite(result["ci95_low"])


def test_holm_adjustment_is_step_down_and_monotone() -> None:
    records = {
        "a": {"mean": 1.0, "one_sided_sign_flip": {"one_sided_p": 0.01}},
        "b": {"mean": 1.0, "one_sided_sign_flip": {"one_sided_p": 0.03}},
        "c": {"mean": 1.0, "one_sided_sign_flip": {"one_sided_p": 0.04}},
    }
    summary.apply_holm(records)
    assert records["a"]["holm_adjusted_one_sided_p"] == 0.03
    assert records["b"]["holm_adjusted_one_sided_p"] == 0.06
    assert records["c"]["holm_adjusted_one_sided_p"] == 0.06
    assert records["a"]["holm_positive_inference"] is True
    assert records["b"]["holm_positive_inference"] is False


def test_matched_sensitivity_uses_highest_attaining_threshold() -> None:
    restored = SimpleNamespace(
        positive=np.asarray([0, 0, 0, 5, 5], dtype=np.int64),
        negative=np.asarray([80, 10, 5, 3, 2], dtype=np.int64),
    )
    result = external_eval.matched_sensitivity_burden(
        restored,
        {"recall_at_0_5": 0.5, "fp": 3, "negative_pixels": 100},
    )
    assert result["restored_threshold_bin"] == 4
    assert result["restored_achieved_sensitivity"] == 0.5
    assert np.isclose(
        result["delta_false_positive_pixel_rate_restored_minus_degraded"],
        -0.01,
    )


def test_global_verdict_requires_restoration_efficacy_floor() -> None:
    optical = {
        "optical_macro_pass": True,
        "restoration_efficacy_floor_pass": False,
        "identity_safety_pass": True,
        "oversharpening_safety_pass": True,
        "backbone_portability_pass": True,
    }
    assert summary.global_verdict(
        optical,
        {"prompt_reliance_pass": True},
        {"external_crack_gate_pass": True},
        {"printed_width_gate_pass": True},
    ) == "NOT_SUPPORTED"
