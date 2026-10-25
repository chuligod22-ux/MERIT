from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from c058_development import (
    bootstrap_condition_mean_ci,
    rank_and_select_candidates,
    summarize_candidate_metrics,
)


def _synthetic_condition_table() -> pd.DataFrame:
    rows = []
    for route_id, alpha, change in (
        ("inverse", -0.25, -0.01),
        ("raw_sharp", 0.0, -0.02),
        ("identity", 0.0, 0.0),
    ):
        for seed in (0, 1, 2):
            for distance in (2.5, 3.5, 4.5, 5.5, 6.5):
                rows.append(
                    {
                        "route_id": route_id,
                        "alpha": alpha,
                        "sigma": 0.0,
                        "sharp_gain": 0.025 if route_id == "raw_sharp" else 0.0,
                        "cap_dn": 4.0,
                        "seed": seed,
                        "condition_cell": f"80km_{distance:.1f}m_ISO100",
                        "distance_m": distance,
                        "abs_error": 0.10 + change,
                        "raw_abs_error": 0.10,
                        "valid_fraction": 1.0,
                        "reference_exact": True,
                        "introduced_clipping_fraction": 0.0,
                        "preclip_overshoot_fraction": 0.0,
                        "max_abs_change_dn": 4.0,
                    }
                )
    return pd.DataFrame(rows)


def test_condition_cluster_bootstrap_is_deterministic() -> None:
    values = np.asarray([-0.05, -0.04, -0.03, -0.02, -0.01])
    first = bootstrap_condition_mean_ci(values, seed=77, count=2000)
    second = bootstrap_condition_mean_ci(values, seed=77, count=2000)
    assert first == second
    assert first[1] < 0.0


def test_summary_averages_all_seeds_and_preserves_dependency_gate() -> None:
    summaries = summarize_candidate_metrics(
        _synthetic_condition_table(),
        bootstrap_count=2000,
        bootstrap_seed=100,
        safety={
            "minimum_valid_fraction": 0.95,
            "maximum_introduced_clipping_fraction": 0.005,
            "maximum_preclip_overshoot_fraction": 0.01,
            "maximum_abs_change_dn": 12.0,
        },
    )
    inverse = next(row for row in summaries if row["route_id"] == "inverse")
    assert inverse["condition_cells"] == 5
    assert inverse["seeds"] == 3
    assert inverse["conditions_improved"] == 5
    assert inverse["all_seed_means_nonpositive"]
    assert inverse["gate_checks"]["merit_residual_dependency"]
    assert inverse["technical_gate"] == "PASS"


def test_selection_prefers_passing_merit_dependent_route_over_raw_only_ablation() -> None:
    summaries = summarize_candidate_metrics(
        _synthetic_condition_table(),
        bootstrap_count=2000,
        bootstrap_seed=100,
        safety={
            "minimum_valid_fraction": 0.95,
            "maximum_introduced_clipping_fraction": 0.005,
            "maximum_preclip_overshoot_fraction": 0.01,
            "maximum_abs_change_dn": 12.0,
        },
    )
    selection = rank_and_select_candidates(summaries)
    assert selection["selected_route_id"] == "inverse"
    assert selection["merit_dependent_selection"]
    assert selection["gate"] == "PASS"
    raw_only = next(row for row in summaries if row["route_id"] == "raw_sharp")
    assert raw_only["technical_gate"] == "PASS"
    assert not raw_only["gate_checks"]["merit_residual_dependency"]


def test_selection_fails_closed_without_a_merit_dependent_technical_pass() -> None:
    summaries = summarize_candidate_metrics(
        _synthetic_condition_table().query("route_id != 'inverse'"),
        bootstrap_count=2000,
        bootstrap_seed=100,
        safety={
            "minimum_valid_fraction": 0.95,
            "maximum_introduced_clipping_fraction": 0.005,
            "maximum_preclip_overshoot_fraction": 0.01,
            "maximum_abs_change_dn": 12.0,
        },
    )
    selection = rank_and_select_candidates(summaries)
    assert selection["selected_route_id"] == "raw_sharp"
    assert selection["gate"] == "FAIL"
    assert selection["failure_reason"] == "NO_MERIT_DEPENDENT_CANDIDATE_PASSED"


def test_summary_requires_exact_five_conditions_and_three_seeds() -> None:
    bad = _synthetic_condition_table().query("distance_m != 6.5")
    with pytest.raises(ValueError, match="five 80 km/h ISO100"):
        summarize_candidate_metrics(
            bad,
            bootstrap_count=100,
            bootstrap_seed=1,
            safety={
                "minimum_valid_fraction": 0.95,
                "maximum_introduced_clipping_fraction": 0.005,
                "maximum_preclip_overshoot_fraction": 0.01,
                "maximum_abs_change_dn": 12.0,
            },
        )
