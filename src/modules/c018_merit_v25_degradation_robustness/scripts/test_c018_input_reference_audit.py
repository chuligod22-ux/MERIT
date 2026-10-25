"""Artifact checks for the R3-1 input-reference comparator audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "input_reference_audit_v1"


def load_result() -> dict:
    with (RUN_ROOT / "RUN_COMPLETE.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_audit_coverage_and_reproduction() -> None:
    result = load_result()
    assert result["status"] == "COMPLETE"
    assert result["scientific_status"] == "POST_CONFIRMATION_DESCRIPTIVE_COMPARATOR_AUDIT"
    assert result["records"] == 5292
    assert result["crops"] == 98
    assert result["parents"] == 68
    assert result["cells"] == 18
    assert result["reproduction"]["max_abs_delta_psnr_error"] == 0.0
    assert result["reproduction"]["max_abs_delta_lpips_error"] == 0.0


def test_audit_separates_improvement_from_motion_preservation() -> None:
    result = load_result()
    assert result["classification_counts"] == {
        "EXACT_SAFE_PRESERVATION": 3,
        "INPUT_REFERENCE_IMPROVEMENT": 15,
    }
    for severity in ("1", "2", "3"):
        motion = result["by_family_severity"]["linear_motion_psf"][severity]
        assert motion["classification"] == "EXACT_SAFE_PRESERVATION"
        assert motion["max_abs_candidate_minus_degraded"] == 0.0
        assert motion["delta_psnr_candidate_minus_degraded"]["mean"] == 0.0
        assert motion["delta_lpips_candidate_minus_degraded"]["mean"] == 0.0


def test_audit_has_no_mixed_or_worse_cell() -> None:
    with (RUN_ROOT / "cell_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert all(row["classification"] != "MIXED_OR_WORSE_VS_INPUT" for row in rows)
    assert sum(row["classification"] == "INPUT_REFERENCE_IMPROVEMENT" for row in rows) == 15
    assert sum(row["classification"] == "EXACT_SAFE_PRESERVATION" for row in rows) == 3
