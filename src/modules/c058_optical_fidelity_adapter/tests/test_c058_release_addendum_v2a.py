from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import c058_release_addendum_v2a as addendum


def test_tolerance_comparison_accepts_csv_roundtrip_drift() -> None:
    left = pd.DataFrame(
        {
            "frame_id": ["f1", "f2"],
            "seed": [0, "none"],
            "value": [0.12345678901234567, 0.2],
            "valid": [True, False],
        }
    )
    right = left.copy()
    right.loc[0, "value"] += 2.220446049250313e-16
    result = addendum.compare_frames_with_tolerance(
        left, right, label="roundtrip"
    )
    assert result["pass"]
    assert result["maximum_absolute_difference"] == pytest.approx(
        2.220446049250313e-16
    )
    assert result["maximum_absolute_difference"] <= 1e-13


def test_tolerance_comparison_rejects_numeric_difference_over_limit() -> None:
    left = pd.DataFrame({"frame_id": ["f"], "value": [0.1]})
    right = pd.DataFrame({"frame_id": ["f"], "value": [0.100000000001]})
    with pytest.raises(addendum.AddendumError, match="tolerance exceeded|maximum"):
        addendum.compare_frames_with_tolerance(left, right, label="bad")


@pytest.mark.parametrize(
    "column",
    ["frame_id", "seed", "run_key", "route_id", "condition_cell", "orientation"],
)
def test_identity_columns_require_exact_match(column: str) -> None:
    base = {
        "frame_id": ["frame"],
        "seed": ["0"],
        "run_key": ["route_0"],
        "route_id": ["route"],
        "condition_cell": ["80km_2.5m_ISO100"],
        "orientation": ["vertical_edge"],
        "value": [0.1],
    }
    left = pd.DataFrame(base)
    right = pd.DataFrame(base)
    right.loc[0, column] = "changed"
    with pytest.raises(addendum.AddendumError, match="identity column"):
        addendum.compare_frames_with_tolerance(left, right, label="identity")


def test_exact_recomputation_uses_canonical_hash() -> None:
    addendum._require_exact(
        {"selection": {"route": "a", "gate": "PASS"}},
        {"selection": {"gate": "PASS", "route": "a"}},
        "selection",
    )
    with pytest.raises(addendum.AddendumError, match="Exact recomputation"):
        addendum._require_exact(
            {"selection": {"route": "a", "gate": "PASS"}},
            {"selection": {"route": "b", "gate": "PASS"}},
            "selection",
        )


def _write_frozen_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    digest = addendum.v2.sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def test_addendum_loader_detects_bound_file_tamper(tmp_path: Path) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("original", encoding="utf-8")
    payload = {
        "schema_version": addendum.ADDENDUM_SCHEMA,
        "state": addendum.ADDENDUM_STATE,
        "frozen_tolerance": {
            "relative": addendum.FLOAT_RTOL,
            "absolute": addendum.FLOAT_ATOL,
            "maximum_allowed_absolute_difference": addendum.MAX_ALLOWED_ABS_DIFF,
        },
        "bindings": {
            "bound": {
                "path": str(bound),
                "sha256": addendum.v2.sha256_file(bound),
            }
        },
    }
    path = tmp_path / "addendum.json"
    _write_frozen_json(path, payload)
    _, digest = addendum.load_addendum(path)
    assert digest == addendum.v2.sha256_file(path)
    bound.write_text("tampered", encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="binding mismatch"):
        addendum.load_addendum(path)


def test_freeze_never_overwrites_existing_original_selection(
    tmp_path: Path, monkeypatch
) -> None:
    original = tmp_path / "selected_route_v2.json"
    original.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(addendum, "ORIGINAL_V2_SELECTION", original)
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        addendum.freeze_selection(selection_path=tmp_path / "selected_route_v2a.json")


def test_frozen_thresholds_are_strictly_predeclared() -> None:
    assert addendum.FLOAT_RTOL == 1e-13
    assert addendum.FLOAT_ATOL == 1e-15
    assert addendum.MAX_ALLOWED_ABS_DIFF == 1e-13
