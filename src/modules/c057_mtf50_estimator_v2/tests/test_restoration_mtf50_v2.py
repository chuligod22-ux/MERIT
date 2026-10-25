from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import run_restoration_mtf50_v2 as restoration


def _write_freeze(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{restoration.sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def test_release_prerequisites_are_bound_to_passed_raw_holdout() -> None:
    protocol, _, cohort, paths = restoration.validate_release_prerequisites()
    assert len(cohort) == 1737
    assert cohort["frame_id"].nunique() == 479
    assert cohort["condition"].nunique() == 50
    assert len(protocol["runs"]) == 7
    assert set(paths) >= {
        "restoration_runner",
        "restoration_tests",
        "mtf50_estimator",
        "raw_execution_freeze",
        "holdout_summary",
        "holdout_results",
        "original_cohort",
        "checkpoint:off_seed0",
        "checkpoint:on_seed2",
    }


def test_restoration_freeze_sidecar_tamper_fails(tmp_path: Path) -> None:
    copied = tmp_path / "freeze.json"
    copied.write_bytes(restoration.DEFAULT_FREEZE.read_bytes())
    copied.with_suffix(copied.suffix + ".sha256").write_text(
        f"{'0' * 64}  {copied.name}\n", encoding="ascii"
    )
    with pytest.raises(restoration.ProtocolError, match="sidecar mismatch"):
        restoration._read_restoration_freeze(copied)


def test_restoration_freeze_file_hash_tamper_fails(tmp_path: Path) -> None:
    payload = restoration.read_json(restoration.DEFAULT_FREEZE)
    payload["files"]["mtf50_estimator"]["sha256"] = "0" * 64
    copied = tmp_path / "freeze.json"
    _write_freeze(copied, payload)
    with pytest.raises(restoration.ProtocolError, match="hash mismatch: mtf50_estimator"):
        restoration.load_verified_environment(copied)


def _coverage_frame(missing_at_distance: float | None = None) -> pd.DataFrame:
    rows = []
    for distance in (2.5, 3.5, 4.5, 5.5, 6.5):
        for index in range(100):
            row = {"distance_m": distance}
            row.update({key: 0.1 for key in restoration.RUN_KEYS})
            if missing_at_distance == distance and index < 11:
                row["on_2"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def test_identical_complete_case_gate_passes_full_coverage() -> None:
    complete, report = restoration.complete_case_coverage(_coverage_frame())
    assert len(complete) == 500
    assert report["gate"] == "PASS"
    assert report["complete_case_fraction"] == 1.0


def test_identical_complete_case_gate_rejects_distance_below_90_percent() -> None:
    complete, report = restoration.complete_case_coverage(_coverage_frame(6.5))
    assert len(complete) == 489
    assert report["complete_case_fraction"] >= 0.95
    assert math.isclose(report["minimum_distance_complete_case_fraction"], 0.89)
    assert report["gate"] == "FAIL"


def test_bootstrap_is_deterministic_and_preserves_direction() -> None:
    values = np.asarray([-0.03, -0.02, -0.01, -0.04])
    first = restoration.bootstrap_mean_ci(values, seed=17, count=1000)
    second = restoration.bootstrap_mean_ci(values, seed=17, count=1000)
    assert first == second
    assert first[1] < 0


def test_progress_rejects_roi_identity_or_orientation_change(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "frame_id": "cell/frame.png",
                "image_sha256": "a" * 64,
                "roi_id": 1,
                "orientation": "vertical_edge",
            },
            {
                "frame_id": "cell/frame.png",
                "image_sha256": "a" * 64,
                "roi_id": 2,
                "orientation": "horizontal_edge",
            },
        ]
    )
    rois = [
        {"roi_id": 1, "orientation": "vertical_edge"},
        {"roi_id": 2, "orientation": "horizontal_edge"},
    ]
    payload = {
        "schema_version": restoration.SCHEMA,
        "restoration_freeze_sha256": "b" * 64,
        "frame_id": "cell/frame.png",
        "image_sha256": "a" * 64,
        "events": [
            {
                "condition": key.split("_", 1)[0],
                "seed": key.split("_", 1)[1],
                "rois": [dict(row) for row in rois],
            }
            for key in restoration.RUN_KEYS
        ],
    }
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restoration.validate_progress(path, frame, {}, "b" * 64)
    payload["events"][-1]["rois"][-1]["orientation"] = "vertical_edge"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(restoration.ProtocolError, match="identity/orientation"):
        restoration.validate_progress(path, frame, {}, "b" * 64)
