from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest


BASE = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ESTIMATOR = load_module("c057_mtf50_estimator", BASE / "mtf50_estimator.py")
sys.path.insert(0, str(BASE))
RUNNER = load_module("c057_run_mtf50_v2", BASE / "run_mtf50_v2.py")


def synthetic_vertical_edge(height: int = 256, width: int = 192) -> np.ndarray:
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    boundary = width / 2 + 0.08 * (rows - height / 2)
    signed = cols[None, :] - boundary[:, None]
    edge = 40.0 + 170.0 / (1.0 + np.exp(-signed / 1.8))
    return np.clip(edge, 0, 255).astype(np.uint8)


def test_v2_orientation_and_polarity_symmetry() -> None:
    vertical = synthetic_vertical_edge()
    variants = (
        (vertical, "vertical_edge"),
        (vertical.T, "horizontal_edge"),
        (255 - vertical, "vertical_edge"),
        ((255 - vertical).T, "horizontal_edge"),
    )
    values = [ESTIMATOR.estimate_mtf50_v2(image, orientation) for image, orientation in variants]
    assert all(result.valid for result in values)
    assert values[0].polarity == values[1].polarity == "positive"
    assert values[2].polarity == values[3].polarity == "negative"
    assert max(result.mtf50 for result in values) - min(result.mtf50 for result in values) < 1e-12


def test_v2_preserves_nan_failure() -> None:
    flat = np.full((128, 128), 127, dtype=np.uint8)
    result = ESTIMATOR.estimate_mtf50_v2(flat, "vertical_edge")
    assert not result.valid
    assert math.isnan(result.mtf50)
    assert result.reason != "ok"


def test_localizer_smoothing_does_not_change_esf_pixels(monkeypatch) -> None:
    image = synthetic_vertical_edge()
    expected = image.astype(np.float64)
    captured: dict[str, np.ndarray] = {}
    original = ESTIMATOR._compute_mtf50_from_line

    def capture(data: np.ndarray, **kwargs):
        captured["data"] = data.copy()
        return original(data, **kwargs)

    monkeypatch.setattr(ESTIMATOR, "_compute_mtf50_from_line", capture)
    result = ESTIMATOR.estimate_mtf50_v2(image, "vertical_edge")
    assert result.valid
    assert np.array_equal(captured["data"], expected)


def test_v1_frozen_regression_reproduces_229_invalid() -> None:
    frozen = (
        BASE.parent
        / "c007_d0_condition_eval"
        / "artifacts"
        / "cam1_v1"
        / "d0_cam1_roi_results.csv"
    )
    assert RUNNER.sha256_file(frozen) == (
        "53cdea6e0d0ccb412541d1645351888060991463eb9f0a30081fd7687f91ce90"
    )
    rows = pd.read_csv(frozen)
    raw = rows[rows["run_condition"] == "raw"]
    assert len(raw) == 1737
    assert int(raw["mtf50"].isna().sum()) == 229


def test_development_artifacts_reproduce_frozen_result() -> None:
    summary_path = BASE / "artifacts" / "development_v2" / "development_summary.json"
    csv_path = BASE / "artifacts" / "development_v2" / "development_roi_results.csv"
    summary = RUNNER.read_json(summary_path)
    rows = pd.read_csv(csv_path)
    metrics = summary["metrics"]
    assert len(rows) == 335
    assert int(rows["v2_valid"].sum()) == 333
    assert metrics["valid_roi_count"] == 333
    assert math.isclose(metrics["valid_roi_fraction"], 333 / 335, abs_tol=1e-15)
    assert math.isclose(metrics["minimum_distance_valid_fraction"], 32 / 33, abs_tol=1e-15)
    assert math.isclose(metrics["roi_spearman_vs_imatest"], 0.936184, abs_tol=5e-7)
    assert math.isclose(metrics["frame_spearman_vs_imatest_mean"], 0.902616, abs_tol=5e-7)
    assert math.isclose(metrics["roi_mean_absolute_error_cy_px"], 0.0077935, abs_tol=5e-8)
    assert math.isclose(metrics["common_roi_mae_change_vs_v1_cy_px"], -0.0000704, abs_tol=5e-8)
    assert math.isclose(metrics["common_roi_spearman_change_vs_v1"], -0.000557, abs_tol=5e-7)
    assert summary["gate"] == "PASS"
    assert all(summary["gate_checks"].values())
    execution_freeze_hash = RUNNER.sha256_file(RUNNER.DEFAULT_EXECUTION_FREEZE)
    assert summary["inputs"]["execution_freeze_sha256"] == execution_freeze_hash
    assert summary["partition_validation"]["union"] == {
        "roi_records": 1737,
        "condition_cells": 50,
        "intersection_roi_records": 0,
    }


def _write_freeze_copy(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{RUNNER.sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def _base_inputs():
    return RUNNER.load_contract_cohort(RUNNER.DEFAULT_CONTRACT, RUNNER.DEFAULT_FREEZE)


def test_execution_freeze_hash_tamper_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "execution_freeze.json"
    copied.write_bytes(RUNNER.DEFAULT_EXECUTION_FREEZE.read_bytes())
    copied.with_suffix(copied.suffix + ".sha256").write_text(
        f"{'0' * 64}  {copied.name}\n", encoding="ascii"
    )
    contract, frozen, _, _ = _base_inputs()
    with pytest.raises(RUNNER.ProtocolError, match="sidecar mismatch"):
        RUNNER.verify_execution_freeze(
            copied,
            RUNNER.DEFAULT_CONTRACT,
            RUNNER.DEFAULT_FREEZE,
            contract,
            frozen,
        )


def test_execution_freeze_content_tamper_fails_closed(tmp_path: Path) -> None:
    payload = RUNNER.read_json(RUNNER.DEFAULT_EXECUTION_FREEZE)
    payload["selected_candidate"]["localization_gaussian_sigma_px"] = 1.0
    copied = tmp_path / "execution_freeze.json"
    _write_freeze_copy(copied, payload)
    contract, frozen, _, _ = _base_inputs()
    with pytest.raises(RUNNER.ProtocolError, match="selected sigma"):
        RUNNER.verify_execution_freeze(
            copied,
            RUNNER.DEFAULT_CONTRACT,
            RUNNER.DEFAULT_FREEZE,
            contract,
            frozen,
        )


def test_execution_freeze_source_hash_tamper_fails_closed(tmp_path: Path) -> None:
    payload = RUNNER.read_json(RUNNER.DEFAULT_EXECUTION_FREEZE)
    payload["files"]["estimator_source"]["sha256"] = "0" * 64
    copied = tmp_path / "execution_freeze.json"
    _write_freeze_copy(copied, payload)
    contract, frozen, _, _ = _base_inputs()
    with pytest.raises(RUNNER.ProtocolError, match="hash mismatch: estimator_source"):
        RUNNER.verify_execution_freeze(
            copied,
            RUNNER.DEFAULT_CONTRACT,
            RUNNER.DEFAULT_FREEZE,
            contract,
            frozen,
        )


def test_partition_contamination_fails_closed() -> None:
    contract, _, cohort, report = _base_inputs()
    assert report["development"]["roi_records"] == 335
    assert report["holdout"]["roi_records"] == 1402
    contaminated = cohort.copy()
    index = contaminated.index[contaminated["iso"] == 100][0]
    contaminated.loc[index, "iso"] = 200
    with pytest.raises(RUNNER.ProtocolError, match="development ROI count"):
        RUNNER.validate_partitions(contaminated, contract)
