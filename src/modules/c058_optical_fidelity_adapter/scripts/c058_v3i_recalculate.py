"""Independent table-level recalculation for the C058 v3i development run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "artifacts/safety_router_v3i_development/all50_margin_gated_v1"
MARGIN_THRESHOLD_CY_PX = 0.001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, *, count: int = 10000, seed: int) -> list[float]:
    array = np.sort(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(count, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def recalculate() -> dict:
    summary_path = SOURCE / "summary.json"
    condition_path = SOURCE / "condition_results.csv"
    frame_path = SOURCE / "frame_results.csv"
    roi_path = SOURCE / "roi_results.csv"
    gate_path = SOURCE / "gate_table.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    condition = pd.read_csv(condition_path)
    frames = pd.read_csv(frame_path)
    roi = pd.read_csv(roi_path)
    gate = pd.read_csv(gate_path)

    deltas = condition["delta_abs_error_cy_px"].to_numpy(dtype=float)
    speed80 = condition[condition["speed_kmh"] == 80]
    reference = condition[condition["reference_condition"] == True]  # noqa: E712
    full_ci = bootstrap_ci(deltas, seed=20261029)
    speed80_ci = bootstrap_ci(speed80["delta_abs_error_cy_px"].to_numpy(dtype=float), seed=20261030)
    expected_active = (
        (gate["reference_condition"] != True)  # noqa: E712
        & (gate["optical_margin_cy_px"] > MARGIN_THRESHOLD_CY_PX)
    )
    valid_pairs = (
        roi[roi["valid"] == True]  # noqa: E712
        .pivot_table(index=["frame_id", "roi_id"], columns="arm", values="mtf50", aggfunc="first")
        .dropna()
    )
    checks = {
        "condition_rows_50_unique": bool(len(condition) == 50 and condition["condition_cell"].nunique() == 50),
        "frame_rows_479_unique": bool(len(frames) == 479 and frames["frame_id"].nunique() == 479),
        "paired_roi_count_1731": bool(len(valid_pairs) == 1731),
        "gate_rows_50_unique": bool(len(gate) == 50 and gate["condition_cell"].nunique() == 50),
        "raw_only_gate_recalculates": bool((gate["route_active"].astype(bool) == expected_active).all()),
        "reference_5_exact": bool(
            len(reference) == 5
            and (reference["route_active"] == False).all()  # noqa: E712
            and (reference["delta_abs_error_cy_px"] == 0.0).all()
            and (reference["maximum_abs_change_dn"] == 0.0).all()
        ),
        "worsened_zero": bool((condition["delta_abs_error_cy_px"] <= 0.0).all()),
        "speed80_mean_negative": bool(speed80["delta_abs_error_cy_px"].mean() < 0.0),
        "speed80_ci_upper_negative": bool(speed80_ci[1] < 0.0),
        "introduced_clipping_zero": bool(condition["maximum_introduced_clipping_fraction"].max() == 0.0),
        "summary_counts_match": bool(
            int((condition["result"] == "improved").sum()) == summary["improved_condition_cells"]
            and int((condition["result"] == "tied").sum()) == summary["tied_condition_cells"]
            and int((condition["result"] == "worsened").sum()) == summary["worsened_condition_cells"]
        ),
        "summary_means_match": bool(
            np.isclose(deltas.mean(), summary["full50_mean_delta_abs_error_cy_px"], atol=1e-15)
            and np.isclose(
                speed80["delta_abs_error_cy_px"].mean(),
                summary["speed80_mean_delta_abs_error_cy_px"],
                atol=1e-15,
            )
        ),
        "summary_ci_matches": bool(
            np.allclose(full_ci, summary["full50_condition_bootstrap_95pct_ci"], atol=1e-15)
            and np.allclose(speed80_ci, summary["speed80_condition_bootstrap_95pct_ci"], atol=1e-15)
        ),
    }
    result = {
        "schema_version": "MERIT-C058-V3I-INDEPENDENT-RECALCULATION-v1",
        "state": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recalculated": {
            "improved_condition_cells": int((condition["result"] == "improved").sum()),
            "tied_condition_cells": int((condition["result"] == "tied").sum()),
            "worsened_condition_cells": int((condition["result"] == "worsened").sum()),
            "full50_mean_delta_abs_error_cy_px": float(deltas.mean()),
            "full50_condition_bootstrap_95pct_ci": full_ci,
            "speed80_mean_delta_abs_error_cy_px": float(speed80["delta_abs_error_cy_px"].mean()),
            "speed80_condition_bootstrap_95pct_ci": speed80_ci,
            "reference_exact_condition_cells": int(
                ((condition["reference_condition"] == True) & (condition["delta_abs_error_cy_px"] == 0.0)).sum()  # noqa: E712
            ),
        },
        "source_hashes": {
            path.name: sha256(path)
            for path in (summary_path, condition_path, frame_path, roi_path, gate_path)
        },
    }
    write_json_atomic(SOURCE / "independent_recalculation.json", result)
    if result["state"] != "PASS":
        raise RuntimeError(f"v3i independent recalculation failed: {checks}")
    return result


if __name__ == "__main__":
    print(json.dumps(recalculate(), ensure_ascii=False, indent=2), flush=True)
