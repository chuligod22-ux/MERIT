"""Independent recalculation for the completed C058 v3h all-50 analysis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "artifacts/safety_router_v3h_development/all50_efficacy_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, *, count: int = 10000, seed: int = 20261022) -> list[float]:
    array = np.sort(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(count, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def subgroup_record(name: str, frame: pd.DataFrame, seed: int) -> dict:
    delta = frame["delta_abs_error_cy_px"].to_numpy(dtype=float)
    return {
        "subgroup": name,
        "condition_cells": int(len(frame)),
        "improved": int((frame["result"] == "improved").sum()),
        "tied": int((frame["result"] == "tied").sum()),
        "worsened": int((frame["result"] == "worsened").sum()),
        "mean_delta_abs_error_cy_px": float(delta.mean()),
        "condition_bootstrap_95pct_ci_low": bootstrap_ci(delta, seed=seed)[0],
        "condition_bootstrap_95pct_ci_high": bootstrap_ci(delta, seed=seed)[1],
    }


def recalculate() -> dict:
    summary_path = SOURCE / "summary.json"
    condition_path = SOURCE / "condition_results.csv"
    frame_path = SOURCE / "frame_results.csv"
    roi_path = SOURCE / "roi_results.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    condition = pd.read_csv(condition_path)
    frames = pd.read_csv(frame_path)
    roi = pd.read_csv(roi_path)

    full = subgroup_record("all_50", condition, 20261022)
    nonreference = subgroup_record(
        "nonreference_45", condition[condition["reference_condition"] != True], 20261023  # noqa: E712
    )
    reference = subgroup_record(
        "reference_preservation_5", condition[condition["reference_condition"] == True], 20261024  # noqa: E712
    )
    speed60 = subgroup_record("speed_60_25", condition[condition["speed_kmh"] == 60], 20261025)
    speed80 = subgroup_record("speed_80_25", condition[condition["speed_kmh"] == 80], 20261026)
    near = subgroup_record("distance_2.5_to_4.5_30", condition[condition["distance_m"] <= 4.5], 20261027)
    far = subgroup_record("distance_5.5_to_6.5_20", condition[condition["distance_m"] >= 5.5], 20261028)
    subgroups = pd.DataFrame([full, nonreference, reference, speed60, speed80, near, far])

    valid_pairs = (
        roi[roi["valid"] == True]  # noqa: E712
        .pivot_table(index=["frame_id", "roi_id"], columns="arm", values="mtf50", aggfunc="first")
        .dropna()
    )
    checks = {
        "condition_rows_50": bool(len(condition) == 50),
        "condition_cells_unique": bool(condition["condition_cell"].nunique() == 50),
        "condition_results_complete": bool(condition["result"].isin(["improved", "tied", "worsened"]).all()),
        "frame_rows_479": bool(len(frames) == 479 and frames["frame_id"].nunique() == 479),
        "roi_rows_two_arms": bool(len(roi) == 1737 * 2),
        "paired_roi_count_1731": bool(len(valid_pairs) == 1731),
        "classification_reconciles_50": bool(full["improved"] + full["tied"] + full["worsened"] == 50),
        "summary_counts_match": bool(
            full["improved"] == summary["improved_condition_cells"]
            and full["tied"] == summary["tied_condition_cells"]
            and full["worsened"] == summary["worsened_condition_cells"]
        ),
        "summary_mean_matches": bool(
            np.isclose(full["mean_delta_abs_error_cy_px"], summary["full50_mean_delta_abs_error_cy_px"], atol=1e-15)
        ),
        "summary_ci_matches": bool(
            np.allclose(
                [full["condition_bootstrap_95pct_ci_low"], full["condition_bootstrap_95pct_ci_high"]],
                summary["full50_condition_bootstrap_95pct_ci"],
                atol=1e-15,
            )
        ),
        "all_50_conditions_used_in_full_estimate": bool(full["condition_cells"] == 50),
        "five_reference_conditions_present": bool(reference["condition_cells"] == 5),
        "far_distance_conditions_20": bool(far["condition_cells"] == 20),
    }
    result = {
        "schema_version": "MERIT-C058-V3H-ALL50-INDEPENDENT-RECALCULATION-v1",
        "state": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "subgroups": subgroups.to_dict(orient="records"),
        "source_hashes": {
            path.name: sha256(path) for path in (summary_path, condition_path, frame_path, roi_path)
        },
    }
    subgroups.to_csv(SOURCE / "subgroup_summary.csv", index=False)
    write_json_atomic(SOURCE / "independent_recalculation.json", result)
    if result["state"] != "PASS":
        raise RuntimeError(f"independent recalculation failed: {checks}")
    return result


if __name__ == "__main__":
    print(json.dumps(recalculate(), ensure_ascii=False, indent=2), flush=True)
