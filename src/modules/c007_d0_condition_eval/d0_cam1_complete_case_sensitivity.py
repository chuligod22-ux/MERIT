#!/usr/bin/env python3
"""Post-hoc complete-case sensitivity for the failed cam1 estimator gate.

This analysis is intentionally secondary.  It uses only ROI identities whose
raw and all six frozen restoration measurements are finite, preventing the
restored runs from gaining extra ROIs merely because restoration made the
in-house estimator return a crossing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_ci(values: np.ndarray, seed: int, count: int = 10000) -> list[float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    sampled = values[indices].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def summarize(condition: pd.DataFrame, arm: str, seed: str, distances: list[float]) -> dict:
    selected = condition[
        (condition["run_condition"] == arm)
        & (condition["seed"] == seed)
        & (condition["primary_condition"] == 1)
        & (condition["distance_m"].isin(distances))
    ].copy()
    delta = (selected["abs_error"] - selected["raw_abs_error"]).to_numpy(float)
    return {
        "arm": arm,
        "seed": seed,
        "distances_m": distances,
        "condition_cells": int(len(selected)),
        "mean_abs_error_cy_px": float(selected["abs_error"].mean()),
        "mean_restored_minus_raw_abs_error_cy_px": float(delta.mean()),
        "mean_delta_95pct_condition_bootstrap_ci": bootstrap_mean_ci(
            delta, 20260922 + int(seed) + (100 if arm == "off" else 200)
        ),
        "conditions_improved": int(np.sum(delta < 0)),
        "conditions_worsened": int(np.sum(delta > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--primary-final", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    primary = json.loads(args.primary_final.read_text(encoding="utf-8"))
    if primary["state"] != "COMPLETE_ESTIMATOR_GATE_FAILED":
        raise RuntimeError("This sensitivity is only defined for the frozen failed gate")
    data = pd.read_csv(args.input, dtype={"seed": str})
    data["run_key"] = data["run_condition"] + "_" + data["seed"]
    identity = [
        "frame_id", "condition_cell", "speed_kmh", "distance_m", "iso", "frame",
        "roi_id", "orientation", "development_condition", "reference_condition",
        "primary_condition",
    ]
    wide = data.pivot(index=identity, columns="run_key", values="mtf50").reset_index()
    expected = ["raw_none", "off_0", "on_0", "off_1", "on_1", "off_2", "on_2"]
    missing_columns = set(expected) - set(wide.columns)
    if missing_columns:
        raise RuntimeError(f"Missing fixed runs: {sorted(missing_columns)}")
    complete = wide.dropna(subset=expected).copy()
    complete["roi_key"] = complete["frame_id"] + "/roi" + complete["roi_id"].astype(str)

    long = complete.melt(
        id_vars=identity + ["roi_key"],
        value_vars=expected,
        var_name="run_key",
        value_name="mtf50",
    )
    long[["run_condition", "seed"]] = long["run_key"].str.split("_", n=1, expand=True)
    frame = (
        long.groupby(
            [
                "frame_id", "condition_cell", "speed_kmh", "distance_m", "iso",
                "run_condition", "seed", "development_condition", "reference_condition",
                "primary_condition",
            ],
            as_index=False,
        )
        .agg(mtf50=("mtf50", "mean"), complete_rois=("roi_key", "count"))
    )
    condition = (
        frame.groupby(
            [
                "condition_cell", "speed_kmh", "distance_m", "iso", "run_condition", "seed",
                "development_condition", "reference_condition", "primary_condition",
            ],
            as_index=False,
        )
        .agg(median_mtf50=("mtf50", "median"), frames=("frame_id", "count"))
    )
    reference = condition[
        (condition["run_condition"] == "raw") & (condition["reference_condition"] == 1)
    ][["distance_m", "median_mtf50"]].rename(columns={"median_mtf50": "reference_mtf50"})
    condition = condition.merge(reference, on="distance_m", how="left", validate="many_to_one")
    condition["abs_error"] = np.abs(condition["median_mtf50"] - condition["reference_mtf50"])
    raw_error = condition[condition["run_condition"] == "raw"][["condition_cell", "abs_error"]].rename(columns={"abs_error": "raw_abs_error"})
    condition = condition.merge(raw_error, on="condition_cell", how="left", validate="many_to_one")

    all_distances = [2.5, 3.5, 4.5, 5.5, 6.5]
    reliable_distances = [2.5, 3.5, 4.5, 5.5]
    summaries = []
    for distances in (all_distances, reliable_distances):
        for arm in ("off", "on"):
            for seed in ("0", "1", "2"):
                summaries.append(summarize(condition, arm, seed, distances))

    arm_summaries = []
    for distances in (all_distances, reliable_distances):
        for arm in ("off", "on"):
            selected = condition[
                (condition["run_condition"] == arm)
                & (condition["primary_condition"] == 1)
                & (condition["distance_m"].isin(distances))
            ]
            averaged = selected.groupby("condition_cell", as_index=False).agg(
                abs_error=("abs_error", "mean"), raw_abs_error=("raw_abs_error", "first")
            )
            delta = (averaged["abs_error"] - averaged["raw_abs_error"]).to_numpy(float)
            arm_summaries.append(
                {
                    "arm": arm,
                    "distances_m": distances,
                    "condition_cells": int(len(averaged)),
                    "mean_restored_minus_raw_abs_error_cy_px": float(delta.mean()),
                    "mean_delta_95pct_condition_bootstrap_ci": bootstrap_mean_ci(
                        delta, 20261022 + (100 if arm == "off" else 200)
                    ),
                    "conditions_improved": int(np.sum(delta < 0)),
                    "conditions_worsened": int(np.sum(delta > 0)),
                }
            )

    per_distance = []
    raw_total_by_distance = (
        data[data["run_condition"] == "raw"].groupby("distance_m").size().to_dict()
    )
    complete_by_distance = complete.groupby("distance_m").size().to_dict()
    for distance in all_distances:
        per_distance.append(
            {
                "distance_m": distance,
                "raw_roi_records": int(raw_total_by_distance.get(distance, 0)),
                "complete_case_roi_records": int(complete_by_distance.get(distance, 0)),
                "complete_case_fraction": float(
                    complete_by_distance.get(distance, 0) / raw_total_by_distance.get(distance, 1)
                ),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    complete.to_csv(args.output_root / "d0_cam1_complete_case_rois.csv", index=False)
    condition.to_csv(args.output_root / "d0_cam1_complete_case_conditions.csv", index=False)
    output = {
        "schema_version": "MERIT-D0-CAM1-COMPLETE-CASE-SENS-v1",
        "status": "POST_HOC_SECONDARY_ONLY",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "reason": "Frozen primary estimator gate failed because raw valid ROI fraction was below 0.95; this sensitivity fixes one identical complete-case ROI set across Raw and all six restoration runs.",
        "prohibition": "Cannot overturn the primary gate failure, cannot be called ISO 12233-compliant, and cannot replace D3.",
        "counts": {
            "all_roi_identities": int(len(wide)),
            "complete_case_roi_identities": int(len(complete)),
            "complete_case_fraction": float(len(complete) / len(wide)),
            "frames_with_complete_case_roi": int(complete["frame_id"].nunique()),
            "condition_cells": int(complete["condition_cell"].nunique()),
        },
        "per_distance_coverage": per_distance,
        "fixed_run_summaries": summaries,
        "three_seed_arm_summaries": arm_summaries,
        "inputs": {
            "roi_results_path": str(args.input.resolve()),
            "roi_results_sha256": sha256_file(args.input),
            "primary_final_path": str(args.primary_final.resolve()),
            "primary_final_sha256": sha256_file(args.primary_final),
            "analysis_code_sha256": sha256_file(Path(__file__)),
        },
    }
    output_path = args.output_root / "d0_cam1_complete_case_sensitivity.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
