#!/usr/bin/env python3
"""Independently recompute the frozen restoration-v2 result from CSV files."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = BASE / "artifacts" / "restoration_v2"
FINAL = ARTIFACTS / "restoration_v2_final.json"
ROI = ARTIFACTS / "restoration_v2_roi_results.csv"
DEVELOPMENT = BASE / "artifacts" / "development_v2" / "development_roi_results.csv"
HOLDOUT = BASE / "artifacts" / "holdout_v2" / "holdout_roi_results.csv"
FREEZE = BASE / "protocol" / "mtf50_v2_restoration_execution_freeze.json"
RUN_KEYS = ["raw_none", "off_0", "on_0", "off_1", "on_1", "off_2", "on_2"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def bootstrap(values: np.ndarray, seed: int, count: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(count, values.size))
    sampled = values[indices].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    final = read_json(FINAL)
    freeze = read_json(FREEZE)
    issues: list[str] = []

    sidecar = FREEZE.with_suffix(FREEZE.suffix + ".sha256")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[0] != sha256_file(FREEZE) or fields[1] != FREEZE.name:
        issues.append("restoration freeze sidecar mismatch")
    for label, record in freeze["files"].items():
        path = resolve_path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            issues.append(f"frozen file mismatch: {label}")
    for label, record in final["inputs"]["artifacts"].items():
        path = resolve_path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            issues.append(f"final artifact mismatch: {label}")

    roi = pd.read_csv(ROI, dtype={"seed": str})
    if len(roi) != 12159:
        issues.append(f"ROI row count {len(roi)} != 12159")
    observed_runs = sorted((roi["run_condition"] + "_" + roi["seed"]).unique())
    if observed_runs != sorted(RUN_KEYS):
        issues.append("run key set changed")
    run_counts = (roi["run_condition"] + "_" + roi["seed"]).value_counts().to_dict()
    if set(run_counts.values()) != {1737}:
        issues.append(f"run counts changed: {run_counts}")

    raw_reference = pd.concat(
        [pd.read_csv(DEVELOPMENT), pd.read_csv(HOLDOUT)], ignore_index=True
    )[["frame_id", "roi_id", "v2_mtf50"]]
    raw = roi[roi["run_condition"] == "raw"][["frame_id", "roi_id", "mtf50"]]
    raw_compare = raw.merge(raw_reference, on=["frame_id", "roi_id"], validate="one_to_one")
    raw_finite = np.isfinite(raw_compare["mtf50"]) & np.isfinite(raw_compare["v2_mtf50"])
    raw_difference = np.abs(
        raw_compare.loc[raw_finite, "mtf50"].to_numpy(float)
        - raw_compare.loc[raw_finite, "v2_mtf50"].to_numpy(float)
    )
    raw_max_difference = float(raw_difference.max()) if len(raw_difference) else 0.0
    if len(raw_compare) != 1737 or raw_max_difference > 1e-12:
        issues.append("RAW values do not reproduce frozen development+holdout CSVs")

    roi["run_key"] = roi["run_condition"] + "_" + roi["seed"]
    identity = [
        "frame_id", "condition_cell", "speed_kmh", "distance_m", "iso", "frame",
        "roi_id", "orientation", "development_condition", "reference_condition",
        "primary_condition",
    ]
    wide = roi.pivot(index=identity, columns="run_key", values="mtf50").reset_index()
    complete = wide.dropna(subset=RUN_KEYS).copy()
    coverage = len(complete) / len(wide)
    per_distance = {
        str(distance): float(
            len(complete[complete["distance_m"] == distance])
            / len(wide[wide["distance_m"] == distance])
        )
        for distance in sorted(wide["distance_m"].unique())
    }
    if len(wide) != 1737 or len(complete) != 1731 or coverage < 0.95 or min(per_distance.values()) < 0.90:
        issues.append("complete-case coverage differs or fails")

    complete["roi_key"] = complete["frame_id"] + "/roi" + complete["roi_id"].astype(str)
    long = complete.melt(
        id_vars=identity + ["roi_key"],
        value_vars=RUN_KEYS,
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
        .agg(mtf50=("mtf50", "mean"))
    )
    condition = (
        frame.groupby(
            [
                "condition_cell", "speed_kmh", "distance_m", "iso", "run_condition",
                "seed", "development_condition", "reference_condition", "primary_condition",
            ],
            as_index=False,
        )
        .agg(median_mtf50=("mtf50", "median"))
    )
    reference = condition[
        (condition["run_condition"] == "raw") & (condition["reference_condition"] == 1)
    ][["distance_m", "median_mtf50"]].rename(columns={"median_mtf50": "reference_mtf50"})
    condition = condition.merge(reference, on="distance_m", validate="many_to_one")
    condition["abs_error"] = np.abs(condition["median_mtf50"] - condition["reference_mtf50"])
    raw_error = condition[condition["run_condition"] == "raw"][["condition_cell", "abs_error"]].rename(columns={"abs_error": "raw_abs_error"})
    condition = condition.merge(raw_error, on="condition_cell", validate="many_to_one")

    arm_results = []
    for arm in ("off", "on"):
        selected = condition[
            (condition["run_condition"] == arm) & (condition["primary_condition"] == 1)
        ]
        averaged = selected.groupby("condition_cell", as_index=False).agg(
            abs_error=("abs_error", "mean"), raw_abs_error=("raw_abs_error", "first")
        )
        delta = (averaged["abs_error"] - averaged["raw_abs_error"]).to_numpy(float)
        result = {
            "arm": arm,
            "condition_cells": int(len(averaged)),
            "mean_restored_minus_raw_abs_error_cy_px": float(delta.mean()),
            "mean_delta_95pct_condition_bootstrap_ci": bootstrap(
                delta, 20261120 + (100 if arm == "off" else 200)
            ),
            "conditions_improved": int(np.sum(delta < 0)),
            "conditions_worsened": int(np.sum(delta > 0)),
        }
        expected = next(row for row in final["arm_three_seed_summaries"] if row["arm"] == arm)
        if not close(result["mean_restored_minus_raw_abs_error_cy_px"], expected["mean_restored_minus_raw_abs_error_cy_px"]):
            issues.append(f"{arm} mean delta mismatch")
        if any(not close(a, b) for a, b in zip(result["mean_delta_95pct_condition_bootstrap_ci"], expected["mean_delta_95pct_condition_bootstrap_ci"])):
            issues.append(f"{arm} bootstrap CI mismatch")
        arm_results.append(result)

    manifest = read_json(ARTIFACTS / "restoration_v2_progress_manifest.json")
    if len(manifest["records"]) != 479:
        issues.append("progress manifest count changed")
    progress_ids = set()
    for record in manifest["records"]:
        path = ARTIFACTS / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            issues.append(f"progress mismatch: {record['frame_id']}")
            continue
        payload = read_json(path)
        progress_ids.add(payload["frame_id"])
        keys = [f"{event['condition']}_{event['seed']}" for event in payload["events"]]
        identities = [
            [(int(row["roi_id"]), row["orientation"]) for row in event["rois"]]
            for event in payload["events"]
        ]
        if keys != RUN_KEYS or any(item != identities[0] for item in identities[1:]):
            issues.append(f"progress run/ROI mismatch: {record['frame_id']}")
    if len(progress_ids) != 479:
        issues.append("progress frame identity count changed")

    output = {
        "schema_version": "MERIT-C057-MTF50-V2-INDEPENDENT-VERIFICATION-v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "counts": {
            "roi_rows": int(len(roi)),
            "roi_identities": int(len(wide)),
            "complete_case_roi_identities": int(len(complete)),
            "progress_files": int(len(manifest["records"])),
            "primary_condition_cells": int(condition.loc[condition["primary_condition"] == 1, "condition_cell"].nunique()),
        },
        "raw_reproduction_maximum_absolute_difference_cy_px": raw_max_difference,
        "complete_case_fraction": float(coverage),
        "per_distance_complete_case_fraction": per_distance,
        "arm_results": arm_results,
        "inputs": {
            "final_sha256": sha256_file(FINAL),
            "roi_results_sha256": sha256_file(ROI),
            "freeze_sha256": sha256_file(FREEZE),
            "verification_code_sha256": sha256_file(Path(__file__)),
        },
    }
    output_path = ARTIFACTS / "restoration_v2_independent_verification.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
