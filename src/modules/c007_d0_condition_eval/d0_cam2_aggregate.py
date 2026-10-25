#!/usr/bin/env python3
"""Mechanical aggregator for frozen cam2 progress files.

The frozen evaluator used a pandas namedtuple attribute for a metric containing
a period (``area_fraction_at_0.55``), which is not a valid attribute name. This
separate aggregator preserves every frozen input and analysis rule and accesses
the column by mapping key instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


EVALUATOR_PATH = Path(__file__).with_name("d0_cam2_response_eval.py")
SPEC = importlib.util.spec_from_file_location("frozen_d0_cam2_response_eval", EVALUATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import frozen evaluator: {EVALUATOR_PATH}")
FROZEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FROZEN)


def aggregate(protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol, cohort = FROZEN.load_frozen_cohort(protocol_path, output_root)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(cohort.itertuples(index=False), 1):
        path = FROZEN.progress_path(output_root, index, row.condition)
        if not path.is_file():
            raise FROZEN.ProtocolError(f"Missing progress: {path}")
        FROZEN.validate_progress(path, row, protocol)
        payload = FROZEN.read_json(path)
        for event in payload["events"]:
            record = {
                "condition_cell": row.condition,
                "speed_kmh": int(row.speed_kmh),
                "distance_m": float(row.distance_m),
                "iso": int(row.iso),
                "frame": row.frame,
                "primary_condition": int(row.primary_condition),
                "reference_condition": int(row.reference_condition),
                "development_condition": int(row.development_condition),
                "arm": event["condition"],
                "seed": str(event["seed"]),
                "inference_seconds": float(event["inference_seconds"]),
            }
            record.update(
                {key: float(event["metrics"][key]) for key in FROZEN.METRICS}
            )
            records.append(record)
    frame_results = pd.DataFrame(records)
    raw = frame_results[frame_results["arm"] == "raw"].copy()
    reference = raw[raw["reference_condition"] == 1].set_index("distance_m")
    if len(reference) != 5 or reference.index.nunique() != 5:
        raise FROZEN.ProtocolError("Expected exactly five raw distance reference cells")
    raw_by_condition = raw.set_index("condition_cell")
    comparisons: list[dict[str, Any]] = []
    for _, restored_row in frame_results[frame_results["arm"] != "raw"].iterrows():
        base = raw_by_condition.loc[restored_row["condition_cell"]]
        out = restored_row.to_dict()
        for metric in FROZEN.METRICS:
            ref_value = float(reference.loc[restored_row["distance_m"], metric])
            raw_value = float(base[metric])
            restored_value = float(restored_row[metric])
            out[f"{metric}_reference"] = ref_value
            out[f"{metric}_raw"] = raw_value
            out[f"{metric}_restored_minus_raw"] = restored_value - raw_value
            out[f"{metric}_raw_abs_reference_error"] = abs(raw_value - ref_value)
            out[f"{metric}_restored_abs_reference_error"] = abs(
                restored_value - ref_value
            )
            out[f"{metric}_delta_abs_reference_error"] = (
                abs(restored_value - ref_value) - abs(raw_value - ref_value)
            )
        comparisons.append(out)
    comparison = pd.DataFrame(comparisons)
    primary = comparison[comparison["primary_condition"] == 1].copy()
    if primary["condition_cell"].nunique() != int(
        protocol["reporting"]["expected_condition_cells"]
    ):
        raise FROZEN.ProtocolError("Unexpected primary comparison cell count")
    bootstrap = protocol["reporting"]["bootstrap"]
    summaries: list[dict[str, Any]] = []
    for (arm, seed), subset in primary.groupby(["arm", "seed"], sort=True):
        for metric in FROZEN.METRICS:
            summaries.append(
                FROZEN.summary_record(subset, metric, arm, seed, bootstrap)
            )
    arm_means = (
        primary.groupby(
            [
                "condition_cell",
                "speed_kmh",
                "distance_m",
                "iso",
                "frame",
                "primary_condition",
                "arm",
            ],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    for arm, subset in arm_means.groupby("arm", sort=True):
        for metric in FROZEN.METRICS:
            summaries.append(
                FROZEN.summary_record(
                    subset, metric, arm, "three_seed_mean", bootstrap
                )
            )
    summary = pd.DataFrame(summaries)
    frame_results.to_csv(output_root / "d0_cam2_response_rows.csv", index=False)
    comparison.to_csv(output_root / "d0_cam2_reference_comparisons.csv", index=False)
    summary.to_csv(output_root / "d0_cam2_run_summary.csv", index=False)
    final = {
        "schema_version": FROZEN.SCHEMA,
        "created_utc": FROZEN.utc_now(),
        "status": "DESCRIPTIVE_RESPONSE_ONLY_NO_ACCURACY_GT",
        "counts": {
            "condition_cells": int(frame_results["condition_cell"].nunique()),
            "selected_frames": int(
                frame_results["frame"].count() // len(protocol["runs"])
            ),
            "runs_per_frame": len(protocol["runs"]),
            "primary_condition_cells": int(primary["condition_cell"].nunique()),
        },
        "detector": {
            "architecture": protocol["detector"]["architecture"],
            "operating_threshold": protocol["detector"]["operating_threshold"],
            "role": protocol["detector"]["role"],
        },
        "frozen_evaluator_sha256": FROZEN.sha256_file(EVALUATOR_PATH),
        "aggregation_code_sha256": FROZEN.sha256_file(Path(__file__).resolve()),
        "aggregation_fix": (
            "Mapping-key access for area_fraction_at_0.55; no input, endpoint, "
            "model, threshold, cohort, or numeric rule changed."
        ),
        "fixed_run_and_arm_summaries": summaries,
        "claim_boundary": protocol["claim_boundary"],
        "interpretation": (
            "These values can show whether MERIT changes or preserves an independent "
            "detector's response across actual acquisition conditions. They cannot "
            "establish detection accuracy or distinguish recovered cracks from newly "
            "induced detector responses."
        ),
    }
    FROZEN.write_json_atomic(
        output_root / "d0_cam2_final.json", final, overwrite=True
    )
    FROZEN.write_json_atomic(
        output_root / "d0_cam2_status.json",
        {
            "stage": "complete",
            "created_utc": FROZEN.utc_now(),
            "status": final["status"],
        },
        overwrite=True,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    aggregate(arguments.protocol, arguments.output_root)


if __name__ == "__main__":
    main()
