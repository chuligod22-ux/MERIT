#!/usr/bin/env python3
"""Freeze restoration code/assets after the official RAW holdout PASS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_restoration_mtf50_v2 as restoration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=restoration.DEFAULT_FREEZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    protocol, _, cohort, paths = restoration.validate_release_prerequisites()
    payload = {
        "schema_version": "MERIT-C057-MTF50-V2-RESTORATION-FREEZE-v1",
        "created_utc": restoration.utc_now(),
        "state": "FROZEN_AFTER_RAW_HOLDOUT_PASS_BEFORE_RESTORATION_OUTPUTS",
        "selected_candidate": {
            "localization_gaussian_sigma_px": 0.75,
            "esf_input": "unaltered pixels for RAW and every restored arm",
        },
        "expected_counts": {
            "frames": int(cohort["frame_id"].nunique()),
            "roi_identities": int(len(cohort)),
            "condition_cells": int(cohort["condition"].nunique()),
            "runs_including_raw": len(protocol["runs"]),
            "primary_condition_cells": int(protocol["primary_endpoint"]["expected_condition_cells"]),
        },
        "complete_case_gate": {
            "minimum_overall_fraction": 0.95,
            "minimum_each_distance_fraction": 0.90,
            "runs": restoration.RUN_KEYS,
        },
        "endpoint": "42 primary condition cells; distance-specific 60 km/h ISO100 RAW reference; restored-minus-RAW absolute MTF50 error; negative is improvement",
        "files": {
            label: {
                "path": restoration.relative_or_absolute(path),
                "sha256": restoration.sha256_file(path),
            }
            for label, path in paths.items()
        },
        "prohibitions": [
            "No estimator, model, checkpoint, cohort, ROI, gate, or endpoint change after this freeze.",
            "No seed selection by outcome.",
            "No claim of prospective confirmation, ISO 12233 compliance, or natural-crack efficacy.",
        ],
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if (args.output.exists() or sidecar.exists()) and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    freeze_hash = restoration.sha256_file(args.output)
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(
        f"{freeze_hash}  {args.output.name}\n", encoding="ascii"
    )
    os.replace(temporary_sidecar, sidecar)
    print(json.dumps({"restoration_execution_freeze_sha256": freeze_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
