#!/usr/bin/env python3
"""Combine unchanged C018 cells with the confirmed C061 motion replacement."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVISION = ROOT.parent
C018 = REVISION / "c018_merit_v25_degradation_robustness"
C010 = REVISION / "c010_merit_v2_development"
OLD_CELLS = C018 / "runs" / "input_reference_audit_v1" / "cell_summary.csv"
CONFIRMATION = ROOT / "runs" / "confirmation_v1" / "RUN_COMPLETE.json"
CONFIRMATION_SAMPLES = ROOT / "runs" / "confirmation_v1" / "sample_metrics.jsonl"
OUTPUT_ROOT = ROOT / "runs" / "integrated_r1_1_v1"
OUTPUT_CSV = OUTPUT_ROOT / "cell_summary.csv"
OUTPUT_JSON = OUTPUT_ROOT / "RUN_COMPLETE.json"
SCANNER_PATH = HERE / "c061_scan_motion.py"


def import_scanner() -> Any:
    specification = importlib.util.spec_from_file_location("c061_scanner_integrate", SCANNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import scanner: {SCANNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


scanner = import_scanner()
common = scanner.common


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def maximum_pixel_changes(confirmation: dict[str, Any]) -> dict[int, float]:
    bank = np.load(C010 / "artifacts" / "confirmatory_v1" / "d0_bank_448.npy", allow_pickle=False)
    clean = bank.astype(np.float32) / 255.0
    changes: dict[int, float] = {}
    for severity, parameters in enumerate(
        [confirmation["cells"][str(i)]["parameters"] for i in (1, 2, 3)], start=1
    ):
        specification = confirmation["cells"][str(severity)]["specification"]
        kernel = common.motion_kernel(
            int(parameters["length_px"]), float(parameters["angle_deg"])
        )
        maximum = 0.0
        for image in clean:
            degraded = scanner.apply_motion(image, parameters)
            estimate = scanner.estimate(degraded, kernel, specification)
            restored = scanner.blend(degraded, estimate, float(specification["alpha"]))
            maximum = max(maximum, float(np.max(np.abs(restored - degraded))))
        changes[severity] = maximum
    return changes


def main() -> None:
    confirmation = load_json(CONFIRMATION)
    if confirmation.get("status") != "PASS" or not confirmation.get(
        "all_three_severities_pass"
    ):
        raise RuntimeError("C061 confirmation did not pass")
    with OLD_CELLS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        old_rows = list(reader)
    if len(old_rows) != 18:
        raise RuntimeError("Expected 18 C018 cells")
    old_lookup = {(row["condition"], int(row["severity"])): row for row in old_rows}
    max_changes = maximum_pixel_changes(confirmation)
    rows: list[dict[str, Any]] = []
    for row in old_rows:
        if row["condition"] != "linear_motion_psf":
            rows.append(dict(row))
            continue
        severity = int(row["severity"])
        cell = confirmation["cells"][str(severity)]
        psnr = cell["summaries"]["delta_psnr_db"]
        lpips = cell["summaries"]["delta_lpips"]
        replacement = dict(row)
        replacement.update(
            {
                "parents": confirmation["parents"],
                "records": confirmation["crops"],
                "classification": "INPUT_REFERENCE_IMPROVEMENT",
                "max_abs_candidate_minus_degraded": max_changes[severity],
                "delta_psnr_vs_input_mean": psnr["mean"],
                "delta_psnr_vs_input_ci95_low": psnr["ci95_low"],
                "delta_psnr_vs_input_ci95_high": psnr["ci95_high"],
                "delta_lpips_vs_input_mean": lpips["mean"],
                "delta_lpips_vs_input_ci95_low": lpips["ci95_low"],
                "delta_lpips_vs_input_ci95_high": lpips["ci95_high"],
                "delta_psnr_vs_base_mean": float(psnr["mean"])
                + float(row["delta_psnr_vs_base_mean"]),
                "delta_lpips_vs_base_mean": float(lpips["mean"])
                + float(row["delta_lpips_vs_base_mean"]),
            }
        )
        rows.append(replacement)
    if len(rows) != 18 or any(
        row["classification"] != "INPUT_REFERENCE_IMPROVEMENT" for row in rows
    ):
        raise RuntimeError("Integrated 18-cell classification is incomplete")
    rows.sort(key=lambda row: (row["condition"], int(row["severity"])))
    write_csv_atomic(OUTPUT_CSV, rows, fieldnames)
    weakest_psnr = min(rows, key=lambda row: float(row["delta_psnr_vs_input_mean"]))
    weakest_lpips = max(rows, key=lambda row: float(row["delta_lpips_vs_input_mean"]))
    result = {
        "schema_version": "MERIT-C061-INTEGRATED-R1-1-v1",
        "status": "PASS",
        "scientific_status": "C018_NONMOTION_RESULTS_PLUS_C061_ROUTE_SPECIFIC_MOTION_CONFIRMATION",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cells": 18,
        "input_reference_improvement_cells": 18,
        "preservation_cells": 0,
        "mixed_or_worse_cells": 0,
        "motion_replacement": {
            "cells": 3,
            "crops_per_cell": confirmation["crops"],
            "parents": confirmation["parents"],
            "development_confirmation_parent_overlap": confirmation[
                "development_confirmation_parent_overlap"
            ],
            "known_kernel_nonblind_only": True,
        },
        "weakest_psnr_cell": weakest_psnr,
        "weakest_lpips_cell": weakest_lpips,
        "source_c018_cell_summary_sha256": scanner.sha256_file(OLD_CELLS),
        "source_c061_confirmation_sha256": scanner.sha256_file(CONFIRMATION),
        "source_c061_sample_metrics_sha256": scanner.sha256_file(CONFIRMATION_SAMPLES),
        "cell_summary_path": str(OUTPUT_CSV.resolve()),
        "cell_summary_sha256": scanner.sha256_file(OUTPUT_CSV),
        "script_sha256": scanner.sha256_file(Path(__file__).resolve()),
    }
    scanner.write_json_atomic(OUTPUT_JSON, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
