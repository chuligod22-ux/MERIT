#!/usr/bin/env python3
"""Evaluate frozen C061 motion routes on the full C006 development cohort."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL = ROOT / "protocol" / "development_evaluation_v1.json"
OUTPUT = ROOT / "runs" / "development_v1" / "selection_complete.json"
SAMPLE_OUTPUT = ROOT / "runs" / "development_v1" / "selection_sample_metrics.jsonl"
SCANNER_PATH = HERE / "c061_scan_motion.py"


def import_scanner() -> Any:
    specification = importlib.util.spec_from_file_location("c061_scanner", SCANNER_PATH)
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


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bootstrap(values: dict[str, list[float]], seed: int) -> dict[str, float | int]:
    parent_values = np.asarray(
        [float(np.mean(items)) for items in values.values()], dtype=np.float64
    )
    if len(parent_values) != 33:
        raise RuntimeError(f"Development parent coverage mismatch: {len(parent_values)} != 33")
    rng = np.random.RandomState(seed)
    draws = rng.choice(
        parent_values, size=(10000, len(parent_values)), replace=True
    ).mean(axis=1)
    return {
        "mean": float(parent_values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "parents": int(len(parent_values)),
        "bootstrap_replicates": 10000,
    }


def verify_freeze(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != "FROZEN_BEFORE_LPIPS_DEVELOPMENT_EVALUATION":
        raise RuntimeError("C061 development evaluation is not frozen")
    files = {
        "scanner": SCANNER_PATH,
        "evaluator": Path(__file__).resolve(),
        "scan_protocol": ROOT / "protocol" / "development_v1a.json",
        "scan_result": ROOT / "runs" / "development_v1" / "motion_scan.json",
    }
    for key, path in files.items():
        actual = scanner.sha256_file(path)
        expected = str(protocol["files"][key]["sha256"])
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch for {key}: {actual} != {expected}")
    scan = load_json(files["scan_result"])
    if scan.get("crops") != 50 or scan.get("parents") != 33:
        raise RuntimeError("Official development scan coverage mismatch")
    if scan.get("max_crops_argument") is not None:
        raise RuntimeError("Smoke scan cannot support final development selection")
    for severity in ("1", "2", "3"):
        frozen = protocol["selected_routes"][severity]["specification"]
        scanned = scan["results"][severity]["selected"]["specification"]
        if frozen != scanned:
            raise RuntimeError(
                f"Frozen route is not the pre-specified per-severity maximum: {severity}"
            )
    return scan


def lpips_values(
    perceptual: torch.nn.Module,
    device: torch.device,
    clean: np.ndarray,
    degraded: np.ndarray,
    restored: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    degraded_values: list[np.ndarray] = []
    restored_values: list[np.ndarray] = []
    for start in range(0, len(clean), 16):
        end = min(start + 16, len(clean))
        clean_tensor = torch.from_numpy(clean[start:end, None]).to(device).repeat(1, 3, 1, 1)
        degraded_tensor = (
            torch.from_numpy(degraded[start:end, None]).to(device).repeat(1, 3, 1, 1)
        )
        restored_tensor = (
            torch.from_numpy(restored[start:end, None]).to(device).repeat(1, 3, 1, 1)
        )
        with torch.inference_mode():
            degraded_values.append(
                perceptual(clean_tensor * 2.0 - 1.0, degraded_tensor * 2.0 - 1.0)
                .flatten()
                .float()
                .cpu()
                .numpy()
            )
            restored_values.append(
                perceptual(clean_tensor * 2.0 - 1.0, restored_tensor * 2.0 - 1.0)
                .flatten()
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(degraded_values), np.concatenate(restored_values)


def summarize_cell(
    records: list[dict[str, Any]], severity: int, gates: dict[str, Any]
) -> dict[str, Any]:
    summaries: dict[str, dict[str, float | int]] = {}
    for metric in (
        "delta_psnr_db",
        "delta_lpips",
        "delta_gradient",
        "delta_overshoot",
    ):
        by_parent: dict[str, list[float]] = defaultdict(list)
        for record in records:
            by_parent[str(record["parent_sha256"])].append(float(record[metric]))
        summaries[metric] = bootstrap(by_parent, seed=20261024 + severity)
    passed = bool(
        summaries["delta_psnr_db"]["mean"] >= float(gates["delta_psnr_mean_gte_db"])
        and summaries["delta_psnr_db"]["ci95_low"]
        > float(gates["delta_psnr_ci95_low_gt_db"])
        and summaries["delta_lpips"]["mean"] <= float(gates["delta_lpips_mean_lte"])
        and summaries["delta_lpips"]["ci95_high"]
        < float(gates["delta_lpips_ci95_high_lt"])
        and summaries["delta_gradient"]["ci95_high"]
        <= float(gates["delta_gradient_ci95_high_lte"])
        and summaries["delta_overshoot"]["ci95_high"]
        <= float(gates["delta_overshoot_ci95_high_lte"])
    )
    return {"summaries": summaries, "pass": passed}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LPIPS evaluation")
    protocol = load_json(PROTOCOL)
    scan = verify_freeze(protocol)
    bank, rows = scanner.load_development(None)
    if bank.shape[0] != 50 or len(rows) != 50:
        raise RuntimeError("Development crop coverage mismatch")
    if len({row["parent_sha256"] for row in rows}) != 33:
        raise RuntimeError("Development parent coverage mismatch")

    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    all_records: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    max_reproduction_error = {
        "delta_psnr_parent_mean_db": 0.0,
        "delta_gradient_parent_mean": 0.0,
        "delta_overshoot_parent_mean": 0.0,
    }

    for severity, parameters in enumerate(protocol["families"]["linear_motion_psf"], start=1):
        specification = protocol["selected_routes"][str(severity)]["specification"]
        kernel = common.motion_kernel(
            int(parameters["length_px"]), float(parameters["angle_deg"])
        )
        degraded = np.stack(
            [scanner.apply_motion(clean, parameters) for clean in bank]
        ).astype(np.float32)
        restored = np.stack(
            [
                scanner.blend(
                    source,
                    scanner.estimate(source, kernel, specification),
                    float(specification["alpha"]),
                )
                for source in degraded
            ]
        ).astype(np.float32)
        degraded_lpips, restored_lpips = lpips_values(
            perceptual, device, bank, degraded, restored
        )
        cell_records: list[dict[str, Any]] = []
        for index, (clean, source, output, row) in enumerate(
            zip(bank, degraded, restored, rows)
        ):
            context = scanner.prepare_overshoot_context(clean)
            record = {
                "sample_id": f"development__b{row['bank_index']}__linear_motion_psf__s{severity}",
                "bank_index": int(row["bank_index"]),
                "parent_sha256": row["parent_sha256"],
                "severity": severity,
                "parameters": parameters,
                "specification": specification,
                "delta_psnr_db": common.psnr(clean, output) - common.psnr(clean, source),
                "delta_lpips": float(restored_lpips[index] - degraded_lpips[index]),
                "delta_gradient": common.gradient_mae(clean, output)
                - common.gradient_mae(clean, source),
                "delta_overshoot": scanner.overshoot_from_context(context, output)
                - scanner.overshoot_from_context(context, source),
                "mean_absolute_change": float(np.mean(np.abs(output - source))),
            }
            cell_records.append(record)
            all_records.append(record)
        cell = summarize_cell(cell_records, severity, protocol["development_gates"])
        cell["parameters"] = parameters
        cell["specification"] = specification
        scanned = scan["results"][str(severity)]["selected"]
        reproduction = {
            "delta_psnr_parent_mean_db": abs(
                float(cell["summaries"]["delta_psnr_db"]["mean"])
                - float(scanned["delta_psnr_parent_mean_db"])
            ),
            "delta_gradient_parent_mean": abs(
                float(cell["summaries"]["delta_gradient"]["mean"])
                - float(scanned["delta_gradient_parent_mean"])
            ),
            "delta_overshoot_parent_mean": abs(
                float(cell["summaries"]["delta_overshoot"]["mean"])
                - float(scanned["delta_overshoot_parent_mean"])
            ),
        }
        cell["scan_reproduction_absolute_error"] = reproduction
        for key, value in reproduction.items():
            max_reproduction_error[key] = max(max_reproduction_error[key], value)
        cells[str(severity)] = cell
        print(
            json.dumps(
                {
                    "severity": severity,
                    "delta_psnr_db": cell["summaries"]["delta_psnr_db"],
                    "delta_lpips": cell["summaries"]["delta_lpips"],
                    "delta_gradient": cell["summaries"]["delta_gradient"],
                    "delta_overshoot": cell["summaries"]["delta_overshoot"],
                    "pass": cell["pass"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    all_pass = all(bool(cells[str(severity)]["pass"]) for severity in (1, 2, 3))
    write_jsonl_atomic(SAMPLE_OUTPUT, all_records)
    result = {
        "schema_version": "MERIT-C061-DEVELOPMENT-SELECTION-v1",
        "status": "PASS" if all_pass else "FAIL",
        "scientific_status": "C006_33_PARENT_DEVELOPMENT_SELECTION",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cells": cells,
        "all_three_severities_pass": all_pass,
        "identity_route": {
            "rule": "Exact input bypass when a supported synthetic motion kernel is not supplied.",
            "maximum_absolute_change": 0.0,
            "pass": True,
        },
        "crops": len(rows),
        "parents": len({row["parent_sha256"] for row in rows}),
        "records": len(all_records),
        "protocol_sha256": scanner.sha256_file(PROTOCOL),
        "scan_sha256": scanner.sha256_file(
            ROOT / "runs" / "development_v1" / "motion_scan.json"
        ),
        "sample_metrics_path": str(SAMPLE_OUTPUT.resolve()),
        "sample_metrics_sha256": scanner.sha256_file(SAMPLE_OUTPUT),
        "max_scan_reproduction_absolute_error": max_reproduction_error,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    scanner.write_json_atomic(OUTPUT, result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT.resolve())}))


if __name__ == "__main__":
    main()
