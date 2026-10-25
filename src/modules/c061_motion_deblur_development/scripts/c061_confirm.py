#!/usr/bin/env python3
"""One-shot parent-disjoint confirmation of the frozen C061 motion route."""

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
REVISION = ROOT.parent
C006 = REVISION / "c006_execution"
C010 = REVISION / "c010_merit_v2_development"
SCANNER_PATH = HERE / "c061_scan_motion.py"
FREEZE = ROOT / "protocol" / "confirmation_freeze_v1.json"
BANK_ROOT = C010 / "artifacts" / "confirmatory_v1"
RUN_ROOT = ROOT / "runs" / "confirmation_v1"
OUTPUT = RUN_ROOT / "RUN_COMPLETE.json"
SAMPLE_OUTPUT = RUN_ROOT / "sample_metrics.jsonl"


def import_scanner() -> Any:
    specification = importlib.util.spec_from_file_location("c061_scanner_confirm", SCANNER_PATH)
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


def write_jsonl_exclusive(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if path.exists():
            raise RuntimeError(f"One-shot output already exists: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bootstrap(values: dict[str, list[float]], seed: int) -> dict[str, float | int]:
    parent_values = np.asarray(
        [float(np.mean(items)) for items in values.values()], dtype=np.float64
    )
    if len(parent_values) != 68:
        raise RuntimeError(f"Confirmation parent coverage mismatch: {len(parent_values)} != 68")
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


def verify_freeze() -> dict[str, Any]:
    freeze = load_json(FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_ANY_C061_CONFIRMATION_OUTPUT":
        raise RuntimeError("C061 confirmation is not frozen")
    files = {
        "scanner": SCANNER_PATH,
        "confirmation_evaluator": Path(__file__).resolve(),
        "development_scan_protocol": ROOT / "protocol" / "development_v1a.json",
        "development_evaluation_protocol": ROOT
        / "protocol"
        / "development_evaluation_v1.json",
        "development_scan": ROOT / "runs" / "development_v1" / "motion_scan.json",
        "development_selection": ROOT
        / "runs"
        / "development_v1"
        / "selection_complete.json",
        "development_sample_metrics": ROOT
        / "runs"
        / "development_v1"
        / "selection_sample_metrics.jsonl",
        "confirmation_bank": BANK_ROOT / "d0_bank_448.npy",
        "confirmation_manifest": BANK_ROOT / "d0_bank_manifest.csv",
    }
    for key, path in files.items():
        actual = scanner.sha256_file(path)
        expected = str(freeze["files"][key]["sha256"])
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch for {key}: {actual} != {expected}")
    development = load_json(files["development_selection"])
    if development.get("status") != "PASS":
        raise RuntimeError("Development selection did not pass")
    for severity in ("1", "2", "3"):
        if not development["cells"][severity]["pass"]:
            raise RuntimeError(f"Development severity did not pass: {severity}")
        if (
            freeze["selected_routes"][severity]["specification"]
            != development["cells"][severity]["specification"]
        ):
            raise RuntimeError(f"Confirmation route differs from development: {severity}")
    if OUTPUT.exists() or SAMPLE_OUTPUT.exists():
        raise RuntimeError("C061 one-shot confirmation output already exists")
    return freeze


def load_confirmation() -> tuple[np.ndarray, list[dict[str, str]]]:
    bank = np.load(BANK_ROOT / "d0_bank_448.npy", allow_pickle=False)
    with (BANK_ROOT / "d0_bank_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(bank) != 98 or len(rows) != 98:
        raise RuntimeError("Frozen confirmation source must contain 98 crops")
    if len({row["parent_sha256"] for row in rows}) != 68:
        raise RuntimeError("Frozen confirmation source must contain 68 parents")
    with (C006 / "artifacts" / "d0_bank_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        development_rows = list(csv.DictReader(handle))
    development_parents = {
        row["parent_sha256"]
        for row in development_rows
        if row["split"] == "synthetic_holdout"
    }
    confirmation_parents = {row["parent_sha256"] for row in rows}
    overlap = development_parents & confirmation_parents
    if overlap:
        raise RuntimeError(f"Development-confirmation parent overlap: {len(overlap)}")
    return bank.astype(np.float32) / 255.0, rows


def restore_route(
    source: np.ndarray,
    parameters: dict[str, Any] | None,
    specification: dict[str, Any] | None,
) -> np.ndarray:
    if parameters is None or specification is None:
        return source.copy()
    kernel = common.motion_kernel(
        int(parameters["length_px"]), float(parameters["angle_deg"])
    )
    estimate = scanner.estimate(source, kernel, specification)
    return scanner.blend(source, estimate, float(specification["alpha"]))


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
        summaries[metric] = bootstrap(by_parent, seed=20261124 + severity)
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
    freeze = verify_freeze()
    bank, rows = load_confirmation()
    identity_outputs = np.stack(
        [restore_route(source, None, None) for source in bank]
    ).astype(np.float32)
    identity_maximum_change = float(np.max(np.abs(identity_outputs - bank)))
    identity_pass = identity_maximum_change == 0.0

    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    all_records: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    for severity, parameters in enumerate(freeze["families"]["linear_motion_psf"], start=1):
        specification = freeze["selected_routes"][str(severity)]["specification"]
        degraded = np.stack(
            [scanner.apply_motion(clean, parameters) for clean in bank]
        ).astype(np.float32)
        restored = np.stack(
            [restore_route(source, parameters, specification) for source in degraded]
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
                "sample_id": f"confirmation__b{row['bank_index']}__linear_motion_psf__s{severity}",
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
        cell = summarize_cell(cell_records, severity, freeze["confirmation_gates"])
        cell["parameters"] = parameters
        cell["specification"] = specification
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

    all_pass = bool(
        identity_pass
        and all(bool(cells[str(severity)]["pass"]) for severity in (1, 2, 3))
    )
    write_jsonl_exclusive(SAMPLE_OUTPUT, all_records)
    result = {
        "schema_version": "MERIT-C061-CONFIRMATION-v1",
        "status": "PASS" if all_pass else "FAIL",
        "scientific_status": "ROUTE_SPECIFIC_PARENT_DISJOINT_HELD_OUT_CONFIRMATION",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cells": cells,
        "all_three_severities_pass": all_pass,
        "identity_route": {
            "maximum_absolute_change": identity_maximum_change,
            "pass": identity_pass,
        },
        "crops": len(rows),
        "parents": len({row["parent_sha256"] for row in rows}),
        "records": len(all_records),
        "development_confirmation_parent_overlap": 0,
        "freeze_sha256": scanner.sha256_file(FREEZE),
        "sample_metrics_path": str(SAMPLE_OUTPUT.resolve()),
        "sample_metrics_sha256": scanner.sha256_file(SAMPLE_OUTPUT),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    scanner.write_json_atomic(OUTPUT, result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT.resolve())}))


if __name__ == "__main__":
    main()
