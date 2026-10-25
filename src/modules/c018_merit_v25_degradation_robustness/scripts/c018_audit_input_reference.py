"""Post-confirmation audit against the unrestored degraded input.

The frozen C018 confirmation compared the selected condition-profile route with
the original C009 NAFNet base.  This audit reconstructs the same 5,292 samples
and outputs, then adds the missing candidate-versus-degraded comparator.  It
does not alter or replace the prespecified confirmation verdict.
"""

from __future__ import annotations

import csv
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
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402
import c018_evaluate_confirmation as confirmation  # noqa: E402
from c018_evaluate_development import bootstrap  # noqa: E402


AUDIT_ROOT = common.C018 / "runs" / "input_reference_audit_v1"
OUTPUT = AUDIT_ROOT / "RUN_COMPLETE.json"
SAMPLE_OUTPUT = AUDIT_ROOT / "sample_metrics.jsonl"
CELL_CSV = AUDIT_ROOT / "cell_summary.csv"
ORIGINAL_OUTPUT = confirmation.OUTPUT
ORIGINAL_SAMPLES = confirmation.SAMPLE_OUTPUT


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Verify the frozen files and the already completed confirmation outputs."""
    freeze = common.load_json(confirmation.FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_ANY_C018_CONFIRMATORY_OUTPUT":
        raise RuntimeError("C018 confirmation freeze is invalid")
    frozen_files = {
        "development_protocol": common.DEVELOPMENT_PROTOCOL,
        "physics_scan": confirmation.PHYSICS_SCAN,
        "poisson_checkpoint": confirmation.POISSON_CHECKPOINT,
        "poisson_mild_fusion": confirmation.POISSON_FUSION,
        "bank": confirmation.BANK_ROOT / "d0_bank_448.npy",
        "manifest": confirmation.BANK_ROOT / "d0_bank_manifest.csv",
        "evaluator": Path(confirmation.__file__).resolve(),
        "common": HERE / "c018_common.py",
        "development_helper": HERE / "c018_evaluate_development.py",
    }
    for key, path in frozen_files.items():
        expected = freeze["files"][key]["sha256"]
        actual = common.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch for {key}: {actual} != {expected}")

    original = common.load_json(ORIGINAL_OUTPUT)
    if original.get("status") != "PASS" or int(original.get("records", 0)) != 5292:
        raise RuntimeError("The original C018 confirmation is incomplete")
    if common.sha256_file(ORIGINAL_SAMPLES) != original["sample_metrics_sha256"]:
        raise RuntimeError("Original confirmation sample metrics hash mismatch")
    with ORIGINAL_SAMPLES.open("r", encoding="utf-8") as handle:
        original_records = [json.loads(line) for line in handle if line.strip()]
    if len(original_records) != 5292:
        raise RuntimeError("Original confirmation sample record count mismatch")
    if OUTPUT.exists() or SAMPLE_OUTPUT.exists() or CELL_CSV.exists():
        raise RuntimeError("Input-reference audit output already exists")
    return freeze, original, original_records


def metric_summary(
    records: list[dict[str, Any]], metric: str, seed: int
) -> dict[str, float]:
    by_parent: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_parent[record["parent_sha256"]].append(float(record[metric]))
    return bootstrap(by_parent, seed=seed)


def classify_cell(
    summaries: dict[str, dict[str, float]], max_abs_output_change: float
) -> str:
    """Separate true input improvement from exact preservation and mixed cells."""
    if max_abs_output_change == 0.0:
        return "EXACT_SAFE_PRESERVATION"
    psnr = summaries["delta_psnr_candidate_minus_degraded"]
    lpips_summary = summaries["delta_lpips_candidate_minus_degraded"]
    gradient = summaries["delta_gradient_candidate_minus_degraded"]
    overshoot = summaries["delta_overshoot_candidate_minus_degraded"]
    if (
        psnr["mean"] > 0.0
        and psnr["ci95_low"] > 0.0
        and lpips_summary["mean"] < 0.0
        and lpips_summary["ci95_high"] < 0.0
        and gradient["ci95_high"] <= 0.0
        and overshoot["ci95_high"] <= 0.01
    ):
        return "INPUT_REFERENCE_IMPROVEMENT"
    return "MIXED_OR_WORSE_VS_INPUT"


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    columns = [
        "condition",
        "severity",
        "parents",
        "records",
        "classification",
        "max_abs_candidate_minus_degraded",
        "delta_psnr_vs_input_mean",
        "delta_psnr_vs_input_ci95_low",
        "delta_psnr_vs_input_ci95_high",
        "delta_lpips_vs_input_mean",
        "delta_lpips_vs_input_ci95_low",
        "delta_lpips_vs_input_ci95_high",
        "delta_psnr_vs_base_mean",
        "delta_lpips_vs_base_mean",
    ]
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    freeze, original, original_records = verify_frozen_inputs()
    physics = common.load_json(confirmation.PHYSICS_SCAN)
    fusion = common.load_json(confirmation.POISSON_FUSION)
    evaluation = common.import_file(
        "c009_controlled_eval_c018_input_reference_audit",
        common.C009 / "scripts" / "c009_controlled_eval.py",
    )
    bank, rows = confirmation.load_source()
    samples = confirmation.make_samples(bank, rows, freeze, evaluation)
    if [sample["sample_id"] for sample in samples] != [
        record["sample_id"] for record in original_records
    ]:
        raise RuntimeError("Reconstructed sample order does not match frozen confirmation")

    device = torch.device("cuda")
    model, provenance = evaluation.common.build_and_load_model(
        {"kind": "wp1", "family": "nafnet", "arm": "base", "seed": 0},
        device,
    )
    all_indices = list(range(len(samples)))
    base_outputs = confirmation.infer_model(model, samples, all_indices, device)
    checkpoint = torch.load(
        confirmation.POISSON_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    poisson_indices = [
        index
        for index, sample in enumerate(samples)
        if sample["condition"] == "poisson_gaussian"
    ]
    poisson_outputs = confirmation.infer_model(model, samples, poisson_indices, device)
    candidate_outputs: dict[int, np.ndarray] = dict(poisson_outputs)
    fusion_sigma = float(fusion["selected"]["sigma"])
    fusion_alpha = float(fusion["selected"]["alpha"])
    for index in poisson_indices:
        if int(samples[index]["severity"]) != 1:
            continue
        base_image = base_outputs[index]
        correction = gaussian_filter(
            candidate_outputs[index] - base_image,
            fusion_sigma,
            mode="reflect",
        )
        candidate_outputs[index] = np.clip(
            base_image + np.float32(fusion_alpha) * correction,
            0.0,
            1.0,
        ).astype(np.float32)
    for index, sample in enumerate(samples):
        if index in candidate_outputs:
            continue
        scan_record = physics["results"][sample["condition"]][str(sample["severity"])]
        candidate_outputs[index] = common.restore_physics(
            sample["degraded"],
            sample["condition"],
            sample["parameters"],
            float(scan_record["selected_gain"]),
        )

    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(samples), 32):
        batch_samples = samples[start : start + 32]
        indices = list(range(start, start + len(batch_samples)))
        clean = np.stack([sample["clean"] for sample in batch_samples]).astype(np.float32)
        degraded = np.stack([sample["degraded"] for sample in batch_samples]).astype(
            np.float32
        )
        base = np.stack([base_outputs[index] for index in indices]).astype(np.float32)
        candidate = np.stack([candidate_outputs[index] for index in indices]).astype(
            np.float32
        )
        clean_tensor = torch.from_numpy(clean[:, None]).to(device).repeat(1, 3, 1, 1)
        degraded_tensor = (
            torch.from_numpy(degraded[:, None]).to(device).repeat(1, 3, 1, 1)
        )
        base_tensor = torch.from_numpy(base[:, None]).to(device).repeat(1, 3, 1, 1)
        candidate_tensor = (
            torch.from_numpy(candidate[:, None]).to(device).repeat(1, 3, 1, 1)
        )
        with torch.inference_mode():
            degraded_lpips = (
                perceptual(clean_tensor * 2.0 - 1.0, degraded_tensor * 2.0 - 1.0)
                .flatten()
                .float()
                .cpu()
                .numpy()
            )
            base_lpips = (
                perceptual(clean_tensor * 2.0 - 1.0, base_tensor * 2.0 - 1.0)
                .flatten()
                .float()
                .cpu()
                .numpy()
            )
            candidate_lpips = (
                perceptual(clean_tensor * 2.0 - 1.0, candidate_tensor * 2.0 - 1.0)
                .flatten()
                .float()
                .cpu()
                .numpy()
            )
        for offset, (sample, degraded_image, base_image, candidate_image) in enumerate(
            zip(batch_samples, degraded, base, candidate)
        ):
            degraded_psnr = common.psnr(sample["clean"], degraded_image)
            base_psnr = common.psnr(sample["clean"], base_image)
            candidate_psnr = common.psnr(sample["clean"], candidate_image)
            degraded_gradient = common.gradient_mae(sample["clean"], degraded_image)
            base_gradient = common.gradient_mae(sample["clean"], base_image)
            candidate_gradient = common.gradient_mae(sample["clean"], candidate_image)
            degraded_overshoot = common.edge_normal_overshoot_p95(
                sample["clean"], degraded_image
            )
            candidate_overshoot = common.edge_normal_overshoot_p95(
                sample["clean"], candidate_image
            )
            original_record = original_records[start + offset]
            record = {
                "sample_id": sample["sample_id"],
                "parent_sha256": sample["parent_sha256"],
                "condition": sample["condition"],
                "severity": sample["severity"],
                "stochastic_seed": sample["stochastic_seed"],
                "degraded_psnr_db": degraded_psnr,
                "base_psnr_db": base_psnr,
                "candidate_psnr_db": candidate_psnr,
                "degraded_lpips": float(degraded_lpips[offset]),
                "base_lpips": float(base_lpips[offset]),
                "candidate_lpips": float(candidate_lpips[offset]),
                "degraded_gradient_mae": degraded_gradient,
                "base_gradient_mae": base_gradient,
                "candidate_gradient_mae": candidate_gradient,
                "degraded_overshoot_p95": degraded_overshoot,
                "candidate_overshoot_p95": candidate_overshoot,
                "delta_psnr_candidate_minus_degraded": candidate_psnr
                - degraded_psnr,
                "delta_lpips_candidate_minus_degraded": float(
                    candidate_lpips[offset] - degraded_lpips[offset]
                ),
                "delta_gradient_candidate_minus_degraded": candidate_gradient
                - degraded_gradient,
                "delta_overshoot_candidate_minus_degraded": candidate_overshoot
                - degraded_overshoot,
                "delta_psnr_candidate_minus_base": candidate_psnr - base_psnr,
                "delta_lpips_candidate_minus_base": float(
                    candidate_lpips[offset] - base_lpips[offset]
                ),
                "delta_gradient_candidate_minus_base": candidate_gradient
                - base_gradient,
                "max_abs_candidate_minus_degraded": float(
                    np.max(np.abs(candidate_image - degraded_image))
                ),
                "original_delta_psnr": float(original_record["delta_psnr"]),
                "original_delta_lpips": float(original_record["delta_lpips"]),
            }
            records.append(record)
        print(
            json.dumps(
                {"processed": min(start + len(batch_samples), len(samples)), "total": len(samples)}
            ),
            flush=True,
        )

    max_psnr_reproduction_error = max(
        abs(record["delta_psnr_candidate_minus_base"] - record["original_delta_psnr"])
        for record in records
    )
    max_lpips_reproduction_error = max(
        abs(record["delta_lpips_candidate_minus_base"] - record["original_delta_lpips"])
        for record in records
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["condition"], int(record["severity"]))].append(record)
    metrics = (
        "delta_psnr_candidate_minus_degraded",
        "delta_lpips_candidate_minus_degraded",
        "delta_gradient_candidate_minus_degraded",
        "delta_overshoot_candidate_minus_degraded",
        "delta_psnr_candidate_minus_base",
        "delta_lpips_candidate_minus_base",
    )
    by_cell: dict[str, dict[str, Any]] = defaultdict(dict)
    csv_rows: list[dict[str, Any]] = []
    classification_counts: dict[str, int] = defaultdict(int)
    for (condition, severity), cell_records in grouped.items():
        summaries = {
            metric: metric_summary(cell_records, metric, 20261101 + severity)
            for metric in metrics
        }
        max_abs_change = max(
            float(record["max_abs_candidate_minus_degraded"])
            for record in cell_records
        )
        classification = classify_cell(summaries, max_abs_change)
        classification_counts[classification] += 1
        cell = {
            **summaries,
            "records": len(cell_records),
            "parents": len({record["parent_sha256"] for record in cell_records}),
            "max_abs_candidate_minus_degraded": max_abs_change,
            "classification": classification,
        }
        by_cell[condition][str(severity)] = cell
        psnr_input = summaries["delta_psnr_candidate_minus_degraded"]
        lpips_input = summaries["delta_lpips_candidate_minus_degraded"]
        csv_rows.append(
            {
                "condition": condition,
                "severity": severity,
                "parents": cell["parents"],
                "records": cell["records"],
                "classification": classification,
                "max_abs_candidate_minus_degraded": max_abs_change,
                "delta_psnr_vs_input_mean": psnr_input["mean"],
                "delta_psnr_vs_input_ci95_low": psnr_input["ci95_low"],
                "delta_psnr_vs_input_ci95_high": psnr_input["ci95_high"],
                "delta_lpips_vs_input_mean": lpips_input["mean"],
                "delta_lpips_vs_input_ci95_low": lpips_input["ci95_low"],
                "delta_lpips_vs_input_ci95_high": lpips_input["ci95_high"],
                "delta_psnr_vs_base_mean": summaries[
                    "delta_psnr_candidate_minus_base"
                ]["mean"],
                "delta_lpips_vs_base_mean": summaries[
                    "delta_lpips_candidate_minus_base"
                ]["mean"],
            }
        )
    csv_rows.sort(key=lambda row: (str(row["condition"]), int(row["severity"])))

    write_jsonl_atomic(SAMPLE_OUTPUT, records)
    write_csv_atomic(CELL_CSV, csv_rows)
    result = {
        "schema_version": "MERIT-C018-INPUT-REFERENCE-AUDIT-v1",
        "status": "COMPLETE",
        "scientific_status": "POST_CONFIRMATION_DESCRIPTIVE_COMPARATOR_AUDIT",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "interpretation": (
            "The original prespecified C018 verdict compared the frozen candidate with "
            "the C009 base. This post-confirmation audit adds the unrestored degraded "
            "input comparator and must not be described as a new independent confirmation."
        ),
        "records": len(records),
        "crops": len(bank),
        "parents": len({row["parent_sha256"] for row in rows}),
        "cells": len(grouped),
        "classification_counts": dict(sorted(classification_counts.items())),
        "by_family_severity": by_cell,
        "reproduction": {
            "original_result_sha256": common.sha256_file(ORIGINAL_OUTPUT),
            "original_sample_metrics_sha256": common.sha256_file(ORIGINAL_SAMPLES),
            "max_abs_delta_psnr_error": max_psnr_reproduction_error,
            "max_abs_delta_lpips_error": max_lpips_reproduction_error,
        },
        "freeze_sha256": common.sha256_file(confirmation.FREEZE),
        "sample_metrics_path": str(SAMPLE_OUTPUT.resolve()),
        "sample_metrics_sha256": common.sha256_file(SAMPLE_OUTPUT),
        "cell_summary_csv": str(CELL_CSV.resolve()),
        "cell_summary_sha256": common.sha256_file(CELL_CSV),
        "source_model": provenance,
    }
    common.write_json_atomic(OUTPUT, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification_counts": result["classification_counts"],
                "output": str(OUTPUT.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
