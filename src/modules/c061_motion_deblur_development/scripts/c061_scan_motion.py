#!/usr/bin/env python3
"""Scan bounded non-blind motion-deblurring routes on C006 development parents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVISION = ROOT.parent
C018 = REVISION / "c018_merit_v25_degradation_robustness"
C006 = REVISION / "c006_execution"
C009 = REVISION / "c009_integrated_experiment"
PROTOCOL = ROOT / "protocol" / "development_v1a.json"
OUTPUT = ROOT / "runs" / "development_v1" / "motion_scan.json"

if str(C018 / "scripts") not in sys.path:
    sys.path.insert(0, str(C018 / "scripts"))

import c018_common as common  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_development(max_crops: int | None) -> tuple[np.ndarray, list[dict[str, str]]]:
    bank_path = C006 / "artifacts" / "d0_bank_448.npy"
    manifest_path = C006 / "artifacts" / "d0_bank_manifest.csv"
    bank_all = np.load(bank_path, allow_pickle=False)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows_all = list(csv.DictReader(handle))
    selected = [
        (bank_all[int(row["bank_index"])], row)
        for row in rows_all
        if row["split"] == "synthetic_holdout"
    ]
    if max_crops is not None:
        selected = selected[:max_crops]
    bank = np.stack([item[0] for item in selected]).astype(np.float32) / 255.0
    rows = [item[1] for item in selected]
    return bank, rows


def apply_motion(clean: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    kernel = common.motion_kernel(
        int(parameters["length_px"]), float(parameters["angle_deg"])
    )
    return cv2.filter2D(
        clean, -1, kernel, borderType=cv2.BORDER_REFLECT_101
    ).astype(np.float32)


def forward(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(
        image.astype(np.float32, copy=False),
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)


def landweber(
    degraded: np.ndarray,
    kernel: np.ndarray,
    iterations: int,
    step: float,
) -> np.ndarray:
    estimate = degraded.astype(np.float32, copy=True)
    adjoint = np.flip(kernel, axis=(0, 1)).copy()
    for _ in range(iterations):
        residual = degraded - forward(estimate, kernel)
        estimate += np.float32(step) * forward(residual, adjoint)
        np.clip(estimate, 0.0, 1.0, out=estimate)
    return estimate


def richardson_lucy(
    degraded: np.ndarray,
    kernel: np.ndarray,
    iterations: int,
) -> np.ndarray:
    estimate = np.maximum(degraded.astype(np.float32, copy=True), np.float32(1e-5))
    adjoint = np.flip(kernel, axis=(0, 1)).copy()
    normalization = np.maximum(forward(np.ones_like(estimate), adjoint), np.float32(1e-5))
    for _ in range(iterations):
        predicted = np.maximum(forward(estimate, kernel), np.float32(1e-5))
        ratio = degraded / predicted
        estimate *= forward(ratio, adjoint) / normalization
        np.clip(estimate, 0.0, 1.0, out=estimate)
    return estimate.astype(np.float32)


def psf_to_otf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape, dtype=np.float32)
    height, width = kernel.shape
    padded[:height, :width] = kernel
    padded = np.roll(padded, -(height // 2), axis=0)
    padded = np.roll(padded, -(width // 2), axis=1)
    return np.fft.rfft2(padded)


def wiener(
    degraded: np.ndarray,
    kernel: np.ndarray,
    regularization: float,
) -> np.ndarray:
    radius = max(24, int(kernel.shape[0]) * 3)
    padded = np.pad(degraded, radius, mode="reflect").astype(np.float32)
    transfer = psf_to_otf(kernel, padded.shape)
    spectrum = np.fft.rfft2(padded)
    estimate = np.fft.irfft2(
        np.conj(transfer) * spectrum / (np.abs(transfer) ** 2 + regularization),
        s=padded.shape,
    ).real
    estimate = estimate[radius:-radius, radius:-radius]
    return np.clip(estimate, 0.0, 1.0).astype(np.float32)


def blend(degraded: np.ndarray, estimate: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(
        degraded + np.float32(alpha) * (estimate - degraded), 0.0, 1.0
    ).astype(np.float32)


def prepare_overshoot_context(clean: np.ndarray) -> dict[str, np.ndarray]:
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 1e-6]
    if positive.size == 0:
        return {"mask": np.zeros_like(clean, dtype=bool)}
    threshold = float(np.percentile(positive, 75.0))
    mask = magnitude >= threshold
    yy, xx = np.indices(clean.shape, dtype=np.float32)
    nx = gx / np.maximum(magnitude, 1e-6)
    ny = gy / np.maximum(magnitude, 1e-6)
    plus_x, plus_y = xx + nx * 1.5, yy + ny * 1.5
    minus_x, minus_y = xx - nx * 1.5, yy - ny * 1.5
    clean_plus = cv2.remap(
        clean, plus_x, plus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    clean_minus = cv2.remap(
        clean, minus_x, minus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return {
        "mask": mask,
        "plus_x": plus_x,
        "plus_y": plus_y,
        "minus_x": minus_x,
        "minus_y": minus_y,
        "lower": np.minimum(clean_plus, clean_minus),
        "upper": np.maximum(clean_plus, clean_minus),
    }


def overshoot_from_context(
    context: dict[str, np.ndarray], candidate: np.ndarray
) -> float:
    mask = context["mask"]
    if not np.any(mask):
        return 0.0
    candidate_plus = cv2.remap(
        candidate,
        context["plus_x"],
        context["plus_y"],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    candidate_minus = cv2.remap(
        candidate,
        context["minus_x"],
        context["minus_y"],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    lower = context["lower"]
    upper = context["upper"]
    overshoot = np.maximum.reduce(
        [
            candidate_plus - upper,
            lower - candidate_plus,
            candidate_minus - upper,
            lower - candidate_minus,
            np.zeros_like(candidate),
        ]
    )
    return float(np.percentile(overshoot[mask], 95.0))


def candidate_specs(protocol: dict[str, Any]) -> Iterable[dict[str, Any]]:
    grid = protocol["candidate_grid"]
    for regularization in grid["wiener_regularization"]:
        for alpha in grid["blend_alpha"]:
            yield {
                "algorithm": "wiener",
                "regularization": float(regularization),
                "alpha": float(alpha),
            }
    for iterations in grid["landweber_iterations"]:
        for step in grid["landweber_steps"]:
            for alpha in grid["blend_alpha"]:
                yield {
                    "algorithm": "landweber",
                    "iterations": int(iterations),
                    "step": float(step),
                    "alpha": float(alpha),
                }
    for iterations in grid["richardson_lucy_iterations"]:
        for alpha in grid["blend_alpha"]:
            yield {
                "algorithm": "richardson_lucy",
                "iterations": int(iterations),
                "alpha": float(alpha),
            }


def spec_id(specification: dict[str, Any]) -> str:
    return "__".join(f"{key}={value}" for key, value in specification.items())


def estimate(
    degraded: np.ndarray,
    kernel: np.ndarray,
    specification: dict[str, Any],
) -> np.ndarray:
    algorithm = specification["algorithm"]
    if algorithm == "wiener":
        estimate = wiener(degraded, kernel, float(specification["regularization"]))
    elif algorithm == "landweber":
        estimate = landweber(
            degraded,
            kernel,
            int(specification["iterations"]),
            float(specification["step"]),
        )
    elif algorithm == "richardson_lucy":
        estimate = richardson_lucy(
            degraded, kernel, int(specification["iterations"])
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return estimate


def base_specification(specification: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in specification.items() if key != "alpha"}


def parent_mean(records: list[dict[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["parent_sha256"])].append(float(record[key]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "delta_psnr_parent_mean_db": parent_mean(records, "delta_psnr_db"),
        "delta_gradient_parent_mean": parent_mean(records, "delta_gradient"),
        "delta_overshoot_parent_mean": parent_mean(records, "delta_overshoot"),
        "mean_absolute_change": parent_mean(records, "mean_absolute_change"),
        "parents": len({record["parent_sha256"] for record in records}),
        "crops": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-crops", type=int)
    arguments = parser.parse_args()
    protocol = load_json(PROTOCOL)
    if protocol.get("status") != "FROZEN_BEFORE_FULL_33_PARENT_SCAN":
        raise RuntimeError("C061 development protocol is not frozen")
    expected_script = str(protocol.get("script_sha256", ""))
    actual_script = sha256_file(Path(__file__).resolve())
    if actual_script != expected_script:
        raise RuntimeError(
            f"C061 scan script hash mismatch: {actual_script} != {expected_script}"
        )
    bank, rows = load_development(arguments.max_crops)
    severities = protocol["families"]["linear_motion_psf"]
    specifications = list(candidate_specs(protocol))
    gates = protocol["development_gates"]
    results: dict[str, Any] = {}

    for severity, parameters in enumerate(severities, start=1):
        kernel = common.motion_kernel(
            int(parameters["length_px"]), float(parameters["angle_deg"])
        )
        samples = [
            {
                "clean": clean,
                "degraded": apply_motion(clean, parameters),
                "parent_sha256": row["parent_sha256"],
                "overshoot_context": prepare_overshoot_context(clean),
            }
            for clean, row in zip(bank, rows)
        ]
        for sample in samples:
            sample["degraded_psnr_db"] = common.psnr(
                sample["clean"], sample["degraded"]
            )
            sample["degraded_gradient"] = common.gradient_mae(
                sample["clean"], sample["degraded"]
            )
            sample["degraded_overshoot"] = overshoot_from_context(
                sample["overshoot_context"], sample["degraded"]
            )
        grid_results: dict[str, Any] = {}
        cached_base_id = ""
        cached_estimates: list[np.ndarray] = []
        for index, specification in enumerate(specifications, start=1):
            base = base_specification(specification)
            base_id = spec_id(base)
            if base_id != cached_base_id:
                cached_estimates = [
                    estimate(sample["degraded"], kernel, base) for sample in samples
                ]
                cached_base_id = base_id
            records: list[dict[str, Any]] = []
            for sample, restored_estimate in zip(samples, cached_estimates):
                output = blend(
                    sample["degraded"],
                    restored_estimate,
                    float(specification["alpha"]),
                )
                records.append(
                    {
                        "parent_sha256": sample["parent_sha256"],
                        "delta_psnr_db": common.psnr(sample["clean"], output)
                        - float(sample["degraded_psnr_db"]),
                        "delta_gradient": common.gradient_mae(sample["clean"], output)
                        - float(sample["degraded_gradient"]),
                        "delta_overshoot": overshoot_from_context(
                            sample["overshoot_context"], output
                        )
                        - float(sample["degraded_overshoot"]),
                        "mean_absolute_change": float(
                            np.mean(np.abs(output - sample["degraded"]))
                        ),
                    }
                )
            summary = summarize(records)
            summary["specification"] = specification
            summary["eligible"] = bool(
                summary["delta_psnr_parent_mean_db"]
                > float(gates["delta_psnr_parent_mean_gt_db"])
                and summary["delta_gradient_parent_mean"]
                <= float(gates["delta_gradient_parent_mean_lte"])
                and summary["delta_overshoot_parent_mean"]
                <= float(gates["delta_overshoot_parent_mean_lte"])
            )
            grid_results[spec_id(specification)] = summary
            if index % 32 == 0:
                print(
                    json.dumps(
                        {
                            "severity": severity,
                            "completed": index,
                            "total": len(specifications),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        eligible = [result for result in grid_results.values() if result["eligible"]]
        ranking_pool = eligible or list(grid_results.values())
        selected = max(
            ranking_pool, key=lambda result: result["delta_psnr_parent_mean_db"]
        )
        results[str(severity)] = {
            "parameters": parameters,
            "selected": selected,
            "eligible_count": len(eligible),
            "grid": grid_results,
        }
        print(
            json.dumps(
                {
                    "severity": severity,
                    "eligible_count": len(eligible),
                    "selected": selected,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = (
        OUTPUT
        if arguments.max_crops is None
        else ROOT / "runs" / "smoke" / f"motion_scan_{len(rows)}crops.json"
    )
    payload = {
        "schema_version": "MERIT-C061-MOTION-SCAN-v1",
        "status": "DEVELOPMENT_SCAN_COMPLETE",
        "scientific_status": "C006_33_PARENT_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_sha256": sha256_file(PROTOCOL),
        "script_sha256": actual_script,
        "bank_sha256": sha256_file(C006 / "artifacts" / "d0_bank_448.npy"),
        "manifest_sha256": sha256_file(C006 / "artifacts" / "d0_bank_manifest.csv"),
        "crops": len(rows),
        "parents": len({row["parent_sha256"] for row in rows}),
        "candidate_count": len(specifications),
        "max_crops_argument": arguments.max_crops,
        "results": results,
    }
    write_json_atomic(output, payload)
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}))


if __name__ == "__main__":
    main()
