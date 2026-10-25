#!/usr/bin/env python3
"""Independent reconstruction of the C063 exported statistical tables.

This module intentionally does not import the execution/aggregation module.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
FREEZE = BASE / "protocol" / "full737_battery_freeze_v1.json"
OUTPUT = BASE / "artifacts" / "full737_battery_v1"
RESULT = OUTPUT / "independent_verification_v2.json"
SCHEMA = "MERIT-C063-INDEPENDENT-VERIFY-v2"
METRICS = ("mean_probability", "maximum_probability", "area_fraction_at_0.55")
METHODS = (
    "wiener_oracle",
    "richardson_lucy_oracle",
    "unsupervised_wiener_oracle",
    "wiener_wrong_psf",
    "tv_denoise",
    "nl_means",
    "unsharp",
    "nafnet_fidelity",
    "merit_core_adapter",
    "merit_routed_v3i",
)
PRIMARY_MERIT = "merit_core_adapter"
BOOTSTRAP_COUNT = 10_000
PERMUTATION_COUNT = 100_000
RNG_SEED = 20261024
RAW_TOLERANCE = 5e-12
# Recovery is a percentage ratio whose denominator can be close to zero.
# Reconstructing it from a CSV round trip amplifies last-place serialization
# differences.  This tolerance is 5e-12 in fractional-recovery units and does
# not change an estimate, confidence interval, p-value, or decision boundary.
RECOVERY_TOLERANCE = 5e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def array_value(value: Any) -> list[float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [float(item) for item in value]


def bootstrap_method(
    degraded_error: np.ndarray,
    method_error: np.ndarray,
    seed: int,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    count = len(degraded_error)
    indices = rng.integers(0, count, size=(BOOTSTRAP_COUNT, count))
    degraded_means = degraded_error[indices].mean(axis=1)
    method_means = method_error[indices].mean(axis=1)
    delta = method_means - degraded_means
    recovery = 100.0 * (degraded_means - method_means) / np.maximum(degraded_means, 1e-15)
    return (
        [float(value) for value in np.percentile(delta, [2.5, 97.5])],
        [float(value) for value in np.percentile(recovery, [2.5, 97.5])],
    )


def bootstrap_mean(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_COUNT, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def sign_flip_pvalue(values: np.ndarray, seed: int) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, PERMUTATION_COUNT, 10_000):
        count = min(10_000, PERMUTATION_COUNT - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed))
    return float((exceed + 1) / (PERMUTATION_COUNT + 1))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def close(actual: float, expected: float, tolerance: float = RAW_TOLERANCE) -> bool:
    return bool(abs(float(actual) - float(expected)) <= tolerance)


def close_array(
    actual: Any,
    expected: Any,
    tolerance: float = RAW_TOLERANCE,
) -> bool:
    return bool(np.allclose(array_value(actual), array_value(expected), rtol=0.0, atol=tolerance))


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frame = pd.read_csv(OUTPUT / "frame_results.csv")
    exported_condition = pd.read_csv(OUTPUT / "condition_results.csv")
    exported_summary = pd.read_csv(OUTPUT / "method_summary.csv")
    exported_dominance = pd.read_csv(OUTPUT / "merit_dominance.csv")
    group_keys = [
        "condition_cell", "speed_kmh", "distance_m", "iso", "primary_condition", "route_active"
    ]
    error_columns = [column for column in frame.columns if column.endswith("_abs_source_error")]
    condition = frame.groupby(group_keys, as_index=False)[error_columns].median()
    condition["frames"] = frame.groupby(group_keys)["frame"].count().to_numpy()
    condition = condition.sort_values(group_keys).reset_index(drop=True)
    exported_condition = exported_condition.sort_values(group_keys).reset_index(drop=True)
    primary = condition[condition["primary_condition"] == True].copy()  # noqa: E712
    checks = {
        "freeze_schema": freeze.get("schema_version") == "MERIT-C063-FULL737-BATTERY-v1",
        "frame_count_737": len(frame) == 737,
        "unique_frame_rows_737": len(frame[["condition_cell", "frame"]].drop_duplicates()) == 737,
        "condition_count_50": len(condition) == 50,
        "primary_condition_count_45": len(primary) == 45,
        "condition_table_reconstructed": (
            list(condition.columns) == list(exported_condition.columns)
            and np.allclose(
                condition.select_dtypes(include=[np.number]).to_numpy(float),
                exported_condition.select_dtypes(include=[np.number]).to_numpy(float),
                rtol=0.0,
                atol=RAW_TOLERANCE,
            )
        ),
        "all_values_finite": bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all()),
    }
    maximum_discrepancy = 0.0
    reconstructed_rows = 0
    for metric_index, metric in enumerate(METRICS):
        degraded = primary[f"{metric}_degraded_abs_source_error"].to_numpy(float)
        method_errors = {
            method: primary[f"{metric}_{method}_abs_source_error"].to_numpy(float)
            for method in METHODS
        }
        pvalues = {
            method: sign_flip_pvalue(
                method_errors[method] - degraded,
                RNG_SEED + 1000 * metric_index + method_index,
            )
            for method_index, method in enumerate(METHODS)
        }
        adjusted = holm_adjust(pvalues)
        for method_index, method in enumerate(METHODS):
            values = method_errors[method]
            delta = values - degraded
            delta_ci, recovery_ci = bootstrap_method(
                degraded, values, RNG_SEED + 10_000 * metric_index + method_index
            )
            row = exported_summary[
                (exported_summary["metric"] == metric) & (exported_summary["method"] == method)
            ].iloc[0]
            expected = {
                "mean_degraded_abs_source_error": degraded.mean(),
                "mean_method_abs_source_error": values.mean(),
                "mean_delta_error": delta.mean(),
                "recovery_percent": 100.0 * (degraded.mean() - values.mean()) / max(degraded.mean(), 1e-15),
                "paired_sign_flip_p_two_sided": pvalues[method],
                "holm_adjusted_p": adjusted[method],
            }
            for key, value in expected.items():
                maximum_discrepancy = max(maximum_discrepancy, abs(float(row[key]) - float(value)))
                tolerance = RECOVERY_TOLERANCE if key == "recovery_percent" else RAW_TOLERANCE
                checks[f"summary_{metric}_{method}_{key}"] = close(row[key], value, tolerance)
            checks[f"summary_{metric}_{method}_delta_ci"] = close_array(
                row["delta_error_95pct_ci"], delta_ci
            )
            checks[f"summary_{metric}_{method}_recovery_ci"] = close_array(
                row["recovery_percent_95pct_ci"], recovery_ci, RECOVERY_TOLERANCE
            )
            reconstructed_rows += 1
        merit_error = method_errors[PRIMARY_MERIT]
        baseline_p = {
            method: sign_flip_pvalue(
                method_errors[method] - merit_error,
                RNG_SEED + 50_000 + 1000 * metric_index + method_index,
            )
            for method_index, method in enumerate(METHODS)
            if method != PRIMARY_MERIT
        }
        baseline_adjusted = holm_adjust(baseline_p)
        for method_index, method in enumerate(METHODS):
            if method == PRIMARY_MERIT:
                continue
            advantage = method_errors[method] - merit_error
            row = exported_dominance[
                (exported_dominance["metric"] == metric)
                & (exported_dominance["baseline"] == method)
            ].iloc[0]
            expected_mean = advantage.mean()
            expected_ci = bootstrap_mean(
                advantage, RNG_SEED + 60_000 + 1000 * metric_index + method_index
            )
            maximum_discrepancy = max(
                maximum_discrepancy,
                abs(float(row["mean_baseline_minus_merit_error"]) - float(expected_mean)),
            )
            checks[f"dominance_{metric}_{method}_mean"] = close(
                row["mean_baseline_minus_merit_error"], expected_mean
            )
            checks[f"dominance_{metric}_{method}_ci"] = close_array(
                row["condition_bootstrap_95pct_ci"], expected_ci
            )
            checks[f"dominance_{metric}_{method}_p"] = close(
                row["paired_sign_flip_p_two_sided"], baseline_p[method]
            )
            checks[f"dominance_{metric}_{method}_holm"] = close(
                row["holm_adjusted_p"], baseline_adjusted[method]
            )
    payload = {
        "schema_version": SCHEMA,
        "state": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reconstructed_method_rows": reconstructed_rows,
        "maximum_scalar_discrepancy": float(maximum_discrepancy),
        "verification_tolerances": {
            "raw_statistic_absolute": RAW_TOLERANCE,
            "recovery_percentage_absolute": RECOVERY_TOLERANCE,
            "reason": (
                "Recovery percentages divide by near-zero detector-response errors; "
                "CSV round-trip last-place differences are amplified by the ratio."
            ),
            "v1_verification_preserved": "independent_verification.json",
        },
        "bindings": {
            "verifier_v2_sha256": sha256_file(Path(__file__)),
            "freeze_sha256": sha256_file(FREEZE),
            "frame_results_sha256": sha256_file(OUTPUT / "frame_results.csv"),
            "condition_results_sha256": sha256_file(OUTPUT / "condition_results.csv"),
            "method_summary_sha256": sha256_file(OUTPUT / "method_summary.csv"),
            "merit_dominance_sha256": sha256_file(OUTPUT / "merit_dominance.csv"),
        },
        "claim_boundary": (
            "Computational reconstruction does not turn clustered frames into independent samples "
            "or create human-mask detection ground truth."
        ),
    }
    write_json_atomic(RESULT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["state"] == "PASS" else 1)


if __name__ == "__main__":
    main()
