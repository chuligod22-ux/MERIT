"""Summarize C012 development or holdout endpoints with clustered bootstrap."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
C012 = HERE.parent
MODIFIED = C012.parent
C011_SCRIPTS = MODIFIED / "c011_field_validation" / "scripts"
if str(C011_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C011_SCRIPTS))

import c011_common as common
import c012_eval as evaluation
import c012_profiles as profiles


SCHEMA = "MERIT-C012-FIELD-MODEL-SUMMARY-v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261031


def mean(values: Iterable[float]) -> float:
    materialized = np.asarray(list(values), dtype=np.float64)
    if not len(materialized):
        raise ValueError("Cannot average an empty collection")
    return float(materialized.mean())


def stratified_bootstrap(rows: list[dict[str, Any]], key: str, seed: int) -> dict[str, Any]:
    by_stratum: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        by_stratum[str(row["tunnel"])].append(float(row[key]))
    per_tunnel = {name: mean(values) for name, values in sorted(by_stratum.items())}
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        values = []
        for tunnel in sorted(by_stratum):
            candidates = np.asarray(by_stratum[tunnel], dtype=np.float64)
            values.append(float(rng.choice(candidates, size=len(candidates), replace=True).mean()))
        samples[index] = mean(values)
    return {
        "mean": mean(per_tunnel.values()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "per_tunnel": per_tunnel,
        "positive_tunnels": sum(value > 0 for value in per_tunnel.values()),
        "negative_or_zero_tunnels": sum(value <= 0 for value in per_tunnel.values()),
        "clusters": len(rows),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
    }


def source_bootstrap(rows: list[dict[str, Any]], key: str, seed: int) -> dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    samples = values[choices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "clusters": len(values),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
    }


def summarize_d2(stage: str) -> dict[str, Any]:
    root = evaluation.output_root(stage, "d2")
    files = sorted((root / "progress").glob("fold_*_seed*.json"))
    expected = len(evaluation.SPLITS[stage]["tunnels"]) * 3
    if len(files) != expected:
        raise ValueError(f"Expected {expected} D2 files, found {len(files)}")
    raw_rows = []
    for path in files:
        payload = common.load_json(path)
        for detector, parents in payload["detectors"].items():
            for parent, conditions in parents.items():
                raw = conditions["raw"]
                field = conditions["field_profile"]
                burden = payload["matched_sensitivity_burden"][detector][parent]
                raw_rows.append(
                    {
                        "tunnel": payload["fold"],
                        "seed": int(payload["seed"]),
                        "detector": detector,
                        "parent": parent,
                        "delta_auprc": float(field["pixel_auprc"] - raw["pixel_auprc"]),
                        "delta_f1": float(field["f1_at_0_5"] - raw["f1_at_0_5"]),
                        "delta_matched_fpr": float(burden["delta_false_positive_pixel_rate_restored_minus_degraded"]),
                    }
                )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_rows:
        grouped[(row["tunnel"], row["detector"], row["parent"])].append(row)
    detector_parent_rows = []
    for (tunnel, detector, parent), rows in sorted(grouped.items()):
        detector_parent_rows.append(
            {
                "tunnel": tunnel,
                "detector": detector,
                "parent": parent,
                **{key: mean(row[key] for row in rows) for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")},
            }
        )
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in detector_parent_rows:
        by_parent[(row["tunnel"], row["parent"])].append(row)
    panel_rows = [
        {
            "tunnel": tunnel,
            "parent": parent,
            **{key: mean(row[key] for row in rows) for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")},
        }
        for (tunnel, parent), rows in sorted(by_parent.items())
    ]
    panel_auprc = stratified_bootstrap(panel_rows, "delta_auprc", BOOTSTRAP_SEED)
    panel_f1 = stratified_bootstrap(panel_rows, "delta_f1", BOOTSTRAP_SEED + 1)
    panel_fpr = stratified_bootstrap(panel_rows, "delta_matched_fpr", BOOTSTRAP_SEED + 2)
    detector_summaries = {}
    for index, detector in enumerate(common.DETECTORS):
        rows = [row for row in detector_parent_rows if row["detector"] == detector]
        detector_summaries[detector] = {
            "delta_auprc": stratified_bootstrap(rows, "delta_auprc", BOOTSTRAP_SEED + 10 + index),
            "delta_f1": stratified_bootstrap(rows, "delta_f1", BOOTSTRAP_SEED + 20 + index),
            "delta_matched_fpr": stratified_bootstrap(rows, "delta_matched_fpr", BOOTSTRAP_SEED + 30 + index),
        }
    detector_nonnegative = all(value["delta_auprc"]["mean"] >= 0.0 for value in detector_summaries.values())
    pass_gate = bool(
        panel_auprc["mean"] > 0.0
        and panel_auprc["ci95_low"] > 0.0
        and panel_auprc["positive_tunnels"] == len(evaluation.SPLITS[stage]["tunnels"])
        and panel_fpr["mean"] <= 0.0
        and detector_nonnegative
    )
    return {
        "stage": stage,
        "files": len(files),
        "tiles": sum(int(common.load_json(path)["tile_count"]) for path in files),
        "parent_detector_clusters": len(detector_parent_rows),
        "panel": {"delta_auprc": panel_auprc, "delta_f1": panel_f1, "delta_matched_fpr": panel_fpr},
        "detectors": detector_summaries,
        "gate": {
            "pass": pass_gate,
            "requirements": "panel delta AUPRC mean and CI lower >0; every tunnel positive; matched FPR mean <=0; every detector mean delta AUPRC >=0",
        },
    }


def summarize_width(stage: str) -> dict[str, Any]:
    root = evaluation.output_root(stage, "width")
    files = sorted((root / "progress").glob("W*_seed*.json"))
    expected = len(evaluation.SPLITS[stage]["width_sources"]) * 3
    if len(files) != expected:
        raise ValueError(f"Expected {expected} width files, found {len(files)}")
    paired = []
    raw_nonestimable = 0
    field_nonestimable = 0
    total_primary = 0
    for path in files:
        payload = common.load_json(path)
        lookup = {(float(row["nominal_width_mm"]), row["condition"]): row for row in payload["measurements"]}
        for nominal in np.arange(0.3, 1.01, 0.1):
            nominal = round(float(nominal), 1)
            raw = lookup[(nominal, "raw")]
            field = lookup[(nominal, "width_profile")]
            total_primary += 1
            raw_nonestimable += raw["status"] != "ESTIMABLE"
            field_nonestimable += field["status"] != "ESTIMABLE"
            if raw["status"] == "ESTIMABLE" and field["status"] == "ESTIMABLE":
                paired.append(
                    {
                        "source_id": payload["source_id"],
                        "seed": int(payload["seed"]),
                        "nominal_width_mm": nominal,
                        "raw_absolute_error_mm": float(raw["absolute_error_mm"]),
                        "field_absolute_error_mm": float(field["absolute_error_mm"]),
                        "delta_absolute_error_mm": float(field["absolute_error_mm"] - raw["absolute_error_mm"]),
                    }
                )
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in paired:
        grouped[(row["source_id"], row["nominal_width_mm"])].append(row)
    source_nominal_rows = [
        {
            "source_id": source,
            "nominal_width_mm": nominal,
            **{key: mean(row[key] for row in rows) for key in ("raw_absolute_error_mm", "field_absolute_error_mm", "delta_absolute_error_mm")},
        }
        for (source, nominal), rows in sorted(grouped.items())
    ]
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in source_nominal_rows:
        by_source[row["source_id"]].append(row)
    source_rows = [
        {
            "source_id": source,
            **{key: mean(row[key] for row in rows) for key in ("raw_absolute_error_mm", "field_absolute_error_mm", "delta_absolute_error_mm")},
        }
        for source, rows in sorted(by_source.items())
    ]
    delta = source_bootstrap(source_rows, "delta_absolute_error_mm", BOOTSTRAP_SEED + 100)
    nominal_deltas = {
        f"{nominal:.1f}": mean(row["delta_absolute_error_mm"] for row in source_nominal_rows if row["nominal_width_mm"] == nominal)
        for nominal in [round(value, 1) for value in np.arange(0.3, 1.01, 0.1)]
    }
    improved_widths = sum(value < 0.0 for value in nominal_deltas.values())
    raw_error = mean(row["raw_absolute_error_mm"] for row in source_rows)
    pass_gate = bool(
        delta["mean"] < 0.0
        and delta["ci95_high"] < 0.0
        and improved_widths >= 6
        and field_nonestimable / total_primary <= raw_nonestimable / total_primary + 0.02
    )
    return {
        "stage": stage,
        "files": len(files),
        "paired_seed_rows": len(paired),
        "source_nominal_clusters": len(source_nominal_rows),
        "delta_absolute_error_mm": delta,
        "raw_mean_absolute_error_mm": raw_error,
        "relative_error_reduction_percent": float(-100.0 * delta["mean"] / raw_error),
        "nominal_width_delta_mm": nominal_deltas,
        "improved_nominal_widths": improved_widths,
        "primary_nominal_widths": 8,
        "raw_nonestimable_rate": raw_nonestimable / total_primary,
        "field_nonestimable_rate": field_nonestimable / total_primary,
        "gate": {
            "pass": pass_gate,
            "requirements": "delta absolute error mean and CI upper <0; at least 6/8 nominal widths improve; non-estimable rate increase <=2 percentage points",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(evaluation.SPLITS), required=True)
    args = parser.parse_args()
    d2 = summarize_d2(args.stage)
    width = summarize_width(args.stage)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS_COMPLETE" if d2["gate"]["pass"] and width["gate"]["pass"] else "COMPLETE_GATE_FAIL",
        "scientific_status": "DEVELOPMENT_ONLY" if args.stage == "development" else "SEALED_INTERNAL_HOLDOUT_NOT_PROSPECTIVE_EXTERNAL",
        "stage": args.stage,
        "profiles": {"detectors": profiles.DETECTOR_PROFILE, "width": profiles.WIDTH_PROFILE},
        "d2": d2,
        "width": width,
        "global_gate_pass": bool(d2["gate"]["pass"] and width["gate"]["pass"]),
        "claim_boundary": "C012 reuses C011 assets split into development and internal holdout. It cannot establish prospective external field efficacy or natural-crack physical-width accuracy.",
    }
    result["payload_sha256"] = common.payload_sha256(result)
    version = "development_v2" if args.stage == "development" else "holdout_v1"
    output = C012 / "runs" / version / "summary.json"
    common.write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
