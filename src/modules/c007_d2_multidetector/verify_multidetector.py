#!/usr/bin/env python3
"""Independent raw-result verifier for the MERIT D2 multi-detector analysis.

This file intentionally does not import d2_multidetector.py or its derived CSVs.
It rebuilds the primary estimands from the twelve raw evaluation JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FOLDS = ("A", "B", "C", "D")
CHAINS = (0, 1, 2)
CONDITIONS = ("reference", "degraded", "task_off", "task_on")
INDEPENDENT = (
    "d2_resnet34_unet",
    "d2_vgg16_unet",
    "d2_deeplabv3_resnet50",
)
LEGACY = "legacy_deepcrack_vgg16_unet"
DETECTORS = (*INDEPENDENT, LEGACY)
METRICS = ("pixel_auprc", "f1_at_0_5")
COMPARISONS = (
    ("reference_minus_degraded", "reference", "degraded"),
    ("task_on_minus_degraded", "task_on", "degraded"),
    ("task_off_minus_degraded", "task_off", "degraded"),
    ("task_on_minus_task_off", "task_on", "task_off"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("empty mean")
    return float(sum(values) / len(values))


def detector_summary(
    parent: dict[tuple[str, str, str, str], dict[str, float]],
    detector: str,
    metric: str,
    left: str,
    right: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    parents = sorted({key[2] for key in parent if key[0] == detector})
    for tunnel in FOLDS:
        for parent_id in parents:
            left_key = (detector, tunnel, parent_id, left)
            right_key = (detector, tunnel, parent_id, right)
            if left_key in parent:
                values[tunnel].append(
                    parent[left_key][metric] - parent[right_key][metric]
                )
    per_tunnel = {tunnel: average(values[tunnel]) for tunnel in FOLDS}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        tunnel_means: list[float] = []
        for tunnel in FOLDS:
            source = np.asarray(values[tunnel], dtype=np.float64)
            tunnel_means.append(
                float(rng.choice(source, size=source.size, replace=True).mean())
            )
        draws[draw] = np.mean(tunnel_means)
    return {
        "point": average(list(per_tunnel.values())),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "positive_tunnels": sum(value > 0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def panel_summary(
    parent: dict[tuple[str, str, str, str], dict[str, float]],
    metric: str,
    left: str,
    right: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for tunnel in FOLDS:
        parent_ids = sorted(
            {key[2] for key in parent if key[0] == INDEPENDENT[0] and key[1] == tunnel}
        )
        for parent_id in parent_ids:
            deltas = [
                parent[(detector, tunnel, parent_id, left)][metric]
                - parent[(detector, tunnel, parent_id, right)][metric]
                for detector in INDEPENDENT
            ]
            values[tunnel].append(average(deltas))
    per_tunnel = {tunnel: average(values[tunnel]) for tunnel in FOLDS}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        tunnel_means: list[float] = []
        for tunnel in FOLDS:
            source = np.asarray(values[tunnel], dtype=np.float64)
            tunnel_means.append(
                float(rng.choice(source, size=source.size, replace=True).mean())
            )
        draws[draw] = np.mean(tunnel_means)
    return {
        "point": average(list(per_tunnel.values())),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "positive_tunnels": sum(value > 0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def passed(result: dict[str, Any]) -> bool:
    return bool(
        result["point"] > 0
        and result["ci_lower"] > 0
        and result["positive_tunnels"] >= 3
    )


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(a - b) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    final_path = root / "multidetector_final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))

    raw: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    evaluation_hashes: dict[str, str] = {}
    fold_tile_counts: dict[str, set[int]] = defaultdict(set)
    parent_sets: dict[str, set[str]] = defaultdict(set)
    for fold in FOLDS:
        for chain in CHAINS:
            path = root / "evaluation" / f"fold_{fold}_chain{chain}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            evaluation_hashes[path.name] = sha256(path)
            if payload["fold"] != fold or int(payload["pairing"]["chain"]) != chain:
                raise ValueError(f"identity mismatch: {path}")
            fold_tile_counts[fold].add(int(payload["tile_count"]))
            if set(payload["detectors"]) != set(DETECTORS):
                raise ValueError(f"detector set mismatch: {path}")
            for detector, parents in payload["detectors"].items():
                parent_sets[fold].update(parents)
                for parent_id, conditions in parents.items():
                    if set(conditions) != set(CONDITIONS):
                        raise ValueError(f"condition set mismatch: {path} {parent_id}")
                    for condition, metrics in conditions.items():
                        key = (detector, fold, parent_id, condition)
                        raw[key].append(
                            {
                                metric: float(metrics[metric]) for metric in METRICS
                            }
                        )

    if any(len(values) != 3 for values in raw.values()):
        raise ValueError("every detector/fold/parent/condition must have three chains")
    if any(len(counts) != 1 for counts in fold_tile_counts.values()):
        raise ValueError("tile count differs across chains")
    parent: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for key in sorted(raw):
        parent[key] = {
            metric: average([row[metric] for row in raw[key]]) for metric in METRICS
        }

    base_seed = 20260924
    resamples = 10_000
    computed: dict[str, Any] = {}
    mismatches: list[str] = []
    for comparison_index, (name, left, right) in enumerate(COMPARISONS):
        computed[name] = {"detectors": {}, "panel": {}}
        for metric_index, metric in enumerate(METRICS):
            for detector_index, detector in enumerate(DETECTORS):
                result = detector_summary(
                    parent,
                    detector,
                    metric,
                    left,
                    right,
                    resamples,
                    base_seed + comparison_index * 100 + metric_index * 10 + detector_index,
                )
                computed[name]["detectors"].setdefault(detector, {})[metric] = result
                expected = final["comparisons"][name]["detectors"][detector][metric]
                for field in ("point", "ci_lower", "ci_upper"):
                    if not close(result[field], float(expected[field])):
                        mismatches.append(f"{name}/{detector}/{metric}/{field}")
                if result["positive_tunnels"] != expected["positive_tunnels"]:
                    mismatches.append(f"{name}/{detector}/{metric}/positive_tunnels")
            result = panel_summary(
                parent,
                metric,
                left,
                right,
                resamples,
                base_seed + comparison_index * 100 + metric_index * 10 + 9,
            )
            computed[name]["panel"][metric] = result
            expected = final["comparisons"][name]["panel"][metric]
            for field in ("point", "ci_lower", "ci_upper"):
                if not close(result[field], float(expected[field])):
                    mismatches.append(f"{name}/panel/{metric}/{field}")
            if result["positive_tunnels"] != expected["positive_tunnels"]:
                mismatches.append(f"{name}/panel/{metric}/positive_tunnels")

    primary = computed["task_on_minus_degraded"]
    detector_gates = {
        detector: passed(primary["detectors"][detector]["pixel_auprc"])
        for detector in INDEPENDENT
    }
    panel = primary["panel"]["pixel_auprc"]
    panel_gate = bool(
        panel["point"] > 0
        and panel["ci_lower"] > 0
        and sum(detector_gates.values()) >= 2
        and panel["positive_tunnels"] >= 3
    )
    if detector_gates != final["gates"]["primary_detector_gates"]:
        mismatches.append("primary_detector_gates")
    if panel_gate != final["gates"]["primary_panel_gate"]:
        mismatches.append("primary_panel_gate")
    verdict = "SUPPORTED_WITH_LIMITS" if panel_gate else "NOT_SUPPORTED"
    if verdict != final["verdict"]:
        mismatches.append("verdict")

    report = {
        "schema_version": "MERIT-C007-D2-MULTIDETECTOR-INDEPENDENT-VERIFY-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "method_independence": (
            "Standalone implementation; reads only the 12 raw evaluation JSON files "
            "and the final JSON for comparison; does not import the analysis module or derived CSVs."
        ),
        "raw_evaluation_files": len(evaluation_hashes),
        "raw_evaluation_sha256": evaluation_hashes,
        "fold_tile_counts": {
            fold: next(iter(fold_tile_counts[fold])) for fold in FOLDS
        },
        "total_tiles": sum(next(iter(fold_tile_counts[fold])) for fold in FOLDS),
        "parents_by_fold": {fold: len(parent_sets[fold]) for fold in FOLDS},
        "recomputed_primary": primary,
        "recomputed_primary_detector_gates": detector_gates,
        "recomputed_primary_panel_gate": panel_gate,
        "recomputed_verdict": verdict,
        "tolerance": 1e-12,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
        "final_json_sha256": sha256(final_path),
    }
    output = root / "independent_verification.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
