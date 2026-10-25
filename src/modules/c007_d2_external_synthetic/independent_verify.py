#!/usr/bin/env python3
"""Independently recompute C007 aggregate results from the 12 progress JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TUNNELS = ("A", "B", "C", "D")
CONDITIONS = ("reference", "degraded", "task_off", "task_on")
METRICS = ("pixel_auprc", "f1_at_0_5")
COMPARISONS = (
    ("reference_minus_degraded", "reference", "degraded"),
    ("task_on_minus_degraded", "task_on", "degraded"),
    ("task_off_minus_degraded", "task_off", "degraded"),
    ("task_on_minus_task_off", "task_on", "task_off"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{label}: independently computed {actual} != stored {expected}")


def bootstrap(
    parent_values: dict[tuple[str, str], dict[str, float]],
    left: str,
    right: str,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float]:
    differences: dict[str, list[float]] = defaultdict(list)
    for (tunnel, _parent), values in parent_values.items():
        differences[tunnel].append(values[left] - values[right])
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        tunnel_means = []
        for tunnel in TUNNELS:
            values = np.asarray(differences[tunnel], dtype=np.float64)
            draw = rng.choice(values, size=values.size, replace=True)
            tunnel_means.append(float(draw.mean()))
        samples[index] = np.mean(tunnel_means)
    point = mean([mean(differences[tunnel]) for tunnel in TUNNELS])
    return {
        "point": point,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    final_path = root / "external_synthetic_final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    protocol = json.loads((root.parent.parent / "protocol.json").read_text(encoding="utf-8"))

    raw: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    observed_tiles: dict[tuple[str, int], int] = {}
    progress_hashes: dict[str, str] = {}
    for tunnel in TUNNELS:
        for chain in range(3):
            path = root / "progress" / f"fold_{tunnel}_chain{chain}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["fold"] != tunnel or int(payload["pairing"]["chain"]) != chain:
                raise RuntimeError(f"Progress identity mismatch: {path}")
            observed_tiles[(tunnel, chain)] = int(payload["tile_count"])
            progress_hashes[path.name] = sha256_file(path)
            for parent, condition_values in payload["parents"].items():
                for condition in CONDITIONS:
                    for metric in METRICS:
                        raw[(tunnel, parent, condition, metric)].append(
                            float(condition_values[condition][metric])
                        )

    if len(progress_hashes) != 12:
        raise RuntimeError("Expected 12 independent progress files")
    if any(len(values) != 3 for values in raw.values()):
        raise RuntimeError("Every parent-condition-metric must contain exactly three chains")

    parent_by_metric: dict[str, dict[tuple[str, str], dict[str, float]]] = {
        metric: defaultdict(dict) for metric in METRICS
    }
    for (tunnel, parent, condition, metric), values in raw.items():
        parent_by_metric[metric][(tunnel, parent)][condition] = mean(values)

    tunnel_values: dict[str, dict[str, dict[str, float]]] = {
        tunnel: {condition: {} for condition in CONDITIONS} for tunnel in TUNNELS
    }
    for metric in METRICS:
        for tunnel in TUNNELS:
            parent_keys = sorted(
                key for key in parent_by_metric[metric] if key[0] == tunnel
            )
            for condition in CONDITIONS:
                tunnel_values[tunnel][condition][metric] = mean(
                    [parent_by_metric[metric][key][condition] for key in parent_keys]
                )

    equal_tunnel: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        equal_tunnel[condition] = {}
        for metric in METRICS:
            value = mean(
                [tunnel_values[tunnel][condition][metric] for tunnel in TUNNELS]
            )
            equal_tunnel[condition][metric] = value
            assert_close(
                value,
                float(final["equal_tunnel_macro"][condition][metric]),
                f"equal_tunnel.{condition}.{metric}",
            )

    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    for comparison_index, (name, left, right) in enumerate(COMPARISONS, start=1):
        comparisons[name] = {}
        for metric_index, metric in enumerate(METRICS):
            result = bootstrap(
                parent_by_metric[metric],
                left,
                right,
                bootstrap_seed + comparison_index * 10 + metric_index,
            )
            per_tunnel = {
                tunnel: tunnel_values[tunnel][left][metric]
                - tunnel_values[tunnel][right][metric]
                for tunnel in TUNNELS
            }
            result["positive_tunnels"] = sum(value > 0.0 for value in per_tunnel.values())
            result["per_tunnel"] = per_tunnel
            stored = final["comparisons"][name][metric]
            for key in ("point", "ci_lower", "ci_upper"):
                assert_close(float(result[key]), float(stored[key]), f"{name}.{metric}.{key}")
            if int(result["positive_tunnels"]) != int(stored["positive_tunnels"]):
                raise RuntimeError(f"{name}.{metric}.positive_tunnels mismatch")
            for tunnel in TUNNELS:
                assert_close(
                    float(result["per_tunnel"][tunnel]),
                    float(stored["per_tunnel"][tunnel]),
                    f"{name}.{metric}.per_tunnel.{tunnel}",
                )
            comparisons[name][metric] = result

    primary = comparisons["task_on_minus_degraded"]["pixel_auprc"]
    task_increment = comparisons["task_on_minus_task_off"]["pixel_auprc"]
    manipulation = comparisons["reference_minus_degraded"]["pixel_auprc"]
    gates = {
        "degradation_manipulation": manipulation["positive_tunnels"] >= 3,
        "task_on_primary": (
            primary["point"] > 0.0
            and primary["ci_lower"] > 0.0
            and primary["positive_tunnels"] >= 3
        ),
        "task_loss_increment": (
            task_increment["point"] > 0.0
            and task_increment["ci_lower"] > 0.0
            and task_increment["positive_tunnels"] >= 3
        ),
    }
    if gates != final["gates"]:
        raise RuntimeError(f"Gate mismatch: {gates} != {final['gates']}")

    report = {
        "status": "PASS",
        "method": "Independent recomputation from 12 progress JSONs; main analysis module not imported.",
        "progress_files": len(progress_hashes),
        "progress_sha256": progress_hashes,
        "observed_tiles_by_fold_chain": {
            f"{tunnel}_chain{chain}": count
            for (tunnel, chain), count in sorted(observed_tiles.items())
        },
        "equal_tunnel_macro": equal_tunnel,
        "comparisons": comparisons,
        "gates": gates,
        "stored_verdict": final["verdict"],
        "final_summary_sha256": sha256_file(final_path),
    }
    output = root / "independent_verification.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
