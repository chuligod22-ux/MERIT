"""Shared factorial estimators for C055."""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
C055 = HERE.parent
MODIFIED = C055.parent
C051_SCRIPTS = MODIFIED / "c051_framework_dominance" / "scripts"
if str(C051_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C051_SCRIPTS))

import c051_common as common  # noqa: E402

TUNNELS = ("A", "B", "C", "D")
BACKBONES = tuple(common.BACKBONES)


def mean(values: list[float]) -> float:
    if not values:
        raise common.C051Error("mean() received an empty group")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def cell(rows: list[dict[str, Any]], backbone: str, metric: str) -> float:
    tunnel_values = []
    for tunnel in TUNNELS:
        parents = sorted(
            {row["parent"] for row in rows if row["backbone"] == backbone and row["tunnel"] == tunnel}
        )
        tunnel_values.append(
            mean(
                [
                    mean(
                        [
                            float(row[metric])
                            for row in rows
                            if row["backbone"] == backbone
                            and row["tunnel"] == tunnel
                            and row["parent"] == parent
                        ]
                    )
                    for parent in parents
                ]
            )
        )
    return mean(tunnel_values)


def effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = {backbone: cell(rows, backbone, "raw") for backbone in BACKBONES}
    framework = {backbone: cell(rows, backbone, "framework") for backbone in BACKBONES}
    raw_spread = max(raw.values()) - min(raw.values())
    if raw_spread > 1e-12:
        raise common.C051Error(f"raw anchor differs across backbones: {raw_spread}")
    gains = {backbone: framework[backbone] - raw[backbone] for backbone in BACKBONES}
    framework_main = mean(list(gains.values()))
    backbone_marginal = {
        backbone: mean([raw[backbone], framework[backbone]]) for backbone in BACKBONES
    }
    pairwise_marginal = {
        f"{left}-minus-{right}": backbone_marginal[left] - backbone_marginal[right]
        for left, right in itertools.combinations(BACKBONES, 2)
    }
    pairwise_conditional = {
        f"{left}-minus-{right}": framework[left] - framework[right]
        for left, right in itertools.combinations(BACKBONES, 2)
    }
    max_backbone_main = max(abs(value) for value in pairwise_marginal.values())
    max_backbone_conditional = max(abs(value) for value in pairwise_conditional.values())
    return {
        "raw_cells": raw,
        "framework_cells": framework,
        "within_backbone_full_gain": gains,
        "framework_main": framework_main,
        "backbone_marginal": backbone_marginal,
        "max_backbone_main": max_backbone_main,
        "factorial_dominance": framework_main - max_backbone_main,
        "max_backbone_conditional": max_backbone_conditional,
        "conditional_sensitivity_dominance": framework_main - max_backbone_conditional,
        "backbone_pairwise_marginal": pairwise_marginal,
        "backbone_pairwise_conditional": pairwise_conditional,
        "raw_backbone_spread": raw_spread,
    }


def selected_parent_rows(
    rows: list[dict[str, Any]], sampled: dict[str, list[str]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["tunnel"], row["parent"])].append(row)
    result = []
    for tunnel, parents in sampled.items():
        for index, parent in enumerate(parents):
            result.extend({**row, "parent": f"{parent}__{index}"} for row in grouped[(tunnel, parent)])
    return result


def grouped_gain(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    if key not in {"detector", "tunnel"}:
        raise common.C051Error(f"unsupported grouping key: {key}")
    result = {}
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if str(row[key]) == value]
        if key == "tunnel":
            result[value] = mean([float(row["framework"]) - float(row["raw"]) for row in subset])
        else:
            result[value] = mean(
                [
                    mean(
                        [
                            float(row["framework"]) - float(row["raw"])
                            for row in subset
                            if row["tunnel"] == tunnel
                        ]
                    )
                    for tunnel in TUNNELS
                ]
            )
    return result


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def analyze(
    rows: list[dict[str, Any]], replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    observed = effects(rows)
    parents = {
        tunnel: sorted({row["parent"] for row in rows if row["tunnel"] == tunnel})
        for tunnel in TUNNELS
    }
    rng = np.random.default_rng(bootstrap_seed)
    keys = (
        "framework_main",
        "max_backbone_main",
        "factorial_dominance",
        "max_backbone_conditional",
        "conditional_sensitivity_dominance",
    )
    boot = {key: np.empty(replicates) for key in keys}
    for index in range(replicates):
        sampled = {
            tunnel: list(rng.choice(values, size=len(values), replace=True))
            for tunnel, values in parents.items()
        }
        value = effects(selected_parent_rows(rows, sampled))
        for key in keys:
            boot[key][index] = value[key]
    by_detector = grouped_gain(rows, "detector")
    by_tunnel = grouped_gain(rows, "tunnel")
    gates = {
        "framework_positive": interval(boot["framework_main"])["ci95_low"] > 0,
        "factorial_dominance": interval(boot["factorial_dominance"])["ci95_low"] > 0,
        "backbone_consistency": all(value > 0 for value in observed["within_backbone_full_gain"].values()),
        "detector_consistency": all(value > 0 for value in by_detector.values()),
        "tunnel_consistency": all(value > 0 for value in by_tunnel.values()),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "primary": {
            key: {"estimate": observed[key], **interval(boot[key])}
            for key in ("framework_main", "max_backbone_main", "factorial_dominance")
        },
        "conditional_sensitivity": {
            key: {"estimate": observed[key], **interval(boot[key])}
            for key in ("max_backbone_conditional", "conditional_sensitivity_dominance")
        },
        "within_backbone_full_gain": observed["within_backbone_full_gain"],
        "framework_cells": observed["framework_cells"],
        "raw_cells": observed["raw_cells"],
        "backbone_marginal": observed["backbone_marginal"],
        "by_detector": by_detector,
        "by_tunnel": by_tunnel,
        "gates": gates,
    }
