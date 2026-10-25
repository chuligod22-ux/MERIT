"""Parent-cluster factorial analysis for C051 framework dominance."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

import c051_common as common

TUNNELS = ("A", "B", "C", "D")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261051


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backbone in common.BACKBONES:
        for arm in common.ARMS:
            for seed in common.SEEDS:
                root = common.EVALUATION / common.model_id(backbone, arm, seed)
                marker = common.load_json(root / "RUN_COMPLETE.json")
                if common.payload_sha256(marker) != marker.get("payload_sha256"):
                    raise common.C051Error(f"Evaluation marker mismatch: {root}")
                for tunnel in TUNNELS:
                    payload = common.load_json(root / f"fold_{tunnel}.json")
                    if common.payload_sha256(payload) != payload.get("payload_sha256"):
                        raise common.C051Error(
                            f"Evaluation fold mismatch: {root}/{tunnel}"
                        )
                    for detector, parents in payload["detectors"].items():
                        for parent, conditions in parents.items():
                            burden = payload["matched_sensitivity_burden"][detector][
                                parent
                            ]
                            rows.append(
                                {
                                    "tunnel": tunnel,
                                    "parent": parent,
                                    "detector": detector,
                                    "backbone": backbone,
                                    "arm": arm,
                                    "seed": seed,
                                    "auprc": float(
                                        conditions["restored"]["pixel_auprc"]
                                    ),
                                    "degraded_auprc": float(
                                        conditions["degraded"]["pixel_auprc"]
                                    ),
                                    "f1": float(conditions["restored"]["f1_at_0_5"]),
                                    "matched_fpr_delta": float(
                                        burden[
                                            "delta_false_positive_pixel_rate_restored_minus_degraded"
                                        ]
                                    ),
                                }
                            )
    return rows


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise common.C051Error("Cannot average an empty collection")
    return float(array.mean())


def selected_parent_rows(
    rows: list[dict[str, Any]], sampled: dict[str, list[str]] | None = None
) -> list[dict[str, Any]]:
    if sampled is None:
        return rows
    selected: list[dict[str, Any]] = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["tunnel"], row["parent"])].append(row)
    for tunnel, parents in sampled.items():
        for draw_index, parent in enumerate(parents):
            for row in grouped[(tunnel, parent)]:
                selected.append({**row, "parent": f"{parent}__draw{draw_index}"})
    return selected


def cell_means(
    rows: list[dict[str, Any]], metric: str = "auprc"
) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    for backbone in common.BACKBONES:
        for arm in common.ARMS:
            tunnel_means = []
            for tunnel in TUNNELS:
                parents = sorted(
                    {
                        row["parent"]
                        for row in rows
                        if row["tunnel"] == tunnel
                        and row["backbone"] == backbone
                        and row["arm"] == arm
                    }
                )
                parent_means = [
                    mean(
                        row[metric]
                        for row in rows
                        if row["tunnel"] == tunnel
                        and row["parent"] == parent
                        and row["backbone"] == backbone
                        and row["arm"] == arm
                    )
                    for parent in parents
                ]
                tunnel_means.append(mean(parent_means))
            cells[(backbone, arm)] = mean(tunnel_means)
    return cells


def effects(cells: dict[tuple[str, str], float]) -> dict[str, Any]:
    within = {
        backbone: cells[(backbone, "on")] - cells[(backbone, "off")]
        for backbone in common.BACKBONES
    }
    framework_main = mean(within.values())
    backbone_marginal = {
        backbone: mean(cells[(backbone, arm)] for arm in common.ARMS)
        for backbone in common.BACKBONES
    }
    pairwise = {
        f"{left}-minus-{right}": backbone_marginal[left] - backbone_marginal[right]
        for left, right in itertools.combinations(common.BACKBONES, 2)
    }
    max_backbone = max(abs(value) for value in pairwise.values())
    grand = mean(cells.values())
    framework_marginal = {
        arm: mean(cells[(backbone, arm)] for backbone in common.BACKBONES)
        for arm in common.ARMS
    }
    ss_framework = len(common.BACKBONES) * sum(
        (value - grand) ** 2 for value in framework_marginal.values()
    )
    ss_backbone = len(common.ARMS) * sum(
        (value - grand) ** 2 for value in backbone_marginal.values()
    )
    return {
        "cells": {f"{key[0]}:{key[1]}": value for key, value in cells.items()},
        "within_backbone": within,
        "framework_main": framework_main,
        "backbone_marginal": backbone_marginal,
        "backbone_pairwise": pairwise,
        "max_backbone": max_backbone,
        "dominance": framework_main - max_backbone,
        "ss_framework": ss_framework,
        "ss_backbone": ss_backbone,
        "ss_ratio": ss_framework / ss_backbone if ss_backbone > 0 else float("inf"),
    }


def group_framework_effect(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    values = sorted({str(row[key]) for row in rows})
    for value in values:
        subset = [row for row in rows if str(row[key]) == value]
        # Equal-weight every remaining design dimension through the same cell estimator.
        if key == "backbone":
            cells = cell_means(subset)
            result[value] = cells[(value, "on")] - cells[(value, "off")]
            continue
        paired: list[float] = []
        identity = ("tunnel", "parent", "backbone", "seed", "detector")
        grouped: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
        for row in subset:
            grouped[tuple(row[name] for name in identity)][row["arm"]] = row["auprc"]
        by_tunnel: dict[str, list[float]] = defaultdict(list)
        for item, arms in grouped.items():
            if set(arms) == set(common.ARMS):
                by_tunnel[str(item[0])].append(arms["on"] - arms["off"])
        paired = [mean(by_tunnel[tunnel]) for tunnel in TUNNELS if by_tunnel[tunnel]]
        result[value] = mean(paired)
    return result


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def execute() -> dict[str, Any]:
    common.validate_protocol()
    rows = load_rows()
    expected = 4 * 2 * 3 * 3 * 14
    if len(rows) != expected:
        raise common.C051Error(
            f"Expected {expected} detector-parent rows, found {len(rows)}"
        )
    observed = effects(cell_means(rows))
    parents_by_tunnel = {
        tunnel: sorted({row["parent"] for row in rows if row["tunnel"] == tunnel})
        for tunnel in TUNNELS
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {
        "framework_main": np.empty(BOOTSTRAP_REPLICATES),
        "max_backbone": np.empty(BOOTSTRAP_REPLICATES),
        "dominance": np.empty(BOOTSTRAP_REPLICATES),
        "ss_ratio": np.empty(BOOTSTRAP_REPLICATES),
    }
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = {
            tunnel: list(rng.choice(parents, size=len(parents), replace=True))
            for tunnel, parents in parents_by_tunnel.items()
        }
        value = effects(cell_means(selected_parent_rows(rows, sampled)))
        for key in boot:
            boot[key][index] = value[key]
    framework_ci = interval(boot["framework_main"])
    backbone_ci = interval(boot["max_backbone"])
    dominance_ci = interval(boot["dominance"])
    ratio_ci = interval(boot["ss_ratio"])
    by_backbone = observed["within_backbone"]
    by_detector = group_framework_effect(rows, "detector")
    by_tunnel = group_framework_effect(rows, "tunnel")
    gates = {
        "framework_positive": framework_ci["ci95_low"] > 0,
        "dominance": dominance_ci["ci95_low"] > 0,
        "backbone_consistency": all(value > 0 for value in by_backbone.values()),
        "detector_consistency": all(value > 0 for value in by_detector.values()),
        "tunnel_consistency": sum(value > 0 for value in by_tunnel.values()) >= 3,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    if status == "PASS":
        claim_status = "SUPPORTED_FOR_CRACK_DETECTION_AUPRC_WITHIN_C051_SCOPE"
    elif gates["framework_positive"]:
        claim_status = "POSITIVE_FRAMEWORK_EFFECT_WITHOUT_BACKBONE_DOMINANCE"
    else:
        claim_status = "FRAMEWORK_EFFECT_NOT_SUPPORTED"
    summary: dict[str, Any] = {
        "schema_version": "MERIT-C051-FACTORIAL-SUMMARY-v1",
        "status": status,
        "claim_status": claim_status,
        "coverage": {
            "rows": len(rows),
            "backbones": len(common.BACKBONES),
            "arms": len(common.ARMS),
            "model_detector_seeds": len(common.SEEDS),
            "detectors": 3,
            "physical_parents": 14,
            "tunnels": 4,
        },
        "primary": {
            "framework_main": {"estimate": observed["framework_main"], **framework_ci},
            "max_backbone": {"estimate": observed["max_backbone"], **backbone_ci},
            "dominance": {"estimate": observed["dominance"], **dominance_ci},
        },
        "secondary": {
            "ss_ratio": {"estimate": observed["ss_ratio"], **ratio_ci},
            "within_backbone": by_backbone,
            "by_detector": by_detector,
            "by_tunnel": by_tunnel,
            "cells": observed["cells"],
            "backbone_marginal": observed["backbone_marginal"],
            "backbone_pairwise": observed["backbone_pairwise"],
        },
        "gates": gates,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "physical parent stratified within tunnel",
        },
        "claim_boundary": "Detection AUPRC under the frozen synthetic degradation on the four revision-added tunnels; C050 optical PSNR remains a mandatory separate result.",
    }
    summary["payload_sha256"] = common.payload_sha256(summary)
    common.write_json_atomic(common.RUNS / "factorial_summary.json", summary)
    csv_path = common.RUNS / "factorial_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
