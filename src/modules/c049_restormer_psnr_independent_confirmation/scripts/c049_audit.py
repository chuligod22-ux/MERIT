"""Read-only independent recomputation audit for the completed C049 result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import c049_common as common


METRICS = (
    "delta_psnr_db",
    "delta_lpips_alex",
    "delta_gradient_mae",
    "delta_overshoot_p95",
)


def bootstrap(values: np.ndarray, seed: int, replicates: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = values[
        rng.integers(0, len(values), size=(replicates, len(values)))
    ].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    protocol = common.load_protocol()
    summary_path = common.OUTPUT / "confirmation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    cell_audit = []
    for seed in protocol["model_seeds"]:
        cell_path = common.OUTPUT / f"c040_seed{seed}.json"
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        records_path = Path(cell["records_path"])
        records_sha = common.sha256_file(records_path)
        if records_sha != cell["records_sha256"]:
            raise common.C049Error(f"Record hash mismatch: seed {seed}")
        seed_rows = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(seed_rows) != 1350 or len({row["parent_sha256"] for row in seed_rows}) != 302:
            raise common.C049Error(f"Coverage mismatch: seed {seed}")
        rows.extend(seed_rows)
        cell_audit.append(
            {
                "model_seed": seed,
                "records": len(seed_rows),
                "parents": len({row["parent_sha256"] for row in seed_rows}),
                "records_sha256": records_sha,
                "identity_max_abs_error": cell["identity_max_abs_error"],
            }
        )

    maximum_difference = 0.0
    conditions: dict[str, Any] = {}
    for condition_index, condition in enumerate(
        item["id"] for item in protocol["conditions"]
    ):
        subset = [row for row in rows if row["condition"] == condition]
        parents = sorted({row["parent_sha256"] for row in subset})
        if len(subset) != 1350 or len(parents) != 302:
            raise common.C049Error(f"Condition coverage mismatch: {condition}")
        recomputed = {}
        for metric_index, metric in enumerate(METRICS):
            parent_values = np.asarray(
                [
                    np.mean(
                        [row[metric] for row in subset if row["parent_sha256"] == parent]
                    )
                    for parent in parents
                ]
            )
            recomputed[metric] = bootstrap(
                parent_values,
                int(protocol["bootstrap_seed"]) + condition_index * 10 + metric_index,
                int(protocol["bootstrap_replicates"]),
            )
            for field, value in recomputed[metric].items():
                difference = abs(
                    value - summary["conditions"][condition]["metrics"][metric][field]
                )
                maximum_difference = max(maximum_difference, difference)
        seed_means = {
            str(seed): float(
                np.mean(
                    [
                        row["delta_psnr_db"]
                        for row in subset
                        if row["model_seed"] == seed
                    ]
                )
            )
            for seed in protocol["model_seeds"]
        }
        for seed, value in seed_means.items():
            maximum_difference = max(
                maximum_difference,
                abs(value - summary["conditions"][condition]["seed_delta_psnr_means"][seed]),
            )
        conditions[condition] = {
            "records": len(subset),
            "parents": len(parents),
            "metrics": recomputed,
            "seed_delta_psnr_means": seed_means,
        }

    identity_error = max(item["identity_max_abs_error"] for item in cell_audit)
    passed = bool(
        summary["status"] == "PASS"
        and summary["claim_authorized"]
        and len(rows) == 4050
        and len({row["parent_sha256"] for row in rows}) == 302
        and identity_error == 0.0
        and maximum_difference <= 1.0e-12
    )
    audit = {
        "schema_version": "MERIT-C049-INDEPENDENT-AUDIT-v1",
        "status": "PASS" if passed else "FAIL",
        "records": len(rows),
        "parents": len({row["parent_sha256"] for row in rows}),
        "conditions": conditions,
        "cells": cell_audit,
        "identity_max_abs_error": identity_error,
        "maximum_summary_difference": maximum_difference,
        "confirmation_summary_sha256": common.sha256_file(summary_path),
        "protocol_payload_sha256": protocol["payload_sha256"],
    }
    common.write_json(common.OUTPUT / "independent_audit_v1.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
