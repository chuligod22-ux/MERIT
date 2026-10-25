"""Independently audit C039 records against the frozen contract and summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import c039_common as common


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def main() -> None:
    freeze = common.load_freeze()
    summary_path = common.OUTPUT_ROOT / "confirmation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    cells = []
    expected_per_seed = (
        int(freeze["confirmation"]["crops"])
        * len(freeze["conditions"])
        * len(freeze["confirmation"]["degradation_seeds"])
    )
    for seed in freeze["confirmation"]["model_seeds"]:
        cell_path = common.OUTPUT_ROOT / f"seed{seed}.json"
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        records_path = Path(cell["records_path"])
        seed_records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if common.sha256_file(records_path) != cell["records_sha256"]:
            raise common.C039Error(f"Seed {seed} record hash mismatch")
        if len(seed_records) != expected_per_seed or cell["records"] != expected_per_seed:
            raise common.C039Error(f"Seed {seed} record coverage mismatch")
        if len({row["parent_sha256"] for row in seed_records}) != freeze["confirmation"]["parents"]:
            raise common.C039Error(f"Seed {seed} parent coverage mismatch")
        if cell["identity_max_abs_error"] != 0.0:
            raise common.C039Error(f"Seed {seed} identity mismatch")
        records.extend(seed_records)
        cells.append(cell)

    checks = {}
    for condition in (item["id"] for item in freeze["conditions"]):
        subset = [row for row in records if row["condition"] == condition]
        parents = sorted({row["parent_sha256"] for row in subset})
        expected_condition_records = (
            int(freeze["confirmation"]["crops"])
            * len(freeze["confirmation"]["degradation_seeds"])
            * len(freeze["confirmation"]["model_seeds"])
        )
        if len(subset) != expected_condition_records:
            raise common.C039Error(f"{condition} record coverage mismatch")
        if len(parents) != freeze["confirmation"]["parents"]:
            raise common.C039Error(f"{condition} parent coverage mismatch")
        recomputed = {}
        for metric in (
            "delta_psnr_db",
            "delta_lpips_alex",
            "delta_gradient_mae",
            "delta_overshoot_p95",
        ):
            parent_values = np.asarray(
                [
                    np.mean([float(row[metric]) for row in subset if row["parent_sha256"] == parent])
                    for parent in parents
                ]
            )
            recomputed[metric] = float(parent_values.mean())
            recorded = summary["conditions"][condition]["metrics"][metric]["mean"]
            if not close(recomputed[metric], recorded):
                raise common.C039Error(f"{condition} {metric} mean mismatch")
        seed_psnr = {
            str(seed): float(
                np.mean([row["delta_psnr_db"] for row in subset if row["model_seed"] == seed])
            )
            for seed in freeze["confirmation"]["model_seeds"]
        }
        for seed, value in seed_psnr.items():
            recorded = summary["conditions"][condition]["seed_delta_psnr_means"][seed]
            if not close(value, recorded):
                raise common.C039Error(f"{condition} seed {seed} PSNR mismatch")
        checks[condition] = {
            "records": len(subset),
            "parents": len(parents),
            "parent_macro_means": recomputed,
            "seed_delta_psnr_means": seed_psnr,
            "status": summary["conditions"][condition]["status"],
        }

    if summary.get("status") != "PASS" or not summary.get("all_conditions_pass"):
        raise common.C039Error("C039 summary did not pass")
    result = {
        "schema_version": "MERIT-C039-INDEPENDENT-AUDIT-v1",
        "status": "PASS",
        "confirmation_freeze_payload_sha256": freeze["payload_sha256"],
        "summary_sha256": common.sha256_file(summary_path),
        "records": len(records),
        "parents": len({row["parent_sha256"] for row in records}),
        "identity_max_abs_error": max(cell["identity_max_abs_error"] for cell in cells),
        "checks": checks,
    }
    common.c023.write_json(common.OUTPUT_ROOT / "independent_audit_v1.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
