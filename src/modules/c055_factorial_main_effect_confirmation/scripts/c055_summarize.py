"""Summarize the frozen seed 1/2 C055 confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C055 = HERE.parent
MODIFIED = C055.parent
C051 = MODIFIED / "c051_framework_dominance"
C051_SCRIPTS = C051 / "scripts"
if str(C051_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C051_SCRIPTS))

import c051_common as common  # noqa: E402
import c055_common as analysis  # noqa: E402

FREEZE = C055 / "protocol" / "confirmation_freeze_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_freeze() -> dict[str, Any]:
    payload = common.load_json(FREEZE)
    if common.payload_sha256(payload) != payload.get("payload_sha256"):
        raise common.C051Error("C055 freeze payload mismatch")
    for relative, expected in payload["code_sha256"].items():
        if sha256_file(C055 / relative) != expected:
            raise common.C051Error(f"C055 frozen code mismatch: {relative}")
    return payload


def load_rows(freeze: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    hashes = []
    for seed in freeze["confirmation_seeds"]:
        for backbone in analysis.BACKBONES:
            for tunnel in analysis.TUNNELS:
                path = C055 / "runs" / f"confirmation_seed{seed}" / backbone / f"fold_{tunnel}.json"
                payload = common.load_json(path)
                if common.payload_sha256(payload) != payload.get("payload_sha256"):
                    raise common.C051Error(f"C055 result hash mismatch: {path}")
                if payload["freeze_payload_sha256"] != freeze["payload_sha256"]:
                    raise common.C051Error(f"C055 result uses wrong freeze: {path}")
                if payload["selected_conditions"] != freeze["selection"][backbone][tunnel]["conditions"]:
                    raise common.C051Error(f"C055 selected condition mismatch: {path}")
                hashes.append(payload["payload_sha256"])
                for detector, parents in payload["parent_results"].items():
                    for parent, values in parents.items():
                        rows.append(
                            {
                                "seed": int(seed),
                                "tunnel": tunnel,
                                "parent": parent,
                                "detector": detector,
                                "backbone": backbone,
                                "raw": float(values["raw"]["pixel_auprc"]),
                                "off": float(values["off"]["pixel_auprc"]),
                                "framework": float(values["framework"]["pixel_auprc"]),
                            }
                        )
    return rows, hashes


def crosscheck_c051(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference: dict[tuple[int, str, str, str, str], tuple[float, float]] = {}
    path = C051 / "runs" / "factorial_rows_v2.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] != "off" or int(row["seed"]) not in (1, 2):
                continue
            key = (int(row["seed"]), row["tunnel"], row["parent"], row["detector"], row["backbone"])
            if key in reference:
                raise common.C051Error(f"duplicate C051 anchor: {key}")
            reference[key] = (float(row["degraded_auprc"]), float(row["auprc"]))
    raw_diff = []
    off_diff = []
    observed = set()
    for row in rows:
        key = (row["seed"], row["tunnel"], row["parent"], row["detector"], row["backbone"])
        if key not in reference:
            raise common.C051Error(f"C055 row missing from C051: {key}")
        observed.add(key)
        raw, off = reference[key]
        raw_diff.append(abs(float(row["raw"]) - raw))
        off_diff.append(abs(float(row["off"]) - off))
    result = {
        "observed_unique_rows": len(observed),
        "reference_rows": len(reference),
        "max_abs_raw_difference": max(raw_diff),
        "max_abs_off_difference": max(off_diff),
        "tolerance": 1e-12,
    }
    if result["max_abs_raw_difference"] > result["tolerance"] or result["max_abs_off_difference"] > result["tolerance"]:
        raise common.C051Error(f"C055/C051 anchor mismatch: {result}")
    return result


def execute() -> dict[str, Any]:
    freeze = load_freeze()
    rows, hashes = load_rows(freeze)
    anchor = crosscheck_c051(rows)
    bootstrap = freeze["bootstrap"]
    result = analysis.analyze(rows, int(bootstrap["replicates"]), int(bootstrap["seed"]))
    per_seed = {
        str(seed): analysis.effects([row for row in rows if row["seed"] == seed])
        for seed in freeze["confirmation_seeds"]
    }
    seed_replication = all(
        values["framework_main"] > 0 and values["factorial_dominance"] > 0
        for values in per_seed.values()
    )
    result["gates"]["seed_replication"] = seed_replication
    result["status"] = "PASS" if all(result["gates"].values()) else "FAIL"
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C055-CONFIRMATION-SUMMARY-v1",
        "status": result["status"],
        "evidence_status": "frozen seed 1/2 confirmation on the four reviewer-added external tunnels",
        "freeze_payload_sha256": freeze["payload_sha256"],
        "result_payload_count": len(hashes),
        "result_payload_sha256": hashes,
        "row_count": len(rows),
        "anchor_crosscheck": anchor,
        "result": result,
        "per_seed": {
            seed: {
                key: values[key]
                for key in (
                    "framework_main",
                    "max_backbone_main",
                    "factorial_dominance",
                    "max_backbone_conditional",
                    "conditional_sensitivity_dominance",
                    "within_backbone_full_gain",
                )
            }
            for seed, values in per_seed.items()
        },
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_atomic(C055 / "runs" / "confirmation_summary.json", payload)
    with (C055 / "runs" / "confirmation_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))
