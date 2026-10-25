"""Correct C054's conditional-range comparison to matched factorial main effects."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C055 = HERE.parent
MODIFIED = C055.parent
C054 = MODIFIED / "c054_triple_path_framework"
C051_SCRIPTS = MODIFIED / "c051_framework_dominance" / "scripts"
if str(C051_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C051_SCRIPTS))

import c051_common as common  # noqa: E402
import c055_common as analysis  # noqa: E402

POLICY = "specific_mean"
REPLICATES = 5000
BOOTSTRAP_SEED = 20261055


def load_rows() -> list[dict[str, Any]]:
    path = C054 / "runs" / f"held__{POLICY}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("raw", "off", "framework"):
            row[key] = float(row[key])
    return rows


def execute() -> dict[str, Any]:
    source = common.load_json(C054 / "runs" / "selection_summary.json")
    if common.payload_sha256(source) != source.get("payload_sha256"):
        raise common.C051Error("C054 selection summary hash mismatch")
    if source["best_policy"] != POLICY:
        raise common.C051Error("C055 policy does not match the C054 winner")
    result = analysis.analyze(load_rows(), REPLICATES, BOOTSTRAP_SEED)
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C055-FACTORIAL-DEVELOPMENT-v1",
        "status": result["status"],
        "evidence_status": "post-C054 estimand correction; seed 1/2 not opened",
        "rationale": (
            "Compare the framework and backbone main effects from the same balanced 2x4 factorial. "
            "The raw bypass is common to all backbone levels, so backbone marginal contrasts average "
            "the raw and selected-framework levels, matching C051's estimator."
        ),
        "source_c054_payload_sha256": source["payload_sha256"],
        "policy": POLICY,
        "selection": source["policy_results"][POLICY]["selection"],
        "bootstrap": {"replicates": REPLICATES, "seed": BOOTSTRAP_SEED},
        "result": result,
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_atomic(C055 / "runs" / "development_summary.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))
