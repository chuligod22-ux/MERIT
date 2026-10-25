"""Freeze C055 selection, estimands, gates, and confirmation code before seed 1/2."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C055 = HERE.parent
MODIFIED = C055.parent
C051_SCRIPTS = MODIFIED / "c051_framework_dominance" / "scripts"
if str(C051_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C051_SCRIPTS))

import c051_common as common  # noqa: E402

FILES = (
    "scripts/c055_common.py",
    "scripts/c055_confirm.py",
    "scripts/c055_queue.py",
    "scripts/c055_summarize.py",
    "scripts/test_c055.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute() -> dict[str, Any]:
    output = C055 / "protocol" / "confirmation_freeze_v1.json"
    if output.exists():
        payload = common.load_json(output)
        if common.payload_sha256(payload) != payload.get("payload_sha256"):
            raise common.C051Error("existing C055 freeze hash mismatch")
        return payload
    development = common.load_json(C055 / "runs" / "development_summary.json")
    if common.payload_sha256(development) != development.get("payload_sha256"):
        raise common.C051Error("C055 development hash mismatch")
    if development["status"] != "PASS":
        raise common.C051Error("C055 cannot freeze a failed development result")
    for seed in (1, 2):
        root = C055 / "runs" / f"confirmation_seed{seed}"
        if root.exists() and any(root.rglob("fold_*.json")):
            raise common.C051Error("confirmation outputs exist before freeze")
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C055-CONFIRMATION-FREEZE-v1",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "development_payload_sha256": development["payload_sha256"],
        "source_c054_payload_sha256": development["source_c054_payload_sha256"],
        "policy": development["policy"],
        "selection": development["selection"],
        "confirmation_seeds": [1, 2],
        "detector_seeds": {"1": 20260922, "2": 20260923},
        "estimands": {
            "framework_main": "mean_backbone(framework_cell - raw_cell)",
            "backbone_main": "max_pairwise(abs(mean_level(raw, framework)_b1 - mean_level(raw, framework)_b2))",
            "factorial_dominance": "framework_main - backbone_main",
            "conditional_sensitivity": "framework_main - max_pairwise_framework_level_contrast",
        },
        "primary_gates": {
            "framework_positive": "parent-bootstrap 95% CI lower > 0",
            "factorial_dominance": "parent-bootstrap 95% CI lower > 0",
            "backbone_consistency": "4/4 pooled backbone full gains > 0",
            "detector_consistency": "3/3 pooled detector gains > 0",
            "tunnel_consistency": "4/4 pooled tunnel gains > 0",
            "seed_replication": "seed 1 and seed 2 point framework gains and factorial dominance > 0",
        },
        "bootstrap": {"replicates": 10000, "seed": 20261056, "cluster": "physical parent within tunnel"},
        "conditional_sensitivity_is_gate": False,
        "code_sha256": {relative: sha256_file(C055 / relative) for relative in FILES},
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_atomic(output, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))
