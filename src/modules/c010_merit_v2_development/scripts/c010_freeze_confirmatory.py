"""Verify and freeze the unused-condition C010 confirmatory bank."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c010_pilot as pilot


C010 = HERE.parent
DESIGN = C010 / "protocol" / "confirmatory_source_design_v1.json"
OUTPUT = C010 / "protocol" / "confirmatory_freeze_v1.json"
ARTIFACTS = C010 / "artifacts" / "confirmatory_v1"
OLD_MANIFEST = (
    C010.parent / "c006_execution" / "artifacts" / "d0_bank_manifest.csv"
)
EVALUATOR = HERE / "c010_confirmatory_eval.py"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Freeze already exists: {OUTPUT}")
    design = pilot.load_json(DESIGN)
    bank_path = ARTIFACTS / "d0_bank_448.npy"
    manifest_path = ARTIFACTS / "d0_bank_manifest.csv"
    ledger_path = ARTIFACTS / "d0_bank_ledger.json"
    bank = np.load(bank_path, allow_pickle=False)
    rows = read_rows(manifest_path)
    old_rows = read_rows(OLD_MANIFEST)
    parents = {row["parent_sha256"] for row in rows}
    old_parents = {row["parent_sha256"] for row in old_rows}
    if bank.shape != (98, 448, 448) or len(rows) != 98:
        raise RuntimeError(f"Unexpected bank coverage: {bank.shape}, {len(rows)}")
    if len(parents) != 68:
        raise RuntimeError(f"Expected 68 unique parents, found {len(parents)}")
    overlap = parents & old_parents
    if overlap:
        raise RuntimeError(f"C006/C009 parent overlap: {len(overlap)}")
    for row in rows:
        index = int(row["bank_index"])
        actual = hashlib.sha256(np.ascontiguousarray(bank[index]).tobytes()).hexdigest()
        if actual != row["crop_sha256"]:
            raise RuntimeError(f"Crop hash mismatch: {index}")
    v3 = pilot.load_json(C010 / "protocol" / "pilot_v3.json")
    base_checkpoints: dict[str, Any] = {}
    for seed in design["model_rule"]["base_seeds"]:
        path = (
            pilot.C009 / "runs" / "wp1" / "nafnet" / f"base_seed{seed}" / "final.pt"
        )
        base_checkpoints[f"seed{seed}"] = {
            "path": str(path.resolve()),
            "sha256": pilot.sha256_file(path),
        }
    freeze = {
        "schema_version": "MERIT-C010-V2-CONFIRMATORY-FREEZE-v1",
        "status": "FROZEN_BEFORE_ANY_CONFIRMATORY_MODEL_OUTPUT",
        "frozen_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "design_path": str(DESIGN.resolve()),
        "design_sha256": pilot.sha256_file(DESIGN),
        "bank_path": str(bank_path.resolve()),
        "bank_sha256": pilot.sha256_file(bank_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": pilot.sha256_file(manifest_path),
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": pilot.sha256_file(ledger_path),
        "parents": len(parents),
        "crops": len(rows),
        "c006_c009_parent_overlap": 0,
        "base_seeds": design["model_rule"]["base_seeds"],
        "base_checkpoints": base_checkpoints,
        "calibration": v3["calibration"],
        "success_gates": design["success_gates"],
        "v3_protocol_sha256": pilot.sha256_file(C010 / "protocol" / "pilot_v3.json"),
        "model_code_sha256": pilot.sha256_file(HERE / "merit_v2_model.py"),
        "evaluator_sha256": pilot.sha256_file(EVALUATOR),
        "claim_boundary": design["claim_boundary"],
    }
    freeze["payload_sha256"] = hashlib.sha256(canonical_bytes(freeze)).hexdigest()
    pilot.write_json_atomic(OUTPUT, freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
