"""Post-run integrity and numerical audit for C051."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import c051_analyze as analysis
import c051_common as common


def execute() -> dict[str, Any]:
    protocol = common.validate_protocol()
    issues: list[str] = []
    training_cells = 0
    evaluation_cells = 0
    fold_files = 0
    for backbone in common.BACKBONES:
        for arm in common.ARMS:
            for seed in common.SEEDS:
                model_id = common.model_id(backbone, arm, seed)
                training_marker_path = (
                    common.run_dir(backbone, arm, seed) / "RUN_COMPLETE.json"
                )
                evaluation_root = common.EVALUATION / model_id
                evaluation_marker_path = evaluation_root / "RUN_COMPLETE.json"
                for label, path in (
                    ("training", training_marker_path),
                    ("evaluation", evaluation_marker_path),
                ):
                    if not path.is_file():
                        issues.append(f"missing {label}: {model_id}")
                        continue
                    payload = common.load_json(path)
                    if common.payload_sha256(payload) != payload.get("payload_sha256"):
                        issues.append(f"payload mismatch {label}: {model_id}")
                    if payload.get("status") != "PASS_COMPLETE":
                        issues.append(f"status mismatch {label}: {model_id}")
                    if label == "training":
                        training_cells += 1
                        checkpoint = Path(str(payload["checkpoint"]))
                        if (
                            common.sha256_file(checkpoint)
                            != payload["checkpoint_sha256"]
                        ):
                            issues.append(f"checkpoint mismatch: {model_id}")
                    else:
                        evaluation_cells += 1
                for tunnel in analysis.TUNNELS:
                    path = evaluation_root / f"fold_{tunnel}.json"
                    if not path.is_file():
                        issues.append(f"missing fold: {model_id}/{tunnel}")
                        continue
                    payload = common.load_json(path)
                    if common.payload_sha256(payload) != payload.get("payload_sha256"):
                        issues.append(f"fold payload mismatch: {model_id}/{tunnel}")
                    fold_files += 1
    summary_path = common.RUNS / "factorial_summary.json"
    rows_path = common.RUNS / "factorial_rows.csv"
    if not summary_path.is_file() or not rows_path.is_file():
        issues.append("missing factorial outputs")
        summary = {}
        rows = []
    else:
        summary = common.load_json(summary_path)
        if common.payload_sha256(summary) != summary.get("payload_sha256"):
            issues.append("factorial summary payload mismatch")
        with rows_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1008:
            issues.append(f"factorial row count mismatch: {len(rows)}")
    result: dict[str, Any] = {
        "schema_version": "MERIT-C051-AUDIT-v1",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "coverage": {
            "training_cells": training_cells,
            "evaluation_cells": evaluation_cells,
            "fold_files": fold_files,
            "factorial_rows": len(rows),
        },
        "protocol_payload_sha256": protocol["payload_sha256"],
        "summary_payload_sha256": summary.get("payload_sha256"),
    }
    result["payload_sha256"] = common.payload_sha256(result)
    common.write_json_atomic(common.RUNS / "independent_audit_v1.json", result)
    return result


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
