#!/usr/bin/env python
"""Audit all frozen C017 training and evaluation artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
sys.path.insert(0, str(HERE))

import c017_evaluate as evaluation
import c017_models as models
import c017_train as training


def main() -> int:
    protocol = evaluation.verify_protocol()
    selection = evaluation.verify_selection()
    expected_units = {
        (fold, seed)
        for fold in protocol["data"]["tunnels"]
        for seed in (0, 1, 2)
    }
    training_records = []
    for model_name in models.MODEL_NAMES:
        for fold in protocol["data"]["tunnels"]:
            for seed in protocol["training"]["seeds"]:
                run_dir = (
                    C017
                    / "runs"
                    / "training_v1"
                    / model_name
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                )
                if not training.completed_run(run_dir):
                    raise RuntimeError(f"Incomplete or invalid training run: {run_dir}")
                record = training.load_json(run_dir / "run_final.json")
                if record["protocol_payload_sha256"] != protocol["payload_sha256"]:
                    raise RuntimeError(f"Training protocol mismatch: {run_dir}")
                training_records.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "seed": seed,
                        "checkpoint_sha256": record["checkpoint_sha256"],
                    }
                )

    observed_units = set()
    evaluation_records = []
    progress = C017 / "runs" / "evaluation_v1" / "progress"
    for path in sorted(progress.glob("fold_*_seed*.json")):
        value = training.load_json(path)
        if evaluation.payload_sha256(value) != value.get("payload_sha256"):
            raise RuntimeError(f"Evaluation payload hash mismatch: {path}")
        if value["protocol_payload_sha256"] != protocol["payload_sha256"]:
            raise RuntimeError(f"Evaluation protocol mismatch: {path}")
        key = (str(value["fold"]), int(value["seed"]))
        observed_units.add(key)
        evaluation_records.append(
            {"fold": key[0], "seed": key[1], "sha256": training.sha256_file(path)}
        )
    if observed_units != expected_units:
        raise RuntimeError(
            f"Evaluation coverage mismatch: expected={expected_units}, observed={observed_units}"
        )

    summary_path = C017 / "runs" / "evaluation_v1" / "summary.json"
    summary = training.load_json(summary_path)
    if evaluation.payload_sha256(summary) != summary.get("payload_sha256"):
        raise RuntimeError("Final summary payload hash mismatch")
    if summary["selection_payload_sha256"] != selection["payload_sha256"]:
        raise RuntimeError("Final summary selection mismatch")

    code_checks = {
        label: training.sha256_file(Path(record["path"])) == record["sha256"]
        for label, record in protocol["code"].items()
    }
    if not all(code_checks.values()):
        raise RuntimeError(f"Frozen code changed: {code_checks}")

    result = {
        "schema_version": "MERIT-C017-FINAL-AUDIT-v1",
        "status": "PASS",
        "training_runs": len(training_records),
        "evaluation_units": len(evaluation_records),
        "code_hash_checks": code_checks,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "selection_payload_sha256": selection["payload_sha256"],
        "summary_payload_sha256": summary["payload_sha256"],
        "efficacy_gate_pass": summary["efficacy_gate_pass"],
        "one_percentage_point_target_pass": summary[
            "one_percentage_point_target_pass"
        ],
        "training": training_records,
        "evaluation": evaluation_records,
    }
    result["payload_sha256"] = evaluation.payload_sha256(result)
    output = C017 / "runs" / "final_audit.json"
    evaluation.base.write_json_atomic(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
