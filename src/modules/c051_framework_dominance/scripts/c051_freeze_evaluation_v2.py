"""Freeze the C051 v2 scale-correction addendum before corrected evaluation."""

from __future__ import annotations

import json

import c051_common as common
import c051_v2_common as v2

CODE_FILES = (
    "c051_v2_common.py",
    "c051_evaluate_v2.py",
    "c051_evaluation_queue_v2.py",
    "c051_analyze_v2.py",
    "c051_audit_v2.py",
    "test_c051_v2.py",
)


def execute() -> dict[str, object]:
    if v2.PROTOCOL_V2.exists():
        raise common.C051Error(f"Refusing to overwrite frozen addendum: {v2.PROTOCOL_V2}")
    protocol_v1 = common.validate_protocol()
    summary_v1_path = common.RUNS / "factorial_summary.json"
    audit_v1_path = common.RUNS / "independent_audit_v1.json"
    summary_v1 = common.load_json(summary_v1_path)
    audit_v1 = common.load_json(audit_v1_path)
    invalidation: dict[str, object] = {
        "schema_version": "MERIT-C051-EVALUATION-V1-INVALIDATION-v1",
        "status": "INVALIDATED_INPUT_SCALE_CONTRACT",
        "discovered_date": "2026-10-19",
        "reason": (
            "c051_evaluate.py returned restored float32 in [0,1], while the frozen "
            "external.detect_batch contract divides every candidate by 255. Restored "
            "images alone were therefore normalized twice."
        ),
        "affected_scope": "all evaluation_v1 folds, factorial_rows.csv, factorial_summary.json, and their gate result",
        "unaffected_scope": "all 24 fixed-final training checkpoints and the degraded-input detector results",
        "invalid_summary_payload_sha256": summary_v1["payload_sha256"],
        "invalid_audit_payload_sha256": audit_v1["payload_sha256"],
        "invalid_evaluator_sha256": common.sha256_file(common.HERE / "c051_evaluate.py"),
        "resolution": "Preserve v1; rerun all 24x4 folds in evaluation_v2 after explicit round-and-clip conversion to uint8 [0,255].",
    }
    invalidation["payload_sha256"] = common.payload_sha256(invalidation)
    common.write_json_atomic(common.RUNS / "evaluation_v1_invalidation.json", invalidation)

    payload: dict[str, object] = {
        "schema_version": "MERIT-C051-EVALUATION-ADDENDUM-v2",
        "status": "FROZEN_BEFORE_CORRECTED_EVALUATION_V2",
        "created_date": "2026-10-19",
        "supersedes_protocol_payload_sha256": protocol_v1["payload_sha256"],
        "v1_invalidation_payload_sha256": invalidation["payload_sha256"],
        "correction": "Convert finite restorer output from float [0,1] to round(clip(x,0,1)*255) uint8 before the unchanged detector preprocessing.",
        "unchanged": [
            "24 trained fixed-final checkpoints",
            "four backbones, two arms, and three seeds",
            "frozen four-tunnel cohort and degradation",
            "three detector families and checkpoint pairing",
            "physical-parent aggregation, estimands, bootstrap, and gates",
        ],
        "forbidden": [
            "retraining or checkpoint selection",
            "changing cohort, detector, endpoint, aggregation, or gates",
            "using the invalid v1 gate result as evidence",
        ],
        "output_root": str(v2.EVALUATION_V2.resolve()),
        "code_sha256": {
            name: common.sha256_file(common.HERE / name) for name in CODE_FILES
        },
        "environment": protocol_v1["environment"],
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_atomic(v2.PROTOCOL_V2, payload)
    return payload


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
