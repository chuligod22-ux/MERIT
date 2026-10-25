"""Freeze the result-neutral C009 D3 postprocess schema repair."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
OUTPUT = C009 / "protocol" / "postprocess_schema_repair_addendum_v1.json"


def main() -> None:
    immutable = [
        "protocol/postprocess_execution_freeze_v3.json",
        "runs/queue/full_evaluation_queue_status_v2.json",
        "runs/queue/postprocess_status_v3.json",
        "scripts/c009_summarize_results.py",
        "scripts/c009_generate_final_report.py",
    ]
    repair_files = [
        "c009_summarize_results_schema_repair_v1.py",
        "c009_postprocess_supervisor_schema_repair_v1.py",
        "test_c009_postprocess_schema_repair.py",
    ]
    payload = {
        "schema_version": "MERIT-C009-POSTPROCESS-SCHEMA-REPAIR-ADDENDUM-v1",
        "status": "PASS_FROZEN_AFTER_IMPLEMENTATION_FAILURE_BEFORE_FINAL_SUMMARY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "required_queue_item_count": 84,
        "result_access_disclosure": (
            "All 84 evaluations were complete when the frozen v3 summarizer raised "
            "KeyError before writing the final summary. The repair is restricted to "
            "the pre-existing label-specific endpoint schema and changes no model, "
            "cohort, threshold, endpoint, gate, bootstrap rule, or global-verdict rule."
        ),
        "failure": {
            "error": "KeyError: any_eligible_component",
            "cause": (
                "The v3 D3 summarizer requested no_visible_crack component fields "
                "from crack_visible frames."
            ),
        },
        "field_dispatch": {
            "crack_visible": "frame_roi_hit_fraction",
            "no_visible_crack_positive_rate": "any_eligible_component",
            "no_visible_crack_burden": "components_per_megapixel",
        },
        "scientific_contract_change": "NONE",
        "immutable_evidence": {
            relative: common.sha256_file(C009 / relative) for relative in immutable
        },
        "repair_files": {
            filename: common.sha256_file(HERE / filename) for filename in repair_files
        },
    }
    payload["payload_sha256"] = hashlib.sha256(
        common.canonical_bytes(payload)
    ).hexdigest()
    common.write_json_atomic(OUTPUT, payload)
    print(OUTPUT)
    print(payload["payload_sha256"])


if __name__ == "__main__":
    main()
