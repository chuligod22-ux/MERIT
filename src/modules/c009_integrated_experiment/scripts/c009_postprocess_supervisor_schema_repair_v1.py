"""Run the C009 postprocess through the hash-bound D3 schema repair."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
ADDENDUM = C009 / "protocol" / "postprocess_schema_repair_addendum_v1.json"
STATUS = C009 / "runs" / "queue" / "postprocess_schema_repair_status_v1.json"


def now_kst() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def update(state: dict[str, Any], **values: Any) -> None:
    state.update(values)
    state["updated_at_kst"] = now_kst()
    common.write_json_atomic(STATUS, state)


def verify_addendum() -> dict[str, Any]:
    value = common.load_json(ADDENDUM)
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != hashlib.sha256(common.canonical_bytes(payload)).hexdigest():
        raise common.EvaluationError("Postprocess repair addendum payload mismatch")
    for relative, expected in value["immutable_evidence"].items():
        if common.sha256_file(C009 / relative) != expected:
            raise common.EvaluationError(
                f"Postprocess repair evidence changed: {relative}"
            )
    for filename, expected in value["repair_files"].items():
        if common.sha256_file(HERE / filename) != expected:
            raise common.EvaluationError(
                f"Postprocess repair file changed: {filename}"
            )
    return value


def main() -> None:
    addendum = verify_addendum()
    state: dict[str, Any] = {
        "schema_version": "MERIT-C009-POSTPROCESS-SCHEMA-REPAIR-STATUS-v1",
        "status": "starting",
        "pid": os.getpid(),
        "started_at_kst": now_kst(),
        "addendum_sha256": common.sha256_file(ADDENDUM),
    }
    common.write_json_atomic(STATUS, state)
    queue_path = C009 / "runs" / "queue" / "full_evaluation_queue_status_v2.json"
    queue = common.load_json(queue_path)
    if queue.get("status") != "complete" or int(
        queue.get("completed_item_count", -1)
    ) != 84:
        raise common.EvaluationError("Full evaluation queue is not complete at 84 items")

    commands = [
        [sys.executable, str(HERE / "c009_summarize_results_schema_repair_v1.py")],
        [sys.executable, str(HERE / "c009_generate_final_report.py")],
    ]
    for index, command in enumerate(commands, start=1):
        update(state, status="running", active_step=index, active_command=command)
        result = subprocess.run(
            command, cwd=HERE, text=True, capture_output=True, check=False
        )
        state[f"step_{index}_stdout"] = result.stdout
        state[f"step_{index}_stderr"] = result.stderr
        state[f"step_{index}_returncode"] = result.returncode
        common.write_json_atomic(STATUS, state)
        if result.returncode != 0:
            raise common.EvaluationError(
                f"Postprocess repair step {index} failed: {result.stderr}"
            )
    update(
        state,
        status="complete",
        active_step=None,
        active_command=None,
        completed_at_kst=now_kst(),
        summary_path=str(
            (C009 / "evaluation" / "final_summary" / "c009_final_summary_v3.json").resolve()
        ),
        report_path=str(
            (
                C009.parent.parent
                / "40_수정본"
                / "d5800_merit_c009_full_experiment_results.md"
            ).resolve()
        ),
        queue_item_count=addendum["required_queue_item_count"],
    )
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        state = common.load_json(STATUS) if STATUS.exists() else {}
        state.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_kst": now_kst(),
            }
        )
        common.write_json_atomic(STATUS, state)
        raise
