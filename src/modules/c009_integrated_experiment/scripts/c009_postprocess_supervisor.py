"""Wait for the 84-item evaluation queue, then summarize and render the report."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
FREEZE = C009 / "protocol" / "postprocess_execution_freeze_v3.json"
STATUS = C009 / "runs" / "queue" / "postprocess_status_v3.json"


def now_kst() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def update(state: dict[str, Any], **values: Any) -> None:
    state.update(values)
    state["updated_at_kst"] = now_kst()
    common.write_json_atomic(STATUS, state)


def verify_freeze() -> dict[str, Any]:
    value = common.load_json(FREEZE)
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != hashlib.sha256(common.canonical_bytes(payload)).hexdigest():
        raise common.EvaluationError("Postprocess freeze payload mismatch")
    for filename, expected in value["frozen_files"].items():
        if common.sha256_file(HERE / filename) != expected:
            raise common.EvaluationError(f"Postprocess frozen file changed: {filename}")
    return value


def main() -> None:
    freeze = verify_freeze()
    state: dict[str, Any] = {
        "schema_version": "MERIT-C009-POSTPROCESS-STATUS-v3",
        "status": "waiting_for_full_evaluation",
        "pid": os.getpid(),
        "started_at_kst": now_kst(),
        "freeze_sha256": common.sha256_file(FREEZE),
    }
    common.write_json_atomic(STATUS, state)
    queue_path = C009 / "runs" / "queue" / "full_evaluation_queue_status_v2.json"
    while True:
        queue = common.load_json(queue_path)
        if queue.get("status") == "failed":
            raise common.EvaluationError("Full evaluation queue failed")
        update(state, full_evaluation_status=queue.get("status"))
        if queue.get("status") == "complete":
            if int(queue.get("completed_item_count", -1)) != 84:
                raise common.EvaluationError("Full evaluation queue has incomplete coverage")
            break
        time.sleep(60)
    commands = [
        [sys.executable, str(HERE / "c009_summarize_results.py")],
        [sys.executable, str(HERE / "c009_generate_final_report.py")],
    ]
    for index, command in enumerate(commands, start=1):
        update(state, status="running", active_step=index, active_command=command)
        result = subprocess.run(command, cwd=HERE, text=True, capture_output=True, check=False)
        state[f"step_{index}_stdout"] = result.stdout
        state[f"step_{index}_stderr"] = result.stderr
        state[f"step_{index}_returncode"] = result.returncode
        common.write_json_atomic(STATUS, state)
        if result.returncode != 0:
            raise common.EvaluationError(f"Postprocess step {index} failed: {result.stderr}")
    update(
        state,
        status="complete",
        active_step=None,
        active_command=None,
        completed_at_kst=now_kst(),
        summary_path=str((C009 / "evaluation" / "final_summary" / "c009_final_summary_v3.json").resolve()),
        report_path=str((C009.parent.parent / "40_수정본" / "d5800_merit_c009_full_experiment_results.md").resolve()),
        queue_item_count=freeze["required_queue_item_count"],
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
