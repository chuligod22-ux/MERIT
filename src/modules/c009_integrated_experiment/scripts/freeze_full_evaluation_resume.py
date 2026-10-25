"""Freeze the operational-only recovery of the interrupted C009 evaluation queue."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
BASE_FREEZE = C009 / "protocol" / "full_evaluation_execution_freeze_v2.json"
INTERRUPTED_STATE = C009 / "runs" / "queue" / "full_evaluation_queue_status_v2.json"
ACTIVE_STATUS = (
    C009
    / "evaluation"
    / "wp6_d3_fullframe"
    / "wp1__restormer__base__seed2"
    / "status.json"
)
OUTPUT = C009 / "protocol" / "full_evaluation_resume_addendum_v1.json"
OVERRIDES = {
    "c009_full_evaluation_queue.py": (
        "Preserve completed_items and cumulative GPU time across an interrupted restart."
    ),
    "test_c009_full_evaluation.py": (
        "Add regression coverage for resume accounting, completion order, and startup safety."
    ),
}


def completed_training_gpu_seconds() -> float:
    total = 0.0
    for root in (C009 / "runs" / "wp1", C009 / "runs" / "wp2", C009 / "runs" / "wp3"):
        for manifest_path in root.glob("**/run_manifest.json"):
            manifest = common.load_json(manifest_path)
            if manifest.get("status") != "complete" or int(
                manifest.get("global_step", -1)
            ) != 8000:
                raise RuntimeError(f"Incomplete training manifest: {manifest_path}")
            total += float(manifest["elapsed_seconds"])
    return total


def main() -> None:
    base = common.load_json(BASE_FREEZE)
    existing = common.load_json(OUTPUT) if OUTPUT.exists() else None
    prior_addendum_hashes: list[str] = []
    if existing:
        interruption_state = existing["interruption_state"]
        prior_addendum_hashes = list(
            existing.get("accepted_prior_addendum_sha256s", [])
        )
        prior_addendum_hashes.append(common.sha256_file(OUTPUT))
        prior_addendum_hashes = list(dict.fromkeys(prior_addendum_hashes))
    else:
        state = common.load_json(INTERRUPTED_STATE)
        active = common.load_json(ACTIVE_STATUS)
        if state.get("status") != "running" or len(state.get("completed_items", [])) != 76:
            raise RuntimeError("Unexpected interrupted queue state")
        if state.get("active_item") != "wp1__restormer__base__seed2":
            raise RuntimeError("Unexpected interrupted queue item")
        if active.get("completed") != 467 or active.get("remaining") != 507:
            raise RuntimeError("Unexpected interrupted frame coverage")
        evaluation_seconds = float(state["evaluation_elapsed_seconds"])
        active_seconds = float(state["active_elapsed_seconds"])
        training_seconds = completed_training_gpu_seconds()
        recovered_evaluation_seconds = evaluation_seconds + active_seconds
        recovered_gpu_hours = (training_seconds + recovered_evaluation_seconds) / 3600.0
        if abs(recovered_gpu_hours - float(state["monitored_gpu_hours"])) > 1e-9:
            raise RuntimeError("Interrupted GPU accounting does not reconcile")
        interruption_state = {
            "path": str(INTERRUPTED_STATE.resolve()),
            "sha256": common.sha256_file(INTERRUPTED_STATE),
            "status": state["status"],
            "completed_items": len(state["completed_items"]),
            "active_block": state["active_block"],
            "active_item": state["active_item"],
            "evaluation_elapsed_seconds": evaluation_seconds,
            "active_elapsed_seconds": active_seconds,
            "recovered_evaluation_elapsed_seconds": recovered_evaluation_seconds,
            "training_gpu_seconds": training_seconds,
            "recovered_monitored_gpu_hours": recovered_gpu_hours,
            "hard_gpu_budget_hours": 240.0,
            "active_completed_frames": active["completed"],
            "active_remaining_frames": active["remaining"],
            "last_cohort_id": active["last_cohort_id"],
        }
    generator = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C009-FULL-EVALUATION-RESUME-ADDENDUM-v1",
        "status": "PASS_OPERATIONAL_RESUME_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Windows Update initiated a planned restart 17 seconds after the last atomic "
            "D3 frame commit. The original queue restart path discarded cumulative evaluation "
            "time and duplicated completion labels."
        ),
        "scientific_scope": {
            "changed": False,
            "statement": (
                "No model, checkpoint, seed, cohort, prompt, detector, endpoint, threshold, "
                "metric, or result-selection rule is changed."
            ),
            "result_access": (
                "Recovery was derived from queue state, completion markers, file hashes, and "
                "coverage counts without inspecting endpoint values."
            ),
        },
        "base_execution_freeze": {
            "path": str(BASE_FREEZE.resolve()),
            "sha256": common.sha256_file(BASE_FREEZE),
            "payload_sha256": base["payload_sha256"],
        },
        "interruption_state": interruption_state,
        "resume_contract": {
            "carry_active_elapsed_into_evaluation_elapsed": True,
            "preserve_completed_prefix": True,
            "validate_completed_marker_payloads": True,
            "skip_completed_items": True,
            "reject_duplicate_or_nonprefix_completion": True,
            "reject_nonfinite_or_negative_elapsed_time": True,
            "enforce_budget_before_child_launch": True,
            "first_missing_frame": 468,
        },
        "frozen_file_overrides": {
            filename: {
                "base_sha256": base["frozen_files"][filename],
                "resume_sha256": common.sha256_file(HERE / filename),
                "reason": reason,
            }
            for filename, reason in OVERRIDES.items()
        },
        "generator": {
            "path": str(generator),
            "sha256": common.sha256_file(generator),
        },
        "validation": {
            "command": "python -m pytest -q scripts/test_c009_full_evaluation.py",
            "required_checks": [
                "base freeze and resume addendum hashes",
                "141.0452304342217 GPU-hour recovery baseline",
                "exact completed-label prefix and marker payload hashes",
                "pre-launch 240 GPU-hour hard stop",
                "startup verification failure leaves the interrupted state untouched",
            ],
        },
    }
    if prior_addendum_hashes:
        payload["accepted_prior_addendum_sha256s"] = prior_addendum_hashes
    payload["payload_sha256"] = hashlib.sha256(common.canonical_bytes(payload)).hexdigest()
    common.write_json_atomic(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
