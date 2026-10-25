"""Wait for training, then run every frozen C009 evaluation block sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_FREEZE = C009 / "protocol" / "full_evaluation_execution_freeze_v2.json"
DEFAULT_RESUME_ADDENDUM = C009 / "protocol" / "full_evaluation_resume_addendum_v1.json"
DEFAULT_QUEUE = C009 / "runs" / "queue" / "full_evaluation_queue_status_v2.json"
DEFAULT_LOGS = C009 / "runs" / "queue" / "full_evaluation_logs_v2"
RESUME_OVERRIDE_FILES = {
    "c009_full_evaluation_queue.py",
    "test_c009_full_evaluation.py",
}


class QueueError(RuntimeError):
    pass


def now_kst() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def verify_resume_addendum(
    path: Path, base_path: Path, base_freeze: dict[str, Any]
) -> dict[str, Any]:
    addendum = common.load_json(path)
    if addendum.get("status") != "PASS_OPERATIONAL_RESUME_ONLY":
        raise QueueError("Full evaluation resume addendum is not PASS")
    payload = dict(addendum)
    recorded = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(common.canonical_bytes(payload)).hexdigest()
    if recorded != actual:
        raise QueueError("Full evaluation resume addendum payload mismatch")
    if set(addendum["frozen_file_overrides"]) != RESUME_OVERRIDE_FILES:
        raise QueueError("Resume addendum override scope is invalid")
    base_record = addendum["base_execution_freeze"]
    if base_record["sha256"] != common.sha256_file(base_path):
        raise QueueError("Resume addendum base execution freeze mismatch")
    for filename, record in addendum["frozen_file_overrides"].items():
        if base_freeze["frozen_files"].get(filename) != record["base_sha256"]:
            raise QueueError(f"Resume addendum base file mismatch: {filename}")
        if common.sha256_file(HERE / filename) != record["resume_sha256"]:
            raise QueueError(f"Resume addendum frozen file changed: {filename}")
    generator = addendum["generator"]
    if common.sha256_file(Path(generator["path"])) != generator["sha256"]:
        raise QueueError("Resume addendum generator changed")
    return addendum


def verify_freeze(
    path: Path, resume_addendum_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = common.load_json(path)
    if freeze.get("status") != "PASS_FROZEN_BEFORE_C009_ENDPOINT_RESULTS":
        raise QueueError("Full evaluation execution freeze is not PASS")
    payload = dict(freeze)
    recorded = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(common.canonical_bytes(payload)).hexdigest()
    if recorded != actual:
        raise QueueError("Full evaluation execution freeze payload mismatch")
    addendum = verify_resume_addendum(resume_addendum_path, path, freeze)
    overrides = addendum["frozen_file_overrides"]
    for filename, expected in freeze["frozen_files"].items():
        if filename in overrides:
            continue
        actual_file = common.sha256_file(HERE / filename)
        if actual_file != expected:
            raise QueueError(f"Frozen evaluation file changed: {filename}")
    return freeze, addendum


def prepare_state(
    previous: dict[str, Any] | None,
    execution_freeze_sha256: str,
    resume_addendum_sha256: str,
) -> dict[str, Any]:
    current_time = now_kst()
    previous = previous or {}
    elapsed_seconds = float(previous.get("evaluation_elapsed_seconds", 0.0))
    carried_active_seconds = float(previous.get("active_elapsed_seconds", 0.0))
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise QueueError("Interrupted evaluation elapsed time is invalid")
    if not math.isfinite(carried_active_seconds) or carried_active_seconds < 0.0:
        raise QueueError("Interrupted active elapsed time is invalid")
    if "hard_gpu_budget_hours" in previous and float(
        previous["hard_gpu_budget_hours"]
    ) != 240.0:
        raise QueueError("Interrupted evaluation hard budget is invalid")
    completed_items = list(previous.get("completed_items", []))
    if len(completed_items) != len(set(completed_items)):
        raise QueueError("Interrupted evaluation state has duplicate completed items")
    state: dict[str, Any] = {
        "schema_version": "MERIT-C009-FULL-EVALUATION-QUEUE-STATUS-v2",
        "status": "initializing",
        "queue_pid": os.getpid(),
        "started_at_kst": previous.get("started_at_kst", current_time),
        "execution_freeze_sha256": execution_freeze_sha256,
        "resume_addendum_sha256": resume_addendum_sha256,
        "contract_sha256": common.EXPECTED_CONTRACT_SHA256,
        "completed_items": completed_items,
        "deadline_state": previous.get(
            "deadline_state", "AT_RISK_PENDING_JOURNAL_EXTENSION"
        ),
        "hard_gpu_budget_hours": 240.0,
        "evaluation_elapsed_seconds": elapsed_seconds + carried_active_seconds,
        "active_elapsed_seconds": 0.0,
    }
    if previous:
        state.update(
            {
                "resumed_at_kst": current_time,
                "resume_count": int(previous.get("resume_count", 0)) + 1,
                "carried_active_elapsed_seconds": carried_active_seconds,
                "interrupted_active_block": previous.get("active_block"),
                "interrupted_active_item": previous.get("active_item"),
            }
        )
    return state


def record_completed(state: dict[str, Any], label: str) -> None:
    if label not in state["completed_items"]:
        state["completed_items"].append(label)


def validate_completed_prefix(state: dict[str, Any], expected: list[str]) -> None:
    completed = state["completed_items"]
    if completed != expected[: len(completed)]:
        raise QueueError("Interrupted evaluation completion order is invalid")


def completion_marker(label: str) -> tuple[Path, str]:
    block, item = label.split(":", 1)
    if block == "controlled":
        return C009 / "evaluation" / "controlled" / item / "RUN_COMPLETE.json", item
    if block == "wp2_prompt":
        seed = int(item.removeprefix("seed"))
        model_id = f"wp1__nafnet__merit_full__seed{seed}"
        return C009 / "evaluation" / "wp2_prompt" / model_id / "RUN_COMPLETE.json", model_id
    roots = {
        "wp5": "wp5_external_four_tunnel",
        "wp8": "wp8_printed_width",
        "wp6": "wp6_d3_fullframe",
    }
    if block not in roots:
        raise QueueError(f"Unknown completed evaluation block: {block}")
    return C009 / "evaluation" / roots[block] / item / "RUN_COMPLETE.json", item


def validate_completed_artifacts(labels: list[str]) -> None:
    for label in labels:
        marker_path, expected_model_id = completion_marker(label)
        if not marker_path.exists():
            raise QueueError(f"Completed evaluation marker is missing: {label}")
        marker = common.load_json(marker_path)
        payload = dict(marker)
        recorded = payload.pop("payload_sha256", None)
        actual = hashlib.sha256(common.canonical_bytes(payload)).hexdigest()
        if recorded != actual:
            raise QueueError(f"Completed evaluation marker hash mismatch: {label}")
        if marker.get("status") != "PASS":
            raise QueueError(f"Completed evaluation marker is not PASS: {label}")
        if marker.get("model_id") != expected_model_id:
            raise QueueError(f"Completed evaluation marker identity mismatch: {label}")
        if marker.get("contract_sha256") != common.EXPECTED_CONTRACT_SHA256:
            raise QueueError(f"Completed evaluation contract mismatch: {label}")


def owns_state_file(state_path: Path) -> bool:
    if not state_path.exists():
        return False
    state = common.load_json(state_path)
    return int(state.get("queue_pid", -1)) == os.getpid()


def update(state_path: Path, state: dict[str, Any], **values: Any) -> None:
    state.update(values)
    state["updated_at_kst"] = now_kst()
    common.write_json_atomic(state_path, state)


def run_command(
    command: list[str],
    log_path: Path,
    state_path: Path,
    state: dict[str, Any],
    block: str,
    item: str,
    training_gpu_seconds: float,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    consumed_before_start = training_gpu_seconds + float(
        state.get("evaluation_elapsed_seconds", 0.0)
    )
    if consumed_before_start >= float(state["hard_gpu_budget_hours"]) * 3600.0:
        raise QueueError("240 GPU-hour hard budget reached before evaluation launch")
    update(
        state_path,
        state,
        status="running",
        active_block=block,
        active_item=item,
        active_command=command,
        active_log=str(log_path.resolve()),
        active_started_at_kst=now_kst(),
        active_elapsed_seconds=0.0,
        monitored_gpu_hours=consumed_before_start / 3600.0,
    )
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"event": "start", "time": now_kst(), "command": command}) + "\n")
        handle.flush()
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=HERE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            active_seconds = time.perf_counter() - started
            consumed = (
                training_gpu_seconds
                + float(state.get("evaluation_elapsed_seconds", 0.0))
                + active_seconds
            )
            update(
                state_path,
                state,
                active_elapsed_seconds=active_seconds,
                monitored_gpu_hours=consumed / 3600.0,
            )
            if consumed >= float(state["hard_gpu_budget_hours"]) * 3600.0:
                process.terminate()
                process.wait(timeout=30)
                raise QueueError("240 GPU-hour hard budget reached during evaluation")
            time.sleep(30)
        returncode = int(process.returncode)
        elapsed = time.perf_counter() - started
        handle.write(json.dumps({"event": "exit", "time": now_kst(), "returncode": returncode}) + "\n")
    state["evaluation_elapsed_seconds"] = float(
        state.get("evaluation_elapsed_seconds", 0.0)
    ) + elapsed
    state["active_elapsed_seconds"] = 0.0
    if returncode != 0:
        raise QueueError(f"Evaluation command failed ({block}/{item}): {returncode}")


def completed_training_gpu_seconds() -> float:
    total = 0.0
    for root in (C009 / "runs" / "wp1", C009 / "runs" / "wp2", C009 / "runs" / "wp3"):
        for manifest_path in root.glob("**/run_manifest.json"):
            manifest = common.load_json(manifest_path)
            if manifest.get("status") != "complete" or int(manifest.get("global_step", -1)) != 8000:
                raise QueueError(f"Incomplete full-run manifest in GPU accounting: {manifest_path}")
            elapsed_seconds = float(manifest["elapsed_seconds"])
            if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
                raise QueueError(f"Invalid GPU time in training manifest: {manifest_path}")
            total += elapsed_seconds
    if not math.isfinite(total):
        raise QueueError("Invalid total training GPU time")
    return total


def wait_for_training(state_path: Path, state: dict[str, Any], poll_seconds: int) -> dict[str, Any]:
    while True:
        wp1_path = C009 / "runs" / "queue" / "full_queue_status.json"
        wp23_path = C009 / "runs" / "queue" / "wp23_queue_status.json"
        wp1 = common.load_json(wp1_path)
        wp23 = common.load_json(wp23_path)
        if wp1.get("status") == "failed" or wp23.get("status") == "failed":
            raise QueueError("Upstream training queue failed")
        update(
            state_path,
            state,
            status="waiting_for_training",
            wp1_observed_status=wp1.get("status"),
            wp23_observed_status=wp23.get("status"),
        )
        if wp1.get("status") == "complete" and wp23.get("status") == "complete":
            contract = common.load_contract()
            common.require_all_training_complete(contract)
            return contract
        time.sleep(poll_seconds)


def command(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(HERE / script), *arguments]


def execute(args: argparse.Namespace) -> None:
    freeze_path = args.freeze.resolve()
    resume_addendum_path = args.resume_addendum.resolve()
    freeze, addendum = verify_freeze(freeze_path, resume_addendum_path)
    state_path = args.state.resolve()
    logs = args.logs.resolve()
    previous: dict[str, Any] | None = None
    if state_path.exists():
        previous = common.load_json(state_path)
        if previous.get("status") == "complete":
            print(json.dumps(previous, ensure_ascii=False, indent=2))
            return
        if "resume_count" not in previous:
            expected_state = addendum["interruption_state"]["sha256"]
            if common.sha256_file(state_path) != expected_state:
                raise QueueError("Interrupted evaluation state does not match resume addendum")
    state = prepare_state(
        previous,
        common.sha256_file(freeze_path),
        common.sha256_file(resume_addendum_path),
    )
    common.write_json_atomic(state_path, state)
    contract = wait_for_training(state_path, state, args.poll_seconds)
    training_gpu_seconds = completed_training_gpu_seconds()
    update(
        state_path,
        state,
        training_gpu_seconds=training_gpu_seconds,
        evaluation_elapsed_seconds=float(state.get("evaluation_elapsed_seconds", 0.0)),
        monitored_gpu_hours=(
            training_gpu_seconds + float(state.get("evaluation_elapsed_seconds", 0.0))
        )
        / 3600.0,
    )
    wp1_specs = common.wp1_specifications(contract)
    follow_specs = common.wp23_specifications(contract)
    expected_labels = (
        [f"controlled:{common.model_id(value)}" for value in wp1_specs + follow_specs]
        + [f"wp2_prompt:seed{seed}" for seed in (0, 1, 2)]
        + [f"wp5:{common.model_id(value)}" for value in wp1_specs]
        + [f"wp8:{common.model_id(value)}" for value in wp1_specs]
        + [f"wp6:{common.model_id(value)}" for value in wp1_specs]
    )
    validate_completed_prefix(state, expected_labels)
    validate_completed_artifacts(state["completed_items"])

    for specification in wp1_specs + follow_specs:
        item = common.model_id(specification)
        label = f"controlled:{item}"
        if label in state["completed_items"]:
            continue
        extra = [] if specification["kind"] == "wp1" else ["--skip-ood"]
        run_command(
            command("c009_controlled_eval.py", "--model-id", item, *extra),
            logs / f"controlled__{item}.log",
            state_path,
            state,
            "controlled_wp1_wp3_and_ood",
            item,
            training_gpu_seconds,
        )
        record_completed(state, label)
        update(state_path, state, active_command=None, active_log=None)

    for seed in (0, 1, 2):
        item = f"seed{seed}"
        label = f"wp2_prompt:{item}"
        if label in state["completed_items"]:
            continue
        run_command(
            command("c009_prompt_eval.py", "--seed", str(seed)),
            logs / f"wp2_prompt__{item}.log",
            state_path,
            state,
            "wp2_prompt_perturbation",
            item,
            training_gpu_seconds,
        )
        record_completed(state, label)
        update(state_path, state, active_command=None, active_log=None)

    for specification in wp1_specs:
        item = common.model_id(specification)
        label = f"wp5:{item}"
        if label in state["completed_items"]:
            continue
        run_command(
            command("c009_d2_external_eval.py", "--model-id", item),
            logs / f"wp5_d2__{item}.log",
            state_path,
            state,
            "wp5_external_four_tunnel",
            item,
            training_gpu_seconds,
        )
        record_completed(state, label)
        update(state_path, state, active_command=None, active_log=None)

    for specification in wp1_specs:
        item = common.model_id(specification)
        label = f"wp8:{item}"
        if label in state["completed_items"]:
            continue
        run_command(
            command("c009_width_eval.py", "--model-id", item),
            logs / f"wp8_width__{item}.log",
            state_path,
            state,
            "wp8_printed_width",
            item,
            training_gpu_seconds,
        )
        record_completed(state, label)
        update(state_path, state, active_command=None, active_log=None)

    for specification in wp1_specs:
        item = common.model_id(specification)
        label = f"wp6:{item}"
        if label in state["completed_items"]:
            continue
        run_command(
            command("c009_d3_fullframe_eval.py", "--model-id", item),
            logs / f"wp6_d3__{item}.log",
            state_path,
            state,
            "wp6_different_tunnel_same_system_974",
            item,
            training_gpu_seconds,
        )
        record_completed(state, label)
        update(state_path, state, active_command=None, active_log=None)

    if state["completed_items"] != expected_labels:
        raise QueueError(
            "Full evaluation queue coverage or order mismatch: "
            f"{len(state['completed_items'])} != {len(expected_labels)}"
        )
    update(
        state_path,
        state,
        status="complete",
        active_block=None,
        active_item=None,
        active_command=None,
        active_log=None,
        completed_at_kst=now_kst(),
        completed_item_count=len(state["completed_items"]),
        frozen_run_count=freeze["queue_item_count"],
    )
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--resume-addendum", type=Path, default=DEFAULT_RESUME_ADDENDUM
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        execute(args)
    except Exception as error:
        state_path = args.state.resolve()
        if owns_state_file(state_path):
            state = common.load_json(state_path)
            update(
                state_path,
                state,
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                failed_at_kst=now_kst(),
            )
        raise


if __name__ == "__main__":
    main()
