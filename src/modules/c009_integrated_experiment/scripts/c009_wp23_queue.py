"""Wait for WP1, close a forced-resume pilot, then run all WP2/WP3 additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
TRAINER = HERE / "c009_wp23_train.py"
DEFAULT_FREEZE = C009_ROOT / "protocol" / "wp23_execution_freeze_v1.json"
DEFAULT_PROTOCOL = C009_ROOT / "protocol" / "wp23_training_contract_v1.json"
DEFAULT_RUNS = C009_ROOT / "runs"
DEFAULT_PILOT_CLOSURE = C009_ROOT / "protocol" / "wp23_pilot_closure_v1.json"


class FollowonQueueError(RuntimeError):
    """Raised when WP2/WP3 cannot proceed safely."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def verify_payload(document: dict[str, Any]) -> None:
    payload = dict(document)
    recorded = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if recorded != actual:
        raise FollowonQueueError("Internal payload hash mismatch")


def verify_freeze(path: Path) -> dict[str, Any]:
    freeze = load_json(path)
    verify_payload(freeze)
    if freeze.get("status") != "PASS":
        raise FollowonQueueError("WP23 execution freeze is not PASS")
    expected = {
        Path(__file__).resolve(): freeze["frozen_files"]["c009_wp23_queue.py"],
        TRAINER.resolve(): freeze["frozen_files"]["c009_wp23_train.py"],
        HERE / "c009_wp23_models.py": freeze["frozen_files"]["c009_wp23_models.py"],
        DEFAULT_PROTOCOL: freeze["frozen_files"]["wp23_training_contract_v1.json"],
    }
    for file_path, expected_hash in expected.items():
        if sha256_file(file_path) != expected_hash:
            raise FollowonQueueError(f"Frozen follow-on file changed: {file_path}")
    return freeze


def run_one(command: list[str], log_path: Path, state: dict[str, Any], state_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state["active_command"] = command
    state["active_log"] = str(log_path)
    state["updated_at_kst"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json_atomic(state_path, state)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "invocation_start",
                    "at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "command": command,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(C009_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(
            json.dumps(
                {
                    "event": "invocation_end",
                    "at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "returncode": result.returncode,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    if result.returncode != 0:
        raise FollowonQueueError(f"Training subprocess failed: {log_path}")


def wait_for_wp1(run_root: Path, state: dict[str, Any], state_path: Path, poll_seconds: int) -> None:
    wp1_status_path = run_root / "queue" / "full_queue_status.json"
    while True:
        if not wp1_status_path.exists():
            wp1_status = None
        else:
            wp1_status = load_json(wp1_status_path)
        status = None if wp1_status is None else wp1_status.get("status")
        if status == "complete":
            if int(wp1_status.get("completed_runs", -1)) != 18:
                raise FollowonQueueError("WP1 reports complete without 18 runs")
            state["wp1_queue_sha256"] = sha256_file(wp1_status_path)
            state["status"] = "pilot_running"
            state["updated_at_kst"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            write_json_atomic(state_path, state)
            return
        if status == "failed":
            raise FollowonQueueError("WP1 queue failed; WP2/WP3 remain blocked")
        state["status"] = "waiting_for_wp1"
        state["wp1_observed_status"] = status
        state["updated_at_kst"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        write_json_atomic(state_path, state)
        time.sleep(poll_seconds)


def close_pilot(run_root: Path, freeze: dict[str, Any], output_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    stream_hashes: set[str] = set()
    trainer_hashes: set[str] = set()
    for wp, structure in (
        ("wp2", "zero_prompt_full"),
        ("wp3", "cascade"),
        ("wp3", "parallel"),
    ):
        run_dir = run_root / "pilot23" / f"{structure}_seed0"
        if (run_dir / "failure.json").exists():
            raise FollowonQueueError(f"Pilot failure remains: {run_dir}")
        manifest_path = run_dir / "run_manifest.json"
        marker_path = run_dir / "RUN_COMPLETE.json"
        final_path = run_dir / "final.pt"
        manifest = load_json(manifest_path)
        marker = load_json(marker_path)
        verify_payload(marker)
        if manifest.get("status") != "complete" or marker.get("status") != "PASS":
            raise FollowonQueueError(f"Pilot is incomplete: {run_dir}")
        if int(manifest.get("global_step", -1)) != 20:
            raise FollowonQueueError(f"Pilot final step is not 20: {run_dir}")
        if int(manifest.get("resume_count", 0)) < 1:
            raise FollowonQueueError(f"Pilot did not resume: {run_dir}")
        if marker["final_checkpoint_sha256"] != sha256_file(final_path):
            raise FollowonQueueError(f"Pilot checkpoint hash mismatch: {run_dir}")
        stream_hashes.add(str(manifest["stream_sha256"]))
        trainer_hashes.add(str(manifest["runner_sha256"]))
        records.append(
            {
                "wp": wp,
                "structure": structure,
                "seed": 0,
                "parameters": manifest["parameters"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "peak_cuda_memory_bytes": manifest["peak_cuda_memory_bytes_this_invocation"],
                "resume_count": manifest["resume_count"],
                "stream_sha256": manifest["stream_sha256"],
                "final_checkpoint_sha256": marker["final_checkpoint_sha256"],
            }
        )
    if len(stream_hashes) != 1:
        raise FollowonQueueError("WP23 pilot structures did not use the same stream")
    if trainer_hashes != {freeze["frozen_files"]["c009_wp23_train.py"]}:
        raise FollowonQueueError("WP23 pilot trainer hash mismatch")
    closure: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP23-PILOT-CLOSURE-v1",
        "status": "PASS",
        "created_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trainer_sha256": next(iter(trainer_hashes)),
        "execution_freeze_sha256": freeze["payload_sha256"],
        "checks": {
            "three_structures_complete": True,
            "forced_resume_exercised": True,
            "shared_stream_hash": next(iter(stream_hashes)),
            "fixed_step_20": True,
            "checkpoint_hashes_valid": True,
            "finite_runtime_failures": 0,
        },
        "records": records,
        "interpretation": "Technical readiness only; no model selection or efficacy inference.",
    }
    closure["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(closure)
    ).hexdigest()
    write_json_atomic(output_path, closure)
    return closure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--pilot-closure", type=Path, default=DEFAULT_PILOT_CLOSURE)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise SystemExit("poll-seconds must be at least 10")

    freeze_path = args.execution_freeze.resolve()
    freeze = verify_freeze(freeze_path)
    run_root = args.run_root.resolve()
    queue_dir = run_root / "queue"
    state_path = queue_dir / "wp23_queue_status.json"
    logs = queue_dir / "wp23_logs"
    state: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP23-QUEUE-STATUS-v1",
        "status": "waiting_for_wp1",
        "queue_pid": os.getpid(),
        "started_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "execution_freeze_sha256": sha256_file(freeze_path),
        "total_additional_full_runs": 9,
        "completed_full_runs": [],
        "deadline_state": "AT_RISK_PENDING_JOURNAL_EXTENSION",
    }
    write_json_atomic(state_path, state)
    started = time.perf_counter()
    try:
        wait_for_wp1(run_root, state, state_path, args.poll_seconds)
        for wp, structure in (
            ("wp2", "zero_prompt_full"),
            ("wp3", "cascade"),
            ("wp3", "parallel"),
        ):
            common = [
                sys.executable,
                str(TRAINER),
                "--wp",
                wp,
                "--structure",
                structure,
                "--seed",
                "0",
                "--pilot",
                "--steps",
                "20",
                "--execution-freeze",
                str(freeze_path),
                "--output-dir",
                str(run_root),
                "--wp1-run-root",
                str(run_root),
                "--log-every",
                "5",
            ]
            run_one(
                common + ["--stop-after-step", "5"],
                logs / f"pilot_{structure}_part1.log",
                state,
                state_path,
            )
            run_one(
                common,
                logs / f"pilot_{structure}_part2.log",
                state,
                state_path,
            )
        closure_path = args.pilot_closure.resolve()
        closure = close_pilot(run_root, freeze, closure_path)
        state["pilot_closure_sha256"] = sha256_file(closure_path)
        state["status"] = "full_running"
        state["updated_at_kst"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        write_json_atomic(state_path, state)

        matrix = load_json(DEFAULT_PROTOCOL)["run_matrix"]
        for row in matrix:
            wp = str(row["wp"])
            structure = str(row["structure"])
            seed = int(row["seed"])
            command = [
                sys.executable,
                str(TRAINER),
                "--wp",
                wp,
                "--structure",
                structure,
                "--seed",
                str(seed),
                "--execution-freeze",
                str(freeze_path),
                "--pilot-closure",
                str(closure_path),
                "--output-dir",
                str(run_root),
                "--wp1-run-root",
                str(run_root),
                "--log-every",
                "100",
            ]
            run_one(
                command,
                logs / f"full_{wp}_{structure}_seed{seed}.log",
                state,
                state_path,
            )
            state["completed_full_runs"].append(
                f"{wp}/{structure}_seed{seed}"
            )
            state["elapsed_seconds"] = time.perf_counter() - started
            state["active_command"] = None
            state["active_log"] = None
            state["updated_at_kst"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            write_json_atomic(state_path, state)
        state.update(
            {
                "status": "complete",
                "completed_at_kst": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "active_command": None,
                "active_log": None,
            }
        )
        write_json_atomic(state_path, state)
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_kst": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json_atomic(state_path, state)
        raise
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
