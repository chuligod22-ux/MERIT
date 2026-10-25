"""Sequential, fail-closed queue for C009 WP1 pilot or full training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
TRAINER = HERE / "c009_wp1_train.py"
DEFAULT_FREEZE = C009_ROOT / "protocol" / "wp1_execution_freeze_v1.json"
DEFAULT_PILOT_CLOSURE = C009_ROOT / "protocol" / "wp1_pilot_closure_v1.json"
DEFAULT_RUNS = C009_ROOT / "runs"


class QueueError(RuntimeError):
    """Raised when a queued run or paired invariant fails."""


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


def verify_freeze(path: Path) -> dict[str, Any]:
    freeze = load_json(path)
    payload = dict(freeze)
    recorded = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if recorded != actual or freeze.get("status") != "PASS":
        raise QueueError("Invalid WP1 execution freeze")
    expected = {
        Path(__file__).resolve(): freeze["frozen_files"]["c009_wp1_queue.py"],
        TRAINER.resolve(): freeze["frozen_files"]["c009_wp1_train.py"],
    }
    for file_path, expected_hash in expected.items():
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise QueueError(f"Frozen queue file changed: {file_path}")
    return freeze


def verify_pilot_closure(path: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    closure = load_json(path)
    payload = dict(closure)
    recorded = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if recorded != actual or closure.get("status") != "PASS":
        raise QueueError("WP1 pilot closure is not a valid PASS artifact")
    if closure.get("runner_sha256") != freeze["frozen_files"]["c009_wp1_train.py"]:
        raise QueueError("Pilot did not use the currently frozen training runner")
    required = {
        "six_runs_complete",
        "forced_resume_exercised_all_runs",
        "paired_initial_states_equal",
        "paired_trainable_parameters_equal",
        "paired_stream_hashes_equal",
        "checkpoint_and_manifest_hashes_valid",
    }
    if any(not closure.get("checks", {}).get(key) for key in required):
        raise QueueError("WP1 pilot closure is missing a required technical PASS")
    if closure.get("checks", {}).get("nonfinite_or_runtime_failures") != 0:
        raise QueueError("WP1 pilot closure records a runtime failure")
    return closure


def paired_check(run_root: Path, run_kind: str, family: str, seed: int) -> None:
    pair_dir = run_root / run_kind / family
    base = load_json(pair_dir / f"base_seed{seed}" / "run_manifest.json")
    full = load_json(pair_dir / f"merit_full_seed{seed}" / "run_manifest.json")
    checks = {
        "initial_state_sha256": (
            base["initial_state_sha256"], full["initial_state_sha256"]
        ),
        "trainable_parameters": (
            base["trainable_parameters"], full["trainable_parameters"]
        ),
        "stream_sha256": (base["stream_sha256"], full["stream_sha256"]),
        "global_step": (base["global_step"], full["global_step"]),
        "contract_sha256": (base["contract_sha256"], full["contract_sha256"]),
    }
    mismatches = {key: value for key, value in checks.items() if value[0] != value[1]}
    if mismatches:
        raise QueueError(
            f"Paired invariant mismatch for {family}/seed{seed}: {mismatches}"
        )
    result: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP1-PAIRED-CHECK-v1",
        "status": "PASS",
        "family": family,
        "seed": seed,
        "checks": {key: value[0] for key, value in checks.items()},
        "checked_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    result["payload_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    write_json_atomic(pair_dir / f"paired_seed{seed}_PASS.json", result)


def run_one(
    command: list[str],
    log_path: Path,
    queue_state: dict[str, Any],
    state_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    queue_state["active_command"] = command
    queue_state["active_log"] = str(log_path)
    queue_state["updated_at_kst"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    write_json_atomic(state_path, queue_state)
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
        raise QueueError(f"Training subprocess failed ({result.returncode}): {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--execution-freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--pilot-closure", type=Path, default=DEFAULT_PILOT_CLOSURE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()

    freeze_path = args.execution_freeze.resolve()
    freeze = verify_freeze(freeze_path)
    pilot_closure_sha256 = None
    if args.mode == "full":
        pilot_closure = verify_pilot_closure(args.pilot_closure.resolve(), freeze)
        pilot_closure_sha256 = sha256_file(args.pilot_closure.resolve())
    run_root = args.run_root.resolve()
    queue_dir = run_root / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    state_path = queue_dir / f"{args.mode}_queue_status.json"
    stdout_dir = queue_dir / f"{args.mode}_logs"
    families = ("nafnet", "restormer", "mprnet")
    seeds = (0,) if args.mode == "pilot" else (0, 1, 2)
    total_runs = len(families) * len(seeds) * 2
    queue_state: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP1-QUEUE-STATUS-v1",
        "status": "running",
        "mode": args.mode,
        "queue_pid": __import__("os").getpid(),
        "started_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "execution_freeze_sha256": sha256_file(freeze_path),
        "run_matrix_sha256": freeze["run_matrix_sha256"],
        "pilot_closure_sha256": pilot_closure_sha256,
        "total_runs": total_runs,
        "completed_runs": 0,
        "completed": [],
        "deadline_state": "AT_RISK_PENDING_JOURNAL_EXTENSION",
    }
    write_json_atomic(state_path, queue_state)
    queue_started = time.perf_counter()
    try:
        for family in families:
            for seed in seeds:
                for arm in ("base", "merit_full"):
                    run_key = f"{family}/{arm}_seed{seed}"
                    common = [
                        sys.executable,
                        str(TRAINER),
                        "--family",
                        family,
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--execution-freeze",
                        str(freeze_path),
                        "--output-dir",
                        str(run_root),
                    ]
                    if args.mode == "pilot":
                        first = common + [
                            "--pilot",
                            "--steps",
                            "20",
                            "--stop-after-step",
                            "5",
                            "--log-every",
                            "5",
                        ]
                        run_one(
                            first,
                            stdout_dir / f"{family}_{arm}_seed{seed}_part1.log",
                            queue_state,
                            state_path,
                        )
                        second = common + [
                            "--pilot",
                            "--steps",
                            "20",
                            "--log-every",
                            "5",
                        ]
                        run_one(
                            second,
                            stdout_dir / f"{family}_{arm}_seed{seed}_part2.log",
                            queue_state,
                            state_path,
                        )
                    else:
                        command = common + ["--log-every", "100"]
                        run_one(
                            command,
                            stdout_dir / f"{family}_{arm}_seed{seed}.log",
                            queue_state,
                            state_path,
                        )
                    queue_state["completed"].append(run_key)
                    queue_state["completed_runs"] = len(queue_state["completed"])
                    queue_state["elapsed_seconds"] = time.perf_counter() - queue_started
                    queue_state["active_command"] = None
                    queue_state["active_log"] = None
                    queue_state["updated_at_kst"] = datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    )
                    write_json_atomic(state_path, queue_state)
                paired_check(run_root, "pilot" if args.mode == "pilot" else "wp1", family, seed)
        queue_state.update(
            {
                "status": "complete",
                "completed_at_kst": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "elapsed_seconds": time.perf_counter() - queue_started,
                "active_command": None,
                "active_log": None,
            }
        )
        write_json_atomic(state_path, queue_state)
    except Exception as error:
        queue_state.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_kst": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "elapsed_seconds": time.perf_counter() - queue_started,
            }
        )
        write_json_atomic(state_path, queue_state)
        raise
    print(json.dumps(queue_state, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
