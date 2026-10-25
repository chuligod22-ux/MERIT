"""Sequential operational runner for the six frozen C006 final runs.

This script does not define scientific computation.  It verifies the frozen
training-code hash, waits for an already-running first run when requested, and
then invokes c006_train.py one run at a time.  It stops on the first failure.
"""

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
PREFLIGHT_PATH = HERE / "c006_preflight_freeze.json"
TRAIN_PATH = HERE / "c006_train.py"
FINAL_ROOT = HERE / "runs" / "final"
STATUS_PATH = HERE / "queue_status.json"
RUNS = (
    ("task_off", 0),
    ("task_on", 0),
    ("task_off", 1),
    ("task_on", 1),
    ("task_off", 2),
    ("task_on", 2),
)


def now_kst() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_status(value: dict[str, Any]) -> None:
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, STATUS_PATH)


def completed_run(arm: str, seed: int) -> bool:
    run_dir = FINAL_ROOT / f"{arm}_seed{seed}"
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "final.pt"
    if not manifest_path.exists() or not checkpoint_path.exists():
        return False
    manifest = read_json(manifest_path)
    return (
        manifest.get("status") == "complete"
        and manifest.get("steps") == 8000
        and manifest.get("final_checkpoint_sha256") == sha256_file(checkpoint_path)
    )


def wait_for_existing_run(arm: str, seed: int, timeout_seconds: int) -> None:
    run_dir = FINAL_ROOT / f"{arm}_seed{seed}"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if completed_run(arm, seed):
            return
        if (run_dir / "failure.json").exists():
            raise RuntimeError(f"existing run failed: {arm}_seed{seed}")
        time.sleep(10)
    raise TimeoutError(f"existing run timeout: {arm}_seed{seed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-first", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    preflight = read_json(PREFLIGHT_PATH)
    expected_train_hash = preflight["train_code_sha256"]
    if sha256_file(TRAIN_PATH) != expected_train_hash:
        raise RuntimeError("frozen c006_train.py hash mismatch")

    status: dict[str, Any] = {
        "schema_version": "MERIT-C006-RUN-QUEUE-v1",
        "started_at_kst": now_kst(),
        "status": "running",
        "pid": os.getpid(),
        "python": sys.executable,
        "train_code_sha256": expected_train_hash,
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "completed_runs": [],
        "active_run": None,
        "d3_one_shot": "blocked",
    }
    write_status(status)

    try:
        for index, (arm, seed) in enumerate(RUNS):
            run_name = f"{arm}_seed{seed}"
            status["active_run"] = run_name
            status["updated_at_kst"] = now_kst()
            write_status(status)

            if completed_run(arm, seed):
                status["completed_runs"].append(run_name)
                continue
            if index == 0 and args.wait_for_first:
                wait_for_existing_run(arm, seed, args.wait_timeout_seconds)
            else:
                if (FINAL_ROOT / run_name / "run_manifest.json").exists():
                    raise RuntimeError(f"incomplete existing run directory: {run_name}")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(TRAIN_PATH),
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--log-every",
                        "100",
                    ],
                    cwd=HERE,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"training exited {result.returncode}: {run_name}")
            if not completed_run(arm, seed):
                raise RuntimeError(f"completion verification failed: {run_name}")
            status["completed_runs"].append(run_name)
            status["updated_at_kst"] = now_kst()
            write_status(status)

        status["status"] = "complete"
        status["active_run"] = None
        status["completed_at_kst"] = now_kst()
        write_status(status)
    except BaseException as error:
        status["status"] = "failed"
        status["failed_at_kst"] = now_kst()
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        write_status(status)
        raise


if __name__ == "__main__":
    main()
