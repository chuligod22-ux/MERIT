#!/usr/bin/env python
"""Resume and complete every frozen C017 stage in dependency order."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
STATUS = C017 / "runs" / "controller_status.json"
TRAINING_SUMMARY = C017 / "runs" / "training_v1" / "summary.json"


def write_status(stage: str, state: str, detail: str | None = None) -> None:
    payload = {
        "schema_version": "MERIT-C017-CONTROLLER-v1",
        "stage": stage,
        "state": state,
        "detail": detail,
        "updated_unix_time": time.time(),
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, STATUS)


def run(stage: str, script: str, *arguments: str) -> None:
    write_status(stage, "RUNNING")
    command = [sys.executable, str(HERE / script), *arguments]
    try:
        subprocess.run(command, check=True)
    except BaseException as error:
        write_status(stage, "FAILED", repr(error))
        raise
    write_status(stage, "COMPLETE")


def wait_for_existing_training() -> None:
    training_root = C017 / "runs" / "training_v1"
    while not TRAINING_SUMMARY.is_file():
        completed = len(list(training_root.glob("*/fold_*/seed_*/run_final.json")))
        write_status(
            "waiting_for_training",
            "RUNNING",
            f"completed_runs={completed}/24",
        )
        time.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--follow-existing-training",
        action="store_true",
        help="Wait for an already running training process, then execute evaluation.",
    )
    args = parser.parse_args()
    stages = [
        ("training", "c017_train.py", "run-all"),
        ("seed0_screen", "c017_evaluate.py", "screen"),
        ("selection_freeze", "c017_evaluate.py", "select"),
        ("held_seed_replication", "c017_evaluate.py", "replicate"),
        ("final_summary", "c017_evaluate.py", "summarize"),
    ]
    if args.follow_existing_training:
        wait_for_existing_training()
        stages = stages[1:]
    for stage in stages:
        run(*stage)
    write_status("all", "PASS_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
