"""Sequential supervisor for the 24 corrected C051 v2 evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import c051_common as common
import c051_v2_common as v2


def execute() -> dict[str, object]:
    protocol_v2 = v2.validate_protocol_v2()
    queue = [
        (backbone, arm, seed)
        for backbone in common.BACKBONES
        for seed in common.SEEDS
        for arm in common.ARMS
    ]
    status_path = common.RUNS / "evaluation_queue_status_v2.json"
    log_dir = common.RUNS / "evaluation_logs_v2"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    complete: list[str] = []
    for backbone, arm, seed in queue:
        model_id = common.model_id(backbone, arm, seed)
        marker = v2.EVALUATION_V2 / model_id / "RUN_COMPLETE.json"
        if marker.is_file():
            payload = common.load_json(marker)
            if payload.get("status") == "PASS_COMPLETE" and common.payload_sha256(
                payload
            ) == payload.get("payload_sha256"):
                complete.append(model_id)
                continue
        state = {
            "schema_version": "MERIT-C051-EVALUATION-QUEUE-v2",
            "status": "running",
            "active": model_id,
            "complete": complete,
            "total": len(queue),
            "elapsed_seconds": time.time() - started,
        }
        common.write_json_atomic(status_path, state)
        command = [
            sys.executable,
            str((Path(__file__).parent / "c051_evaluate_v2.py").resolve()),
            "--backbone",
            backbone,
            "--arm",
            arm,
            "--seed",
            str(seed),
        ]
        with (
            (log_dir / f"{model_id}.stdout.log").open("a", encoding="utf-8") as stdout,
            (log_dir / f"{model_id}.stderr.log").open("a", encoding="utf-8") as stderr,
        ):
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if process.returncode != 0:
            state.update({"status": "FAILED", "returncode": process.returncode})
            common.write_json_atomic(status_path, state)
            raise common.C051Error(f"Evaluation failed: {model_id}")
        complete.append(model_id)
    result: dict[str, object] = {
        "schema_version": "MERIT-C051-EVALUATION-QUEUE-v2",
        "status": "PASS_COMPLETE",
        "active": None,
        "complete": complete,
        "total": len(queue),
        "elapsed_seconds": time.time() - started,
        "evaluation_addendum_payload_sha256": protocol_v2["payload_sha256"],
    }
    result["payload_sha256"] = common.payload_sha256(result)
    common.write_json_atomic(status_path, result)
    return result


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(json.dumps(execute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
