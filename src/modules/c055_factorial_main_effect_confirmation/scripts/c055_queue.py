"""Sequential resumable queue for the 32 frozen C055 confirmation units."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import c055_common as analysis

HERE = Path(__file__).resolve().parent
C055 = HERE.parent
RUNS = C055 / "runs"
LOGS = RUNS / "logs"
STATUS = RUNS / "confirmation_queue_status.json"


def output_path(seed: int, backbone: str, fold: str) -> Path:
    return RUNS / f"confirmation_seed{seed}" / backbone / f"fold_{fold}.json"


def execute() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    units = [
        (seed, backbone, fold)
        for seed in (1, 2)
        for backbone in analysis.BACKBONES
        for fold in analysis.TUNNELS
    ]
    complete = []
    started = time.perf_counter()
    for seed, backbone, fold in units:
        name = f"seed{seed}__{backbone}__{fold}"
        if output_path(seed, backbone, fold).exists():
            complete.append(name)
            continue
        analysis.common.write_json_atomic(
            STATUS,
            {"status": "running", "active": name, "complete": complete, "total": len(units)},
        )
        command = [
            sys.executable,
            str(HERE / "c055_confirm.py"),
            "--seed",
            str(seed),
            "--backbone",
            backbone,
            "--fold",
            fold,
        ]
        with (LOGS / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (
            LOGS / f"{name}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if result.returncode != 0:
            analysis.common.write_json_atomic(
                STATUS,
                {
                    "status": "FAILED",
                    "active": name,
                    "returncode": result.returncode,
                    "complete": complete,
                    "total": len(units),
                },
            )
            raise SystemExit(result.returncode)
        complete.append(name)
    analysis.common.write_json_atomic(
        STATUS,
        {
            "status": "PASS_COMPLETE",
            "active": None,
            "complete": complete,
            "total": len(units),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    execute()
