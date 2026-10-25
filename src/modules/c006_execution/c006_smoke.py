"""Run and verify the paired C006 task-OFF/task-ON smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.torch_version import TorchVersion


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_arm(arm: str, seed: int, steps: int) -> None:
    command = [
        sys.executable,
        str(HERE / "c006_train.py"),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--smoke",
        "--allow-overwrite",
        "--log-every",
        "1",
    ]
    subprocess.run(command, cwd=HERE, check=True)


def checkpoint_is_finite(path: Path, expected_step: int) -> bool:
    # The checkpoint is generated locally by c006_train.py. Keep weights_only=True
    # and allow only PyTorch's version wrapper used in our metadata.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["metadata"]["global_step"] != expected_step:
        return False
    return all(
        torch.isfinite(tensor).all().item()
        for tensor in checkpoint["state"].values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify preserved smoke outputs without rerunning either arm.",
    )
    args = parser.parse_args()

    if not args.verify_only:
        for arm in ("task_off", "task_on"):
            run_arm(arm, args.seed, args.steps)
    smoke_root = HERE / "runs" / "smoke"
    manifests = {
        arm: read_json(smoke_root / f"{arm}_seed{args.seed}" / "run_manifest.json")
        for arm in ("task_off", "task_on")
    }
    initial_state_match = (
        manifests["task_off"]["initial_state_sha256"]
        == manifests["task_on"]["initial_state_sha256"]
    )
    stream_match = (
        manifests["task_off"]["stream_prefix_sha256"]
        == manifests["task_on"]["stream_prefix_sha256"]
    )
    checkpoint_checks = {
        arm: checkpoint_is_finite(
            smoke_root / f"{arm}_seed{args.seed}" / "final.pt", args.steps
        )
        for arm in ("task_off", "task_on")
    }
    passed = initial_state_match and stream_match and all(checkpoint_checks.values())
    ledger = {
        "protocol_id": manifests["task_off"]["protocol_id"],
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "steps": args.steps,
        "initial_state_match": initial_state_match,
        "initial_state_sha256": manifests["task_off"]["initial_state_sha256"],
        "stream_match": stream_match,
        "stream_prefix_sha256": manifests["task_off"]["stream_prefix_sha256"],
        "checkpoint_finite_and_step_match": checkpoint_checks,
        "task_off_peak_cuda_memory_bytes": manifests["task_off"][
            "peak_cuda_memory_bytes"
        ],
        "task_on_peak_cuda_memory_bytes": manifests["task_on"][
            "peak_cuda_memory_bytes"
        ],
        "task_off_elapsed_seconds": manifests["task_off"]["elapsed_seconds"],
        "task_on_elapsed_seconds": manifests["task_on"]["elapsed_seconds"],
        "smoke_code_sha256": sha256_file(Path(__file__).resolve()),
        "train_code_sha256": sha256_file(HERE / "c006_train.py"),
        "passed": passed,
    }
    write_json(HERE / "smoke_ledger.json", ledger)
    print(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
