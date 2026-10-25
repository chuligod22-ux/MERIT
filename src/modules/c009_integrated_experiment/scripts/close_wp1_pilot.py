"""Validate and close the six-run C009 WP1 checkpoint/resume pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
DEFAULT_RUN_ROOT = C009_ROOT / "runs_validation_v2"
DEFAULT_OUTPUT = C009_ROOT / "protocol" / "wp1_pilot_closure_v1.json"


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


def verify_payload(document: dict[str, Any]) -> None:
    payload = dict(document)
    recorded = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if recorded != actual:
        raise RuntimeError("Internal pilot artifact payload hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    queue_path = run_root / "queue" / "pilot_queue_status.json"
    queue = load_json(queue_path)
    if queue.get("status") != "complete" or queue.get("completed_runs") != 6:
        raise RuntimeError("Pilot queue is not complete")

    records: list[dict[str, Any]] = []
    runner_hashes: set[str] = set()
    for family in ("nafnet", "restormer", "mprnet"):
        pair: dict[str, dict[str, Any]] = {}
        for arm in ("base", "merit_full"):
            run_dir = run_root / "pilot" / family / f"{arm}_seed0"
            failure = run_dir / "failure.json"
            if failure.exists():
                raise RuntimeError(f"Failure artifact remains: {failure}")
            manifest_path = run_dir / "run_manifest.json"
            marker_path = run_dir / "RUN_COMPLETE.json"
            final_path = run_dir / "final.pt"
            manifest = load_json(manifest_path)
            marker = load_json(marker_path)
            verify_payload(marker)
            if manifest.get("status") != "complete" or marker.get("status") != "PASS":
                raise RuntimeError(f"Pilot run is not complete: {run_dir}")
            if int(manifest.get("global_step", -1)) != 20:
                raise RuntimeError(f"Pilot did not reach 20 steps: {run_dir}")
            if int(manifest.get("resume_count", 0)) < 1:
                raise RuntimeError(f"Pilot did not exercise resume: {run_dir}")
            if marker["final_checkpoint_sha256"] != sha256_file(final_path):
                raise RuntimeError(f"Final checkpoint hash mismatch: {run_dir}")
            if marker["run_manifest_sha256"] != sha256_file(manifest_path):
                raise RuntimeError(f"Run manifest hash mismatch: {run_dir}")
            runner_hashes.add(str(manifest["runner_sha256"]))
            pair[arm] = manifest
            records.append(
                {
                    "family": family,
                    "arm": arm,
                    "seed": 0,
                    "global_step": 20,
                    "resume_count": int(manifest["resume_count"]),
                    "elapsed_seconds": float(manifest["elapsed_seconds"]),
                    "peak_cuda_memory_bytes_second_invocation": int(
                        manifest["peak_cuda_memory_bytes_this_invocation"]
                    ),
                    "initial_state_sha256": manifest["initial_state_sha256"],
                    "stream_sha256": manifest["stream_sha256"],
                    "final_checkpoint_sha256": marker["final_checkpoint_sha256"],
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
        for key in (
            "initial_state_sha256",
            "trainable_parameters",
            "stream_sha256",
            "global_step",
            "contract_sha256",
        ):
            if pair["base"][key] != pair["merit_full"][key]:
                raise RuntimeError(f"Paired pilot mismatch: {family}/{key}")
        paired_path = run_root / "pilot" / family / "paired_seed0_PASS.json"
        paired = load_json(paired_path)
        verify_payload(paired)
        if paired.get("status") != "PASS":
            raise RuntimeError(f"Paired PASS is invalid: {paired_path}")
    if len(runner_hashes) != 1:
        raise RuntimeError(f"Pilot used multiple runner hashes: {runner_hashes}")

    document: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP1-PILOT-CLOSURE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "3 families x 2 arms x seed 0; forced stop at step 5 and resume to step 20",
        "queue_status_path": str(queue_path),
        "queue_status_sha256": sha256_file(queue_path),
        "pilot_execution_freeze_sha256": queue["execution_freeze_sha256"],
        "runner_sha256": next(iter(runner_hashes)),
        "checks": {
            "six_runs_complete": True,
            "forced_resume_exercised_all_runs": True,
            "all_fixed_pilot_final_step_20": True,
            "paired_initial_states_equal": True,
            "paired_trainable_parameters_equal": True,
            "paired_stream_hashes_equal": True,
            "checkpoint_and_manifest_hashes_valid": True,
            "nonfinite_or_runtime_failures": 0,
        },
        "records": records,
        "interpretation": (
            "This closes only technical checkpoint/resume and paired-stream readiness. "
            "It is not a restoration-performance result and is not used for model selection."
        ),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    document["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
