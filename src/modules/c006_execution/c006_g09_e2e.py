#!/usr/bin/env python3
"""Prepare and aggregate the G09 operational 8K E2E gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "MERIT-C006-G09-E2E-v1"
BENCHMARK_SCHEMA = "MERIT-C005-E2E-BENCHMARK-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Unexpected operational protocol schema")
    return payload


def projected_hours(results: list[dict[str, Any]], cohort_frames: int) -> tuple[float, dict[str, float]]:
    if len(results) != 3:
        raise ValueError("Exactly three seed results are required")
    raw = max(float(item["condition_seconds"]["raw"]) for item in results)
    off = sum(float(item["condition_seconds"]["off"]) for item in results)
    on = sum(float(item["condition_seconds"]["on"]) for item in results)
    seconds = cohort_frames * (raw + off + on)
    return seconds / 3600.0, {
        "raw_max_seconds": raw,
        "off_sum_seconds": off,
        "on_sum_seconds": on,
        "per_frame_all_seven_maps_seconds": raw + off + on,
        "projected_total_seconds": seconds,
    }


def prepare(protocol_path: Path, output_root: Path, python_executable: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    adapter = Path(protocol["adapter_path"])
    benchmark = Path(protocol["benchmark_path"])
    assets = {
        "protocol": {"path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path)},
        "adapter": {"path": str(adapter), "sha256": sha256_file(adapter)},
        "gate_driver": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "benchmark": {"path": str(benchmark), "sha256": sha256_file(benchmark)},
    }
    if assets["benchmark"]["sha256"] != protocol["benchmark_sha256"]:
        raise ValueError("Frozen C005 benchmark hash mismatch")
    require_hash(
        Path(protocol["restoration"]["model_code_path"]),
        protocol["restoration"]["model_code_sha256"],
        "C006 model code",
    )
    for run_name, record in protocol["restoration"]["checkpoints"].items():
        require_hash(Path(record["path"]), record["sha256"], run_name)
    for path_key, hash_key, label in (
        ("code_path", "code_sha256", "D2 detector code"),
        ("protocol_path", "protocol_sha256", "D2 protocol"),
        ("checkpoint_path", "checkpoint_sha256", "D2 checkpoint"),
    ):
        require_hash(
            Path(protocol["detector"][path_key]),
            protocol["detector"][hash_key],
            label,
        )
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    specs: dict[str, dict[str, str]] = {}
    for seed in (0, 1, 2):
        values = [
            str(python_executable),
            "-X",
            "utf8",
            str(adapter),
            "--protocol",
            str(protocol_path.resolve()),
        ]
        spec = {
            "schema_version": BENCHMARK_SCHEMA,
            "width": int(protocol["input_shape"][1]),
            "height": int(protocol["input_shape"][0]),
            "model_seed": seed,
            "detector_threshold": float(protocol["detector"]["operating_threshold"]),
            "commands": {
                "restoration_off": values + [
                    "restore", "--arm", "task_off", "--seed", str(seed),
                    "--input", "{input}", "--output", "{output}",
                ],
                "restoration_on": values + [
                    "restore", "--arm", "task_on", "--seed", str(seed),
                    "--input", "{input}", "--output", "{output}",
                ],
                "detector": values + [
                    "detect", "--input", "{input}", "--output", "{output}",
                ],
            },
        }
        spec_path = output_root / f"e2e_spec_seed{seed}.json"
        write_new_json(spec_path, spec)
        specs[str(seed)] = {"path": str(spec_path.resolve()), "sha256": sha256_file(spec_path)}
    freeze = {
        "schema_version": "MERIT-C006-G09-PRERUN-FREEZE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_operational_e2e": True,
        "d3_read_or_used": False,
        "assets": assets,
        "specs": specs,
        "restoration_checkpoints": protocol["restoration"]["checkpoints"],
        "detector_checkpoint": {
            "path": protocol["detector"]["checkpoint_path"],
            "sha256": protocol["detector"]["checkpoint_sha256"],
        },
        "authoritative_projection_rule": protocol["gate"]["projection_rule"],
        "note": "The frozen C005 runner's per-seed legacy date flag is non-authoritative; d5500 G09 supplies the current deadline.",
    }
    freeze_path = output_root / "g09_prerun_freeze.json"
    write_new_json(freeze_path, freeze)
    return freeze


def aggregate(protocol_path: Path, result_paths: list[Path], output: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if len({path.resolve().parent for path in result_paths}) != 1:
        raise ValueError("All results must share one pre-run-freeze directory")
    freeze_path = result_paths[0].resolve().parent / "g09_prerun_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    results = []
    for expected_seed, path in zip((0, 1, 2), result_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != BENCHMARK_SCHEMA or payload.get("mode") != "PRODUCTION":
            raise ValueError(f"Invalid production E2E result: {path}")
        if payload.get("input_shape") != protocol["input_shape"]:
            raise ValueError(f"Input shape mismatch in result: {path}")
        if payload.get("spec_sha256") != freeze["specs"][str(expected_seed)]["sha256"]:
            raise ValueError(f"Spec binding mismatch for seed {expected_seed}: {path}")
        results.append(payload)
    hours, components = projected_hours(results, int(protocol["gate"]["cohort_frames"]))
    now = datetime.now(timezone.utc).astimezone()
    deadline = datetime.fromisoformat(protocol["gate"]["complete_by_kst"])
    memory_values = [item.get("peak_gpu_memory_delta_mb") for item in results]
    memory_available = all(value is not None for value in memory_values)
    threshold = float(protocol["gate"]["stop_if_projected_hours_gt"])
    passed = hours <= threshold and now <= deadline and memory_available
    payload = {
        "schema_version": "MERIT-C006-G09-E2E-AGGREGATE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "PRODUCTION",
        "d3_read_or_used": False,
        "protocol_sha256": sha256_file(protocol_path),
        "results": [
            {"seed": seed, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for seed, path in zip((0, 1, 2), result_paths)
        ],
        "projection": {
            "cohort_frames": int(protocol["gate"]["cohort_frames"]),
            "raw_runs": 1,
            "off_runs": 3,
            "on_runs": 3,
            **components,
            "projected_total_hours": hours,
        },
        "peak_gpu_memory_delta_mb": max(memory_values) if memory_available else None,
        "no_oom": True,
        "gate": {
            "stop_if_projected_hours_gt": threshold,
            "complete_by_kst": protocol["gate"]["complete_by_kst"],
            "completed_before_deadline": now <= deadline,
            "memory_measurement_available": memory_available,
            "pass": passed,
        },
        "legacy_runner_gate_note": "Per-seed runner date flags use the superseded 2026-09-15 deadline and are not the G09 decision.",
    }
    write_new_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--python-executable", type=Path, required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--results", type=Path, nargs=3, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare(args.protocol, args.output_root, args.python_executable)
    else:
        payload = aggregate(args.protocol, args.results, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
