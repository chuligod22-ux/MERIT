#!/usr/bin/env python3
"""Run the frozen MERIT D3 endpoint without persisting full probability maps.

The frozen C005 endpoint explicitly permits importing its evaluator for
streaming use.  This runner keeps every model fixed, evaluates one complete
frame (RAW plus the six paired restoration runs), and atomically records the
endpoint events and byte hashes needed for an auditable resume.  A resume may
repeat only a frame that did not reach its atomic commit; it never changes a
model, threshold, cohort, label, ROI, or endpoint rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.torch_version import TorchVersion

from c006_operational_inference import (
    NAFNetUNet,
    ProtocolError,
    _load_d2_class,
    read_bgr,
    require_cuda,
    require_hash,
    sha256_file,
    tiled_detect_rgb,
    tiled_restore_gray,
)


SCHEMA = "MERIT-C006-D3-STREAM-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ProtocolError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    sibling_root = str(path.parent.resolve())
    inserted = sibling_root not in sys.path
    if inserted:
        sys.path.insert(0, sibling_root)
    try:
        specification.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(sibling_root)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if path.exists() and not overwrite:
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("schema_version") != SCHEMA:
        raise ProtocolError("Unexpected D3 streaming protocol schema")
    return protocol


def verify_assets(protocol: dict[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for name, record in protocol["assets"].items():
        asset_path = Path(record["path"])
        verified[name] = require_hash(asset_path, record["sha256"], name)
    for key, record in protocol["restoration"]["checkpoints"].items():
        checkpoint_path = Path(record["path"])
        verified[f"restoration:{key}"] = require_hash(
            checkpoint_path, record["sha256"], key
        )
    detector = protocol["detector"]
    for name in ("code", "protocol", "checkpoint", "ledger", "imagenet_weight"):
        record = detector[name]
        verified[f"detector:{name}"] = require_hash(
            Path(record["path"]), record["sha256"], f"detector {name}"
        )
    return verified


def load_inputs(protocol: dict[str, Any]):
    cohort_path = Path(protocol["assets"]["cohort"]["path"])
    consensus_path = Path(protocol["assets"]["consensus"]["path"])
    rows = read_csv(cohort_path)
    if len(rows) != int(protocol["cohort"]["total_frames"]):
        raise ProtocolError("Frozen cohort row count changed")
    cohort = {row["cohort_id"]: row for row in rows}
    if len(cohort) != len(rows):
        raise ProtocolError("Duplicate cohort IDs")
    consensus_payload = load_json(consensus_path)
    if consensus_payload.get("stage") != "consensus_final":
        raise ProtocolError("Consensus is not final")
    if consensus_payload.get("cohort_sha256") != sha256_file(cohort_path):
        raise ProtocolError("Consensus/cohort hash mismatch")
    consensus = consensus_payload["records"]
    clear_labels = set(protocol["endpoint"]["clear_labels"])
    eligible_rows = [
        row for row in rows if consensus[row["cohort_id"]]["label"] in clear_labels
    ]
    if len(eligible_rows) != int(protocol["cohort"]["eligible_frames"]):
        raise ProtocolError("Eligible-frame count changed")
    label_counts = Counter(record["label"] for record in consensus.values())
    if dict(sorted(label_counts.items())) != protocol["cohort"]["label_counts"]:
        raise ProtocolError("Consensus label counts changed")
    return cohort, eligible_rows, consensus


def load_models(protocol: dict[str, Any], device: torch.device):
    restoration_models: dict[tuple[str, str], torch.nn.Module] = {}
    with torch.serialization.safe_globals([TorchVersion]):
        for run in protocol["runs"]:
            if run["condition"] == "raw":
                continue
            key = run["checkpoint_key"]
            record = protocol["restoration"]["checkpoints"][key]
            checkpoint = torch.load(
                Path(record["path"]), map_location="cpu", weights_only=True
            )
            model = NAFNetUNet(width=int(protocol["restoration"]["model_width"]))
            model.load_state_dict(checkpoint["state"], strict=True)
            restoration_models[(run["condition"], run["seed"])] = model.to(device).eval()
    detector_record = protocol["detector"]
    detector_class = _load_d2_class(Path(detector_record["code"]["path"]))
    detector_model = detector_class()
    detector_checkpoint = torch.load(
        Path(detector_record["checkpoint"]["path"]),
        map_location="cpu",
        weights_only=True,
    )
    detector_model.load_state_dict(detector_checkpoint["model_state"], strict=True)
    return restoration_models, detector_model.to(device).eval()


def detector_probability(
    bgr: np.ndarray,
    detector_model: torch.nn.Module,
    device: torch.device,
    protocol: dict[str, Any],
) -> np.ndarray:
    detector = protocol["detector"]
    return tiled_detect_rgb(
        bgr,
        detector_model,
        device,
        tile_size=int(detector["tiling"]["tile_size"]),
        batch_size=int(detector["tiling"]["batch_size"]),
        mean=tuple(detector["normalization"]["mean"]),
        std=tuple(detector["normalization"]["std"]),
    )


def endpoint_event(
    probability: np.ndarray,
    record: dict[str, Any],
    row: dict[str, str],
    condition: str,
    seed: str,
    endpoint_module,
    protocol: dict[str, Any],
    elapsed_seconds: float,
    restored_sha256: str | None,
) -> dict[str, Any]:
    results = endpoint_module.evaluate_probability_map_sensitivities(
        probability,
        record,
        float(protocol["endpoint"]["threshold"]),
        None,
        tuple(int(value) for value in protocol["endpoint"]["minimum_areas"]),
    )
    return {
        "condition": condition,
        "seed": seed,
        "probability_sha256_float32_c": array_sha256(
            probability.astype(np.float32, copy=False)
        ),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "restored_uint8_bgr_sha256": restored_sha256,
        "inference_seconds": elapsed_seconds,
        "subsets": endpoint_module.analysis_subsets(row),
        "groups": endpoint_module.descriptive_groups(row),
        "endpoint_by_minimum_area": {
            str(area): result for area, result in results.items()
        },
    }


def progress_path(output_root: Path, index: int, cohort_id: str) -> Path:
    return output_root / "progress" / f"{index:04d}_{cohort_id}.json"


def validate_completed_progress(
    path: Path,
    row: dict[str, str],
    expected_runs: list[tuple[str, str]],
) -> None:
    payload = load_json(path)
    if payload.get("cohort_id") != row["cohort_id"]:
        raise ProtocolError(f"Progress cohort mismatch: {path}")
    if payload.get("source_sha256") != row["sha256"]:
        raise ProtocolError(f"Progress source mismatch: {path}")
    actual_runs = [
        (event["condition"], event["seed"]) for event in payload.get("events", [])
    ]
    if actual_runs != expected_runs:
        raise ProtocolError(f"Progress run coverage mismatch: {path}")


def run_d3(protocol_path: Path, output_root: Path) -> None:
    protocol = load_protocol(protocol_path)
    verified = verify_assets(protocol)
    _, eligible_rows, consensus = load_inputs(protocol)
    endpoint_module = load_module(
        Path(protocol["assets"]["endpoint_code"]["path"]),
        "merit_frozen_c005_endpoint",
    )
    detector_identity = endpoint_module.load_detector_freeze(
        Path(protocol["detector"]["ledger"]["path"]),
        Path(protocol["detector"]["checkpoint"]["path"]),
        Path(protocol["detector"]["imagenet_weight"]["path"]),
        float(protocol["endpoint"]["threshold"]),
    )
    expected_runs = [(run["condition"], run["seed"]) for run in protocol["runs"]]
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "d3_endpoint_streaming_final.json").exists():
        raise FileExistsError("Final endpoint already exists; D3 cannot be rerun")
    completed = 0
    for index, row in enumerate(eligible_rows, start=1):
        path = progress_path(output_root, index, row["cohort_id"])
        if path.exists():
            validate_completed_progress(path, row, expected_runs)
            completed += 1
    device = require_cuda()
    restoration_models, detector_model = load_models(protocol, device)
    raw_root = Path(protocol["raw_root"])
    start_time = time.perf_counter()
    session_processed = 0
    print(
        json.dumps(
            {
                "stage": "start",
                "eligible": len(eligible_rows),
                "already_complete": completed,
                "created_utc": utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for index, row in enumerate(eligible_rows, start=1):
        path = progress_path(output_root, index, row["cohort_id"])
        if path.exists():
            continue
        frame_started = time.perf_counter()
        source_path = raw_root / Path(row["relative_path"])
        source_hash = require_hash(source_path, row["sha256"], row["cohort_id"])
        image = read_bgr(source_path)
        expected_shape = tuple(int(value) for value in protocol["input_shape"])
        if image.shape != expected_shape:
            raise ProtocolError(
                f"Input shape mismatch for {row['cohort_id']}: {image.shape}"
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        events: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats(device)
        detected_at = time.perf_counter()
        probability = detector_probability(image, detector_model, device, protocol)
        events.append(
            endpoint_event(
                probability,
                consensus[row["cohort_id"]],
                row,
                "raw",
                "none",
                endpoint_module,
                protocol,
                time.perf_counter() - detected_at,
                None,
            )
        )
        del probability
        restoration_config = protocol["restoration"]["tiling"]
        for run in protocol["runs"]:
            if run["condition"] == "raw":
                continue
            run_started = time.perf_counter()
            model = restoration_models[(run["condition"], run["seed"])]
            restored = tiled_restore_gray(
                gray,
                model,
                device,
                tile_size=int(restoration_config["tile_size"]),
                overlap=int(restoration_config["overlap"]),
                batch_size=int(restoration_config["batch_size"]),
                blend_floor=float(restoration_config["positive_hann_floor"]),
                amp_dtype=protocol["restoration"]["amp_dtype"],
            )
            encoded = np.rint(restored * np.float32(255.0)).astype(np.uint8)
            restored_bgr = cv2.cvtColor(encoded, cv2.COLOR_GRAY2BGR)
            restored_hash = array_sha256(restored_bgr)
            del restored, encoded
            probability = detector_probability(
                restored_bgr, detector_model, device, protocol
            )
            events.append(
                endpoint_event(
                    probability,
                    consensus[row["cohort_id"]],
                    row,
                    run["condition"],
                    run["seed"],
                    endpoint_module,
                    protocol,
                    time.perf_counter() - run_started,
                    restored_hash,
                )
            )
            del probability, restored_bgr
        elapsed = time.perf_counter() - frame_started
        payload = {
            "schema_version": SCHEMA,
            "cohort_id": row["cohort_id"],
            "eligible_index": index,
            "source_relative_path": row["relative_path"],
            "source_sha256": source_hash,
            "consensus_label": consensus[row["cohort_id"]]["label"],
            "events": events,
            "frame_elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "committed_utc": utc_now(),
        }
        write_json_atomic(path, payload, overwrite=False)
        completed += 1
        session_processed += 1
        total_elapsed = time.perf_counter() - start_time
        average = total_elapsed / session_processed
        remaining = len(eligible_rows) - completed
        status = {
            "schema_version": SCHEMA,
            "state": "RUNNING" if remaining else "INFERENCE_COMPLETE",
            "completed": completed,
            "eligible": len(eligible_rows),
            "remaining": remaining,
            "last_cohort_id": row["cohort_id"],
            "last_frame_seconds": elapsed,
            "session_average_seconds": average,
            "session_projected_remaining_seconds": average * remaining,
            "updated_utc": utc_now(),
            "verified_asset_hashes": verified,
            "detector_identity": detector_identity,
        }
        write_json_atomic(output_root / "status.json", status, overwrite=True)
        print(json.dumps(status, sort_keys=True), flush=True)
    print(json.dumps({"stage": "inference_complete", "completed": completed}), flush=True)


def aggregate(protocol_path: Path, output_root: Path) -> None:
    protocol = load_protocol(protocol_path)
    verify_assets(protocol)
    _, eligible_rows, consensus = load_inputs(protocol)
    endpoint_module = load_module(
        Path(protocol["assets"]["endpoint_code"]["path"]),
        "merit_frozen_c005_endpoint_aggregate",
    )
    expected_runs = [(run["condition"], run["seed"]) for run in protocol["runs"]]
    areas = [int(value) for value in protocol["endpoint"]["minimum_areas"]]
    events_by_area: dict[int, list[dict[str, Any]]] = {area: [] for area in areas}
    manifest_records = []
    for index, row in enumerate(eligible_rows, start=1):
        path = progress_path(output_root, index, row["cohort_id"])
        if not path.is_file():
            raise ProtocolError(f"Missing D3 progress: {path}")
        validate_completed_progress(path, row, expected_runs)
        payload = load_json(path)
        manifest_records.append(
            {
                "cohort_id": row["cohort_id"],
                "path": str(path.relative_to(output_root)),
                "sha256": sha256_file(path),
            }
        )
        for event in payload["events"]:
            identity = {
                "cohort_id": row["cohort_id"],
                "condition": event["condition"],
                "seed": event["seed"],
                "subsets": event["subsets"],
                "groups": event["groups"],
            }
            for area in areas:
                result = dict(event["endpoint_by_minimum_area"][str(area)])
                result.update(identity)
                events_by_area[area].append(result)
    manifest = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "record_count": len(manifest_records),
        "records": manifest_records,
    }
    manifest_path = output_root / "d3_stream_progress_manifest.json"
    write_json_atomic(manifest_path, manifest, overwrite=False)
    sensitivity_outputs = {}
    model_seeds = list(protocol["endpoint"]["model_seeds"])
    primary_area = int(protocol["endpoint"]["primary_minimum_area"])
    for area in areas:
        summaries = endpoint_module.summarize_events(events_by_area[area])
        effects = endpoint_module.task_effects(summaries, model_seeds)
        sensitivity_outputs[str(area)] = {
            "min_component_area": area,
            "min_roi_overlap_pixels": area,
            "operational": area == primary_area,
            "run_summaries": summaries,
            "task_effects": effects,
            "claim_gates": endpoint_module.claim_gates_with_denominators(
                effects, summaries
            ),
        }
    primary = sensitivity_outputs[str(primary_area)]
    restoration_identities = {
        f"{run['condition']}|{run['seed']}": {
            "checkpoint_path": protocol["restoration"]["checkpoints"][run["checkpoint_key"]]["path"],
            "checkpoint_sha256": protocol["restoration"]["checkpoints"][run["checkpoint_key"]]["sha256"],
        }
        for run in protocol["runs"]
        if run["condition"] != "raw"
    }
    detector_identity = endpoint_module.load_detector_freeze(
        Path(protocol["detector"]["ledger"]["path"]),
        Path(protocol["detector"]["checkpoint"]["path"]),
        Path(protocol["detector"]["imagenet_weight"]["path"]),
        float(protocol["endpoint"]["threshold"]),
    )
    output = {
        "schema_version": "MERIT-C005-D3-ENDPOINT-v2",
        "execution_mode": "frozen_streaming_import",
        "threshold": float(protocol["endpoint"]["threshold"]),
        "min_component_area": primary_area,
        "min_roi_overlap_pixels": primary_area,
        "sensitivity_min_areas": areas,
        "model_seeds": model_seeds,
        "eligible_consensus_frames": len(eligible_rows),
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "cohort_sha256": protocol["assets"]["cohort"]["sha256"],
            "consensus_sha256": protocol["assets"]["consensus"]["sha256"],
            "stream_progress_manifest_sha256": sha256_file(manifest_path),
            "endpoint_code_sha256": protocol["assets"]["endpoint_code"]["sha256"],
            "detector": detector_identity,
            "restoration_checkpoints": restoration_identities,
        },
        "run_summaries": primary["run_summaries"],
        "task_effects": primary["task_effects"],
        "claim_gates": primary["claim_gates"],
        "min_area_sensitivity": sensitivity_outputs,
    }
    final_path = output_root / "d3_endpoint_streaming_final.json"
    write_json_atomic(final_path, output, overwrite=False)
    print(
        json.dumps(
            {
                "output": str(final_path),
                "events_per_min_area": len(events_by_area[primary_area]),
                "claim_gates": primary["claim_gates"],
            },
            sort_keys=True,
        )
    )


def self_test(protocol_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    verify_assets(protocol)
    endpoint_module = load_module(
        Path(protocol["assets"]["endpoint_code"]["path"]),
        "merit_frozen_c005_endpoint_selftest",
    )
    device = require_cuda()
    restoration_models, detector_model = load_models(protocol, device)
    axis = np.arange(512, dtype=np.uint16)
    gray = ((axis[:, None] + axis[None, :]) % 256).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raw_probability = detector_probability(bgr, detector_model, device, protocol)
    first_run = next(run for run in protocol["runs"] if run["condition"] != "raw")
    config = protocol["restoration"]["tiling"]
    restored = tiled_restore_gray(
        gray,
        restoration_models[(first_run["condition"], first_run["seed"])],
        device,
        tile_size=int(config["tile_size"]),
        overlap=int(config["overlap"]),
        batch_size=int(config["batch_size"]),
        blend_floor=float(config["positive_hann_floor"]),
        amp_dtype=protocol["restoration"]["amp_dtype"],
    )
    restored_bgr = cv2.cvtColor(
        np.rint(restored * np.float32(255.0)).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    restored_probability = detector_probability(
        restored_bgr, detector_model, device, protocol
    )
    dummy = {"label": "crack_visible", "rois": [{"x": 0, "y": 0, "width": 512, "height": 512}]}
    raw_result = endpoint_module.evaluate_probability_map_sensitivities(
        raw_probability,
        dummy,
        float(protocol["endpoint"]["threshold"]),
        None,
        tuple(int(value) for value in protocol["endpoint"]["minimum_areas"]),
    )
    print(
        json.dumps(
            {
                "pass": raw_probability.shape == (512, 512)
                and restored_probability.shape == (512, 512)
                and all(np.isfinite(raw_probability).flat)
                and all(np.isfinite(restored_probability).flat),
                "raw_probability_sha256": array_sha256(raw_probability),
                "restored_probability_sha256": array_sha256(restored_probability),
                "endpoint": raw_result,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output-root", type=Path, required=True)
    aggregate_command = commands.add_parser("aggregate")
    aggregate_command.add_argument("--output-root", type=Path, required=True)
    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "run":
        run_d3(args.protocol, args.output_root)
    elif args.command == "aggregate":
        aggregate(args.protocol, args.output_root)
    else:
        self_test(args.protocol)


if __name__ == "__main__":
    main()
