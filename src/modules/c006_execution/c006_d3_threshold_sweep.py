#!/usr/bin/env python3
"""Post-hoc, leakage-controlled threshold sweep for the frozen MERIT D3 endpoint.

The original threshold=0.55 endpoint remains the primary frozen analysis.  This
addendum re-runs the identical seven image/model conditions and evaluates the
pre-existing D2 grid (0.05..0.95 by 0.05) while each probability map is in
memory.  Full probability maps and restored images are not retained.

Threshold calibration is restricted to non-R acquisitions.  R acquisitions
are held out for the final operating-point evaluation.  Every probability and
restored-image fingerprint must reproduce the completed D3 run exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import c006_d3_streaming as d3
from c006_operational_inference import (
    ProtocolError,
    read_bgr,
    require_cuda,
    require_hash,
    sha256_file,
    tiled_restore_gray,
)


SCHEMA = "MERIT-C006-D3-THRESHOLD-SWEEP-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(path)
    if protocol.get("schema_version") != SCHEMA:
        raise ProtocolError("Unexpected D3 threshold-sweep protocol schema")
    require_hash(
        Path(protocol["base_protocol"]["path"]),
        protocol["base_protocol"]["sha256"],
        "base D3 protocol",
    )
    require_hash(
        Path(protocol["runner"]["path"]),
        protocol["runner"]["sha256"],
        "threshold-sweep runner",
    )
    base = d3.load_protocol(Path(protocol["base_protocol"]["path"]))
    d3.verify_assets(base)
    thresholds = [float(value) for value in protocol["analysis"]["thresholds"]]
    if thresholds != sorted(set(thresholds)):
        raise ProtocolError("Threshold grid must be strictly increasing and unique")
    if not thresholds or thresholds[0] <= 0.0 or thresholds[-1] >= 1.0:
        raise ProtocolError("Thresholds must lie strictly inside (0, 1)")
    if not any(math.isclose(value, 0.55, abs_tol=1e-12) for value in thresholds):
        raise ProtocolError("Threshold grid must retain the frozen 0.55 endpoint")
    if int(protocol["analysis"]["minimum_area"]) != int(
        base["endpoint"]["primary_minimum_area"]
    ):
        raise ProtocolError("Sweep minimum area changed from the D3 primary endpoint")
    return protocol, base


def split_name(pass_id: str) -> str:
    """Keep acquisition directions disjoint; never split adjacent frames."""
    return "evaluation_R" if pass_id.upper().endswith("R") else "calibration_nonR"


def empty_endpoint(record: dict[str, Any], valid_pixels: int) -> dict[str, Any]:
    label = record["label"]
    output: dict[str, Any] = {
        "label": label,
        "eligible_component_count": 0,
        "valid_pixels": valid_pixels,
    }
    if label == "crack_visible":
        rois = record.get("rois", [])
        if not rois:
            raise ValueError("Consensus-positive frame has no ROI")
        output.update(
            {
                "roi_count": len(rois),
                "roi_hit_count": 0,
                "frame_roi_hit_fraction": 0.0,
            }
        )
    elif label == "no_visible_crack":
        output.update(
            {
                "any_eligible_component": False,
                "components_per_megapixel": 0.0 if valid_pixels else None,
            }
        )
    else:
        raise ValueError(f"Ineligible label: {label}")
    return output


def threshold_results(
    probability: np.ndarray,
    record: dict[str, Any],
    endpoint_module,
    thresholds: list[float],
    minimum_area: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate the sweep exactly, with a logically exact empty-map fast path."""
    maximum = float(probability.max())
    valid_pixels = int(probability.size)
    output: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        if maximum < threshold:
            output[key] = empty_endpoint(record, valid_pixels)
        else:
            output[key] = endpoint_module.evaluate_probability_map_sensitivities(
                probability,
                record,
                threshold,
                None,
                (minimum_area,),
            )[minimum_area]
    return output


def original_identity_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (event["condition"], event["seed"]): event
        for event in payload.get("events", [])
    }


def sweep_event(
    probability: np.ndarray,
    record: dict[str, Any],
    row: dict[str, str],
    condition: str,
    seed: str,
    endpoint_module,
    thresholds: list[float],
    minimum_area: int,
    elapsed_seconds: float,
    restored_sha256: str | None,
    original_event: dict[str, Any],
) -> dict[str, Any]:
    probability_hash = d3.array_sha256(probability.astype(np.float32, copy=False))
    if probability_hash != original_event["probability_sha256_float32_c"]:
        raise ProtocolError(
            f"Probability fingerprint did not reproduce for {row['cohort_id']} "
            f"{condition}|{seed}"
        )
    if restored_sha256 != original_event["restored_uint8_bgr_sha256"]:
        raise ProtocolError(
            f"Restoration fingerprint did not reproduce for {row['cohort_id']} "
            f"{condition}|{seed}"
        )
    subsets = endpoint_module.analysis_subsets(row)
    subsets.append(split_name(row["pass_id"]))
    return {
        "condition": condition,
        "seed": seed,
        "probability_sha256_float32_c": probability_hash,
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "restored_uint8_bgr_sha256": restored_sha256,
        "inference_and_sweep_seconds": elapsed_seconds,
        "subsets": subsets,
        "groups": endpoint_module.descriptive_groups(row),
        "threshold_results": threshold_results(
            probability, record, endpoint_module, thresholds, minimum_area
        ),
    }


def progress_path(output_root: Path, index: int, cohort_id: str) -> Path:
    return output_root / "progress" / f"{index:04d}_{cohort_id}.json"


def validate_completed_progress(
    path: Path,
    row: dict[str, str],
    expected_runs: list[tuple[str, str]],
    thresholds: list[float],
) -> None:
    payload = load_json(path)
    if payload.get("schema_version") != SCHEMA:
        raise ProtocolError(f"Progress schema mismatch: {path}")
    if payload.get("cohort_id") != row["cohort_id"]:
        raise ProtocolError(f"Progress cohort mismatch: {path}")
    if payload.get("source_sha256") != row["sha256"]:
        raise ProtocolError(f"Progress source mismatch: {path}")
    if payload.get("analysis_split") != split_name(row["pass_id"]):
        raise ProtocolError(f"Progress split mismatch: {path}")
    events = payload.get("events", [])
    if [(event["condition"], event["seed"]) for event in events] != expected_runs:
        raise ProtocolError(f"Progress run coverage mismatch: {path}")
    expected_thresholds = [f"{value:.2f}" for value in thresholds]
    for event in events:
        if list(event.get("threshold_results", {})) != expected_thresholds:
            raise ProtocolError(f"Progress threshold coverage mismatch: {path}")


def run_sweep(protocol_path: Path, output_root: Path, *, limit: int | None = None) -> None:
    protocol, base = load_protocol(protocol_path)
    _, eligible_rows, consensus = d3.load_inputs(base)
    if limit is not None:
        eligible_rows = eligible_rows[:limit]
    endpoint_module = d3.load_module(
        Path(base["assets"]["endpoint_code"]["path"]),
        "merit_d3_threshold_sweep_endpoint",
    )
    thresholds = [float(value) for value in protocol["analysis"]["thresholds"]]
    minimum_area = int(protocol["analysis"]["minimum_area"])
    expected_runs = [(run["condition"], run["seed"]) for run in base["runs"]]
    original_root = Path(protocol["original_d3_output_root"])
    if not (original_root / "d3_endpoint_streaming_final.json").is_file():
        raise ProtocolError("Completed original D3 endpoint is missing")
    output_root.mkdir(parents=True, exist_ok=True)
    final_path = output_root / "d3_threshold_sweep_final.json"
    if final_path.exists():
        raise FileExistsError("Final threshold sweep already exists")
    completed = 0
    for index, row in enumerate(eligible_rows, start=1):
        path = progress_path(output_root, index, row["cohort_id"])
        if path.exists():
            validate_completed_progress(path, row, expected_runs, thresholds)
            completed += 1
    device = require_cuda()
    restoration_models, detector_model = d3.load_models(base, device)
    raw_root = Path(base["raw_root"])
    start_time = time.perf_counter()
    session_processed = 0
    print(
        json.dumps(
            {
                "stage": "start",
                "eligible": len(eligible_rows),
                "already_complete": completed,
                "thresholds": thresholds,
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
        if image.shape != tuple(int(value) for value in base["input_shape"]):
            raise ProtocolError(f"Input shape mismatch for {row['cohort_id']}")
        original_path = d3.progress_path(original_root, index, row["cohort_id"])
        original_payload = load_json(original_path)
        original_index = original_identity_index(original_payload)
        if set(original_index) != set(expected_runs):
            raise ProtocolError(f"Original D3 run coverage changed: {original_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        events: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        probability = d3.detector_probability(image, detector_model, device, base)
        events.append(
            sweep_event(
                probability,
                consensus[row["cohort_id"]],
                row,
                "raw",
                "none",
                endpoint_module,
                thresholds,
                minimum_area,
                time.perf_counter() - started,
                None,
                original_index[("raw", "none")],
            )
        )
        del probability
        restoration_config = base["restoration"]["tiling"]
        for run in base["runs"]:
            if run["condition"] == "raw":
                continue
            started = time.perf_counter()
            model = restoration_models[(run["condition"], run["seed"])]
            restored = tiled_restore_gray(
                gray,
                model,
                device,
                tile_size=int(restoration_config["tile_size"]),
                overlap=int(restoration_config["overlap"]),
                batch_size=int(restoration_config["batch_size"]),
                blend_floor=float(restoration_config["positive_hann_floor"]),
                amp_dtype=base["restoration"]["amp_dtype"],
            )
            encoded = np.rint(restored * np.float32(255.0)).astype(np.uint8)
            restored_bgr = cv2.cvtColor(encoded, cv2.COLOR_GRAY2BGR)
            restored_hash = d3.array_sha256(restored_bgr)
            del restored, encoded
            probability = d3.detector_probability(
                restored_bgr, detector_model, device, base
            )
            run_key = (run["condition"], run["seed"])
            events.append(
                sweep_event(
                    probability,
                    consensus[row["cohort_id"]],
                    row,
                    run["condition"],
                    run["seed"],
                    endpoint_module,
                    thresholds,
                    minimum_area,
                    time.perf_counter() - started,
                    restored_hash,
                    original_index[run_key],
                )
            )
            del probability, restored_bgr
        elapsed = time.perf_counter() - frame_started
        payload = {
            "schema_version": SCHEMA,
            "cohort_id": row["cohort_id"],
            "eligible_index": index,
            "analysis_split": split_name(row["pass_id"]),
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
        }
        write_json_atomic(output_root / "status.json", status, overwrite=True)
        print(json.dumps(status, sort_keys=True), flush=True)
    print(json.dumps({"stage": "inference_complete", "completed": completed}), flush=True)


def metric(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get(name)
    return None if value is None else float(value)


def select_operating_point(
    curve: list[dict[str, Any]],
    *,
    mode: str,
    target: float,
) -> dict[str, Any]:
    if mode == "sensitivity_matched":
        feasible = [point for point in curve if point["hit"] is not None and point["hit"] >= target]
        if feasible:
            chosen = min(feasible, key=lambda p: (p["burden"], -p["threshold"]))
        else:
            chosen = max(curve, key=lambda p: (p["hit"], -p["burden"]))
    elif mode == "burden_matched":
        feasible = [point for point in curve if point["burden"] is not None and point["burden"] <= target]
        if feasible:
            chosen = max(feasible, key=lambda p: (p["hit"], -p["burden"], p["threshold"]))
        else:
            chosen = min(curve, key=lambda p: (p["burden"], -p["hit"]))
    else:
        raise ValueError(mode)
    output = dict(chosen)
    output.update({"mode": mode, "target": target, "feasible": bool(feasible)})
    return output


def aggregate(protocol_path: Path, output_root: Path) -> None:
    protocol, base = load_protocol(protocol_path)
    _, eligible_rows, _ = d3.load_inputs(base)
    endpoint_module = d3.load_module(
        Path(base["assets"]["endpoint_code"]["path"]),
        "merit_d3_threshold_sweep_aggregate",
    )
    thresholds = [float(value) for value in protocol["analysis"]["thresholds"]]
    expected_runs = [(run["condition"], run["seed"]) for run in base["runs"]]
    events_by_threshold: dict[str, list[dict[str, Any]]] = {
        f"{threshold:.2f}": [] for threshold in thresholds
    }
    manifest_records = []
    split_counts: defaultdict[str, int] = defaultdict(int)
    for index, row in enumerate(eligible_rows, start=1):
        path = progress_path(output_root, index, row["cohort_id"])
        if not path.is_file():
            raise ProtocolError(f"Missing sweep progress: {path}")
        validate_completed_progress(path, row, expected_runs, thresholds)
        payload = load_json(path)
        split_counts[payload["analysis_split"]] += 1
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
            for key, result in event["threshold_results"].items():
                record = dict(result)
                record.update(identity)
                events_by_threshold[key].append(record)
    manifest = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "record_count": len(manifest_records),
        "split_counts": dict(sorted(split_counts.items())),
        "records": manifest_records,
    }
    manifest_path = output_root / "d3_threshold_sweep_progress_manifest.json"
    write_json_atomic(manifest_path, manifest, overwrite=False)
    summaries_by_threshold: dict[str, dict[str, Any]] = {}
    curves: defaultdict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key in [f"{value:.2f}" for value in thresholds]:
        summaries = endpoint_module.summarize_events(events_by_threshold[key])
        summaries_by_threshold[key] = summaries
        for summary in summaries.values():
            if summary["group"] != "overall":
                continue
            if summary["subset"] not in {"calibration_nonR", "evaluation_R", "all"}:
                continue
            run_key = f"{summary['condition']}|{summary['seed']}"
            curves[run_key][summary["subset"]].append(
                {
                    "threshold": float(key),
                    "hit": metric(summary, "frame_equal_roi_hit_fraction"),
                    "pooled_hit": metric(summary, "pooled_roi_hit_fraction_secondary"),
                    "burden": metric(summary, "components_per_no_visible_frame"),
                    "any_component_rate": metric(
                        summary, "any_component_no_visible_frame_proportion"
                    ),
                    "positive_frames": int(summary["positive_frames"]),
                    "positive_rois": int(summary["positive_rois"]),
                    "no_visible_frames": int(summary["no_visible_frames"]),
                }
            )
    raw_cal_curve = curves["raw|none"]["calibration_nonR"]
    raw_reference = next(
        point for point in raw_cal_curve if math.isclose(point["threshold"], 0.55)
    )
    selected: dict[str, Any] = {}
    for run_key, split_curves in sorted(curves.items()):
        calibration = split_curves["calibration_nonR"]
        evaluation_by_threshold = {
            point["threshold"]: point for point in split_curves["evaluation_R"]
        }
        sensitivity_choice = select_operating_point(
            calibration,
            mode="sensitivity_matched",
            target=float(raw_reference["hit"]),
        )
        burden_choice = select_operating_point(
            calibration,
            mode="burden_matched",
            target=float(raw_reference["burden"]),
        )
        selected[run_key] = {
            "sensitivity_matched": {
                "calibration": sensitivity_choice,
                "evaluation": evaluation_by_threshold[sensitivity_choice["threshold"]],
            },
            "burden_matched": {
                "calibration": burden_choice,
                "evaluation": evaluation_by_threshold[burden_choice["threshold"]],
            },
        }
    output = {
        "schema_version": SCHEMA,
        "analysis_status": "post_hoc_addendum_primary_0.55_unchanged",
        "created_utc": utc_now(),
        "thresholds": thresholds,
        "minimum_area": int(protocol["analysis"]["minimum_area"]),
        "split_rule": protocol["analysis"]["split_rule"],
        "split_counts": dict(sorted(split_counts.items())),
        "limitations": protocol["analysis"]["limitations"],
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "base_protocol_sha256": protocol["base_protocol"]["sha256"],
            "original_d3_final_sha256": sha256_file(
                Path(protocol["original_d3_output_root"])
                / "d3_endpoint_streaming_final.json"
            ),
            "progress_manifest_sha256": sha256_file(manifest_path),
        },
        "raw_0.55_calibration_reference": raw_reference,
        "curves": {key: dict(value) for key, value in curves.items()},
        "development_selected_operating_points": selected,
        "run_summaries_by_threshold": summaries_by_threshold,
    }
    final_path = output_root / "d3_threshold_sweep_final.json"
    write_json_atomic(final_path, output, overwrite=False)
    print(json.dumps({"output": str(final_path), "split_counts": output["split_counts"]}, sort_keys=True))


def self_test(protocol_path: Path) -> None:
    protocol, base = load_protocol(protocol_path)
    endpoint_module = d3.load_module(
        Path(base["assets"]["endpoint_code"]["path"]),
        "merit_d3_threshold_sweep_selftest",
    )
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:18, 10:18] = 0.60
    record = {
        "label": "crack_visible",
        "rois": [{"x": 8, "y": 8, "width": 16, "height": 16}],
    }
    thresholds = [0.55, 0.65]
    results = threshold_results(probability, record, endpoint_module, thresholds, 32)
    exact_empty = endpoint_module.evaluate_probability_map_sensitivities(
        probability, record, 0.65, None, (32,)
    )[32]
    curve = [
        {"threshold": 0.50, "hit": 1.0, "burden": 3.0},
        {"threshold": 0.55, "hit": 0.8, "burden": 1.0},
        {"threshold": 0.60, "hit": 0.5, "burden": 0.0},
    ]
    choice = select_operating_point(curve, mode="sensitivity_matched", target=0.8)
    passed = (
        results["0.55"]["roi_hit_count"] == 1
        and results["0.65"] == exact_empty
        and choice["threshold"] == 0.55
        and split_name("2") == "calibration_nonR"
        and split_name("2R") == "evaluation_R"
    )
    print(json.dumps({"pass": passed, "results": results, "choice": choice}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--limit", type=int)
    aggregate_command = commands.add_parser("aggregate")
    aggregate_command.add_argument("--output-root", type=Path, required=True)
    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "run":
        run_sweep(args.protocol, args.output_root, limit=args.limit)
    elif args.command == "aggregate":
        aggregate(args.protocol, args.output_root)
    else:
        self_test(args.protocol)


if __name__ == "__main__":
    main()
