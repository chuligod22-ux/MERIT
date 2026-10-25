#!/usr/bin/env python3
"""Frozen descriptive cam2 50-condition independent-detector response audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.torch_version import TorchVersion


SCHEMA = "MERIT-D0-CAM2-50COND-v1"
METRICS = ("mean_probability", "maximum_probability", "area_fraction_at_0.55")


class ProtocolError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ProtocolError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ProtocolError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != SCHEMA:
        raise ProtocolError("Unexpected protocol schema")
    return payload


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError(f"Could not decode image: {path}")
    return image


def parse_condition(name: str) -> tuple[int, float, int]:
    matched = re.fullmatch(
        r"crack_d(25|35|45|55|65)_ISO(100|200|400|800|1600)_V(60|80)",
        name,
    )
    if not matched:
        raise ProtocolError(f"Unexpected condition name: {name}")
    return int(matched.group(3)), int(matched.group(1)) / 10.0, int(matched.group(2))


def lower_median_name(names: list[str]) -> str:
    if not names:
        raise ProtocolError("Cannot select a median from an empty frame list")
    ordered = sorted(names)
    return ordered[(len(ordered) - 1) // 2]


def static_assets(protocol: dict[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    historical = protocol["source"]["historical_selection_csv"]
    verified["historical_selection_csv"] = require_hash(
        Path(historical["path"]), historical["sha256"], "historical cam2 selection"
    )
    for label in ("model_code", "operational_code"):
        record = protocol["restoration"][label]
        verified[label] = require_hash(Path(record["path"]), record["sha256"], label)
    for key, record in protocol["restoration"]["checkpoints"].items():
        verified[f"checkpoint:{key}"] = require_hash(
            Path(record["path"]), record["sha256"], key
        )
    for label in ("code", "protocol", "checkpoint"):
        record = protocol["detector"][label]
        verified[f"detector:{label}"] = require_hash(
            Path(record["path"]), record["sha256"], f"detector {label}"
        )
    return verified


def build_cohort(protocol_path: Path, output_root: Path) -> None:
    protocol = load_protocol(protocol_path)
    verified = static_assets(protocol)
    root = Path(protocol["source"]["root"])
    expected_shape = tuple(protocol["source"]["image_shape"])
    crop_fraction = float(protocol["source"]["analysis_crop"]["fraction"])
    development = set(protocol["development_conditions"])
    rows: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        speed, distance, iso = parse_condition(directory.name)
        frames = sorted(
            path.name for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        selected = lower_median_name(frames)
        image_path = directory / selected
        image = read_bgr(image_path)
        if image.shape != expected_shape:
            raise ProtocolError(f"Unexpected image shape {image.shape}: {image_path}")
        reference = speed == 60 and iso == 100
        crop_rows = int(np.floor(image.shape[0] * crop_fraction))
        rows.append(
            {
                "condition": directory.name,
                "speed_kmh": speed,
                "distance_m": f"{distance:.1f}",
                "iso": iso,
                "available_frames": len(frames),
                "selection_index_zero_based": (len(frames) - 1) // 2,
                "frame": selected,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "crop_rows": crop_rows,
                "development_condition": int(directory.name in development),
                "reference_condition": int(reference),
                "primary_condition": int(directory.name not in development and not reference),
            }
        )
        del image
    cohort = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
    if len(cohort) != int(protocol["source"]["expected_frames"]):
        raise ProtocolError(f"Expected 50 selected frames, found {len(cohort)}")
    if cohort["condition"].nunique() != int(protocol["source"]["expected_condition_cells"]):
        raise ProtocolError("Condition cells are not unique and complete")
    if int(cohort["primary_condition"].sum()) != int(protocol["reporting"]["expected_condition_cells"]):
        raise ProtocolError("Unexpected primary condition count")
    cohort_path = output_root / "d0_cam2_cohort.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(cohort_path, index=False, encoding="utf-8")
    freeze = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "analysis_code_path": str(Path(__file__).resolve()),
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        "cohort_path": str(cohort_path.resolve()),
        "cohort_sha256": sha256_file(cohort_path),
        "selected_frames": len(cohort),
        "condition_cells": int(cohort["condition"].nunique()),
        "primary_condition_cells": int(cohort["primary_condition"].sum()),
        "available_frame_count": int(cohort["available_frames"].sum()),
        "static_asset_hashes": verified,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(output_root / "d0_cam2_freeze.json", freeze, overwrite=False)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


def load_frozen_cohort(protocol_path: Path, output_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    protocol = load_protocol(protocol_path)
    freeze = read_json(output_root / "d0_cam2_freeze.json")
    require_hash(protocol_path, freeze["protocol_sha256"], "frozen cam2 protocol")
    require_hash(Path(__file__).resolve(), freeze["analysis_code_sha256"], "frozen cam2 code")
    cohort_path = Path(freeze["cohort_path"])
    require_hash(cohort_path, freeze["cohort_sha256"], "frozen cam2 cohort")
    static_assets(protocol)
    cohort = pd.read_csv(cohort_path, dtype={"frame": str})
    for row in cohort.itertuples(index=False):
        require_hash(Path(row.image_path), row.image_sha256, row.condition)
    return protocol, cohort


def import_components(protocol: dict[str, Any]):
    execution_dir = Path(protocol["restoration"]["model_code"]["path"]).parent
    if str(execution_dir) not in sys.path:
        sys.path.insert(0, str(execution_dir))
    from c006_model import NAFNetUNet
    from c006_operational_inference import tiled_detect_rgb, tiled_restore_gray

    detector_path = Path(protocol["detector"]["code"]["path"])
    specification = importlib.util.spec_from_file_location(
        "merit_cam2_frozen_d2_detector", detector_path
    )
    if specification is None or specification.loader is None:
        raise ProtocolError(f"Could not import D2 detector: {detector_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return NAFNetUNet, tiled_restore_gray, tiled_detect_rgb, module.ResNet34UNet


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ProtocolError("cam2 restoration audit requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda")


def load_models(protocol: dict[str, Any], device: torch.device):
    model_class, tiled_restore, tiled_detect, detector_class = import_components(protocol)
    models: dict[tuple[str, str], torch.nn.Module] = {}
    with torch.serialization.safe_globals([TorchVersion]):
        for run in protocol["runs"]:
            if run["condition"] == "raw":
                continue
            record = protocol["restoration"]["checkpoints"][run["checkpoint_key"]]
            checkpoint = torch.load(Path(record["path"]), map_location="cpu", weights_only=True)
            model = model_class(width=int(protocol["restoration"]["model_width"]))
            model.load_state_dict(checkpoint["state"], strict=True)
            models[(run["condition"], str(run["seed"]))] = model.to(device).eval()
    detector_record = protocol["detector"]["checkpoint"]
    detector_checkpoint = torch.load(
        Path(detector_record["path"]), map_location="cpu", weights_only=True
    )
    detector = detector_class()
    detector.load_state_dict(detector_checkpoint["model_state"], strict=True)
    detector = detector.to(device).eval()
    return models, detector, tiled_restore, tiled_detect


def probability_metrics(probability: np.ndarray, threshold: float) -> dict[str, float]:
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ProtocolError("Detector probability map must be finite 2-D")
    return {
        "mean_probability": float(probability.mean(dtype=np.float64)),
        "maximum_probability": float(probability.max()),
        "area_fraction_at_0.55": float(np.mean(probability >= threshold)),
    }


def detect_gray(
    gray: np.ndarray,
    detector: torch.nn.Module,
    device: torch.device,
    protocol: dict[str, Any],
    tiled_detect,
) -> dict[str, float]:
    detector_config = protocol["detector"]
    bgr = np.repeat(gray[:, :, None], 3, axis=2)
    probability = tiled_detect(
        bgr,
        detector,
        device,
        tile_size=int(detector_config["tiling"]["tile_size"]),
        batch_size=int(detector_config["tiling"]["batch_size"]),
        mean=tuple(detector_config["normalization"]["mean"]),
        std=tuple(detector_config["normalization"]["std"]),
    )
    metrics = probability_metrics(probability, float(detector_config["operating_threshold"]))
    del bgr, probability
    return metrics


def evaluate_gray(
    gray: np.ndarray,
    protocol: dict[str, Any],
    models: dict[tuple[str, str], torch.nn.Module],
    detector: torch.nn.Module,
    tiled_restore,
    tiled_detect,
    device: torch.device,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    events.append(
        {
            "condition": "raw",
            "seed": "none",
            "metrics": detect_gray(gray, detector, device, protocol, tiled_detect),
            "inference_seconds": time.perf_counter() - started,
        }
    )
    tiling = protocol["restoration"]["tiling"]
    for run in protocol["runs"]:
        if run["condition"] == "raw":
            continue
        started = time.perf_counter()
        model = models[(run["condition"], str(run["seed"]))]
        restored = tiled_restore(
            gray,
            model,
            device,
            tile_size=int(tiling["tile_size"]),
            overlap=int(tiling["overlap"]),
            batch_size=int(tiling["batch_size"]),
            blend_floor=float(tiling["positive_hann_floor"]),
            amp_dtype=protocol["restoration"]["amp_dtype"],
        )
        restored_u8 = np.rint(restored * np.float32(255.0)).astype(np.uint8)
        metrics = detect_gray(restored_u8, detector, device, protocol, tiled_detect)
        events.append(
            {
                "condition": run["condition"],
                "seed": str(run["seed"]),
                "metrics": metrics,
                "inference_seconds": time.perf_counter() - started,
            }
        )
        del restored, restored_u8
        torch.cuda.empty_cache()
    return events


def progress_path(output_root: Path, index: int, condition: str) -> Path:
    return output_root / "progress" / f"{index:03d}_{condition}.json"


def expected_run_keys(protocol: dict[str, Any]) -> set[tuple[str, str]]:
    return {(run["condition"], str(run["seed"])) for run in protocol["runs"]}


def validate_progress(path: Path, row: Any, protocol: dict[str, Any]) -> None:
    payload = read_json(path)
    if payload["image_sha256"] != row.image_sha256:
        raise ProtocolError(f"Progress input hash mismatch: {path}")
    keys = {(event["condition"], str(event["seed"])) for event in payload["events"]}
    if keys != expected_run_keys(protocol):
        raise ProtocolError(f"Incomplete progress run set: {path}")


def run_evaluation(protocol_path: Path, output_root: Path) -> None:
    protocol, cohort = load_frozen_cohort(protocol_path, output_root)
    device = require_cuda()
    models, detector, tiled_restore, tiled_detect = load_models(protocol, device)
    progress_dir = output_root / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    complete = 0
    for index, row in enumerate(cohort.itertuples(index=False), 1):
        target = progress_path(output_root, index, row.condition)
        if target.exists():
            validate_progress(target, row, protocol)
            complete += 1
    write_json_atomic(
        output_root / "d0_cam2_status.json",
        {
            "stage": "start",
            "created_utc": utc_now(),
            "frames": len(cohort),
            "already_complete": complete,
            "gpu": torch.cuda.get_device_name(device),
        },
        overwrite=True,
    )
    for index, row in enumerate(cohort.itertuples(index=False), 1):
        target = progress_path(output_root, index, row.condition)
        if target.exists():
            continue
        image = read_bgr(Path(row.image_path))
        gray = cv2.cvtColor(image[: int(row.crop_rows), :], cv2.COLOR_BGR2GRAY)
        del image
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        events = evaluate_gray(
            gray, protocol, models, detector, tiled_restore, tiled_detect, device
        )
        elapsed = time.perf_counter() - started
        payload = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "index": index,
            "condition_cell": row.condition,
            "speed_kmh": int(row.speed_kmh),
            "distance_m": float(row.distance_m),
            "iso": int(row.iso),
            "frame": row.frame,
            "image_path": row.image_path,
            "image_sha256": row.image_sha256,
            "crop_shape": list(gray.shape),
            "primary_condition": int(row.primary_condition),
            "events": events,
            "frame_elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        write_json_atomic(target, payload, overwrite=False)
        write_json_atomic(
            output_root / "d0_cam2_status.json",
            {
                "stage": "running",
                "created_utc": utc_now(),
                "completed_frames": index,
                "total_frames": len(cohort),
                "last_condition": row.condition,
                "last_frame_seconds": elapsed,
            },
            overwrite=True,
        )
        print(
            f"[{index:02d}/{len(cohort)}] {row.condition} {row.frame} "
            f"{elapsed:.1f}s peak={payload['peak_cuda_memory_bytes']/1e9:.2f}GB",
            flush=True,
        )
        del gray
    write_json_atomic(
        output_root / "d0_cam2_status.json",
        {"stage": "evaluation_complete", "created_utc": utc_now(), "frames": len(cohort)},
        overwrite=True,
    )


def bootstrap_mean_ci(values: np.ndarray, count: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    sampled = generator.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def summary_record(
    subset: pd.DataFrame,
    metric: str,
    arm: str,
    seed: str,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    delta_error = subset[f"{metric}_delta_abs_reference_error"].to_numpy(float)
    delta_response = subset[f"{metric}_restored_minus_raw"].to_numpy(float)
    return {
        "metric": metric,
        "arm": arm,
        "seed": seed,
        "condition_cells": len(subset),
        "mean_restored_minus_raw_response": float(delta_response.mean()),
        "mean_restored_minus_raw_abs_reference_error": float(delta_error.mean()),
        "mean_delta_abs_reference_error_95pct_condition_bootstrap_ci": bootstrap_mean_ci(
            delta_error, int(bootstrap["resamples"]), int(bootstrap["seed"])
        ),
        "conditions_closer_to_reference": int(np.count_nonzero(delta_error < 0)),
        "conditions_farther_from_reference": int(np.count_nonzero(delta_error > 0)),
    }


def aggregate(protocol_path: Path, output_root: Path) -> None:
    protocol, cohort = load_frozen_cohort(protocol_path, output_root)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(cohort.itertuples(index=False), 1):
        path = progress_path(output_root, index, row.condition)
        if not path.is_file():
            raise ProtocolError(f"Missing progress: {path}")
        validate_progress(path, row, protocol)
        payload = read_json(path)
        for event in payload["events"]:
            record = {
                "condition_cell": row.condition,
                "speed_kmh": int(row.speed_kmh),
                "distance_m": float(row.distance_m),
                "iso": int(row.iso),
                "frame": row.frame,
                "primary_condition": int(row.primary_condition),
                "reference_condition": int(row.reference_condition),
                "development_condition": int(row.development_condition),
                "arm": event["condition"],
                "seed": str(event["seed"]),
                "inference_seconds": float(event["inference_seconds"]),
            }
            record.update({key: float(event["metrics"][key]) for key in METRICS})
            records.append(record)
    frame_results = pd.DataFrame(records)
    raw = frame_results[frame_results["arm"] == "raw"].copy()
    reference = raw[raw["reference_condition"] == 1].set_index("distance_m")
    if len(reference) != 5 or reference.index.nunique() != 5:
        raise ProtocolError("Expected exactly five raw distance reference cells")
    raw_by_condition = raw.set_index("condition_cell")
    comparisons: list[dict[str, Any]] = []
    for row in frame_results[frame_results["arm"] != "raw"].itertuples(index=False):
        base = raw_by_condition.loc[row.condition_cell]
        out = row._asdict()
        for metric in METRICS:
            ref_value = float(reference.loc[row.distance_m, metric])
            raw_value = float(base[metric])
            restored_value = float(getattr(row, metric))
            out[f"{metric}_reference"] = ref_value
            out[f"{metric}_raw"] = raw_value
            out[f"{metric}_restored_minus_raw"] = restored_value - raw_value
            out[f"{metric}_raw_abs_reference_error"] = abs(raw_value - ref_value)
            out[f"{metric}_restored_abs_reference_error"] = abs(restored_value - ref_value)
            out[f"{metric}_delta_abs_reference_error"] = (
                abs(restored_value - ref_value) - abs(raw_value - ref_value)
            )
        comparisons.append(out)
    comparison = pd.DataFrame(comparisons)
    primary = comparison[comparison["primary_condition"] == 1].copy()
    if primary["condition_cell"].nunique() != int(protocol["reporting"]["expected_condition_cells"]):
        raise ProtocolError("Unexpected primary comparison cell count")
    bootstrap = protocol["reporting"]["bootstrap"]
    summaries: list[dict[str, Any]] = []
    for (arm, seed), subset in primary.groupby(["arm", "seed"], sort=True):
        for metric in METRICS:
            summaries.append(summary_record(subset, metric, arm, seed, bootstrap))
    arm_means = (
        primary.groupby(
            [
                "condition_cell", "speed_kmh", "distance_m", "iso",
                "frame", "primary_condition", "arm",
            ],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    for arm, subset in arm_means.groupby("arm", sort=True):
        for metric in METRICS:
            summaries.append(summary_record(subset, metric, arm, "three_seed_mean", bootstrap))
    summary = pd.DataFrame(summaries)
    frame_results.to_csv(output_root / "d0_cam2_response_rows.csv", index=False)
    comparison.to_csv(output_root / "d0_cam2_reference_comparisons.csv", index=False)
    summary.to_csv(output_root / "d0_cam2_run_summary.csv", index=False)
    final = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "status": "DESCRIPTIVE_RESPONSE_ONLY_NO_ACCURACY_GT",
        "counts": {
            "condition_cells": int(frame_results["condition_cell"].nunique()),
            "selected_frames": int(frame_results["frame"].count() // len(protocol["runs"])),
            "runs_per_frame": len(protocol["runs"]),
            "primary_condition_cells": int(primary["condition_cell"].nunique()),
        },
        "detector": {
            "architecture": protocol["detector"]["architecture"],
            "operating_threshold": protocol["detector"]["operating_threshold"],
            "role": protocol["detector"]["role"],
        },
        "fixed_run_and_arm_summaries": summaries,
        "claim_boundary": protocol["claim_boundary"],
        "interpretation": (
            "These values can show whether MERIT changes or preserves an independent detector's "
            "response across actual acquisition conditions. They cannot establish detection "
            "accuracy or distinguish recovered cracks from newly induced detector responses."
        ),
    }
    write_json_atomic(output_root / "d0_cam2_final.json", final, overwrite=True)
    write_json_atomic(
        output_root / "d0_cam2_status.json",
        {"stage": "complete", "created_utc": utc_now(), "status": final["status"]},
        overwrite=True,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


def self_test(protocol_path: Path, output_root: Path) -> None:
    protocol, cohort = load_frozen_cohort(protocol_path, output_root)
    device = require_cuda()
    models, detector, tiled_restore, tiled_detect = load_models(protocol, device)
    row = next(cohort.itertuples(index=False))
    image = read_bgr(Path(row.image_path))
    gray = cv2.cvtColor(image[:1024, :1024], cv2.COLOR_BGR2GRAY)
    del image
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    events = evaluate_gray(
        gray, protocol, models, detector, tiled_restore, tiled_detect, device
    )
    result = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "status": "PASS",
        "scope": "1024x1024 plumbing test only; not an analysis result",
        "events": len(events),
        "expected_events": len(protocol["runs"]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu": torch.cuda.get_device_name(device),
    }
    if result["events"] != result["expected_events"]:
        raise ProtocolError("Self-test run coverage mismatch")
    write_json_atomic(output_root / "d0_cam2_self_test.json", result, overwrite=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "self-test", "run", "aggregate"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build_cohort(arguments.protocol, arguments.output_root)
    elif arguments.command == "self-test":
        self_test(arguments.protocol, arguments.output_root)
    elif arguments.command == "run":
        run_evaluation(arguments.protocol, arguments.output_root)
    else:
        aggregate(arguments.protocol, arguments.output_root)


if __name__ == "__main__":
    main()
