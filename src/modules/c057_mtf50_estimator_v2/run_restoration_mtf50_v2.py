#!/usr/bin/env python3
"""Run the frozen MTF50-v2 estimator on RAW and six C006 restorers.

The inference stage is frame-resumable.  The aggregate stage uses one
identical seven-run complete-case ROI set and preserves positive and negative
results without changing the frozen estimator or efficacy rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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


BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_DIR = BASE.parent / "c007_d0_condition_eval"
ORIGINAL_PROTOCOL = ORIGINAL_DIR / "d0_cam1_protocol.json"
ORIGINAL_OUTPUT = ORIGINAL_DIR / "artifacts" / "cam1_v1"
RAW_EXECUTION_FREEZE = BASE / "protocol" / "mtf50_v2_execution_freeze.json"
DEVELOPMENT_SUMMARY = BASE / "artifacts" / "development_v2" / "development_summary.json"
DEVELOPMENT_RESULTS = BASE / "artifacts" / "development_v2" / "development_roi_results.csv"
HOLDOUT_SUMMARY = BASE / "artifacts" / "holdout_v2" / "holdout_summary.json"
HOLDOUT_RESULTS = BASE / "artifacts" / "holdout_v2" / "holdout_roi_results.csv"
DEFAULT_FREEZE = BASE / "protocol" / "mtf50_v2_restoration_execution_freeze.json"
DEFAULT_OUTPUT = BASE / "artifacts" / "restoration_v2"
SCHEMA = "MERIT-C057-MTF50-V2-RESTORATION-v1"
RUN_KEYS = ["raw_none", "off_0", "on_0", "off_1", "on_1", "off_2", "on_2"]

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import run_mtf50_v2 as raw_runner  # noqa: E402
from mtf50_estimator import estimate_mtf50_v2  # noqa: E402


class ProtocolError(RuntimeError):
    pass


def _load_original_module():
    path = ORIGINAL_DIR / "d0_cam1_eval.py"
    spec = importlib.util.spec_from_file_location("c057_original_d0_cam1_eval", path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"Could not import original evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D0 = _load_original_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def resolve_record_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def expected_release_paths(protocol: dict[str, Any], cohort_path: Path) -> dict[str, Path]:
    paths = {
        "restoration_runner": Path(__file__),
        "restoration_tests": BASE / "tests" / "test_restoration_mtf50_v2.py",
        "mtf50_estimator": BASE / "mtf50_estimator.py",
        "raw_runner": BASE / "run_mtf50_v2.py",
        "raw_execution_freeze": RAW_EXECUTION_FREEZE,
        "raw_execution_freeze_sidecar": RAW_EXECUTION_FREEZE.with_suffix(
            RAW_EXECUTION_FREEZE.suffix + ".sha256"
        ),
        "development_summary": DEVELOPMENT_SUMMARY,
        "development_results": DEVELOPMENT_RESULTS,
        "holdout_summary": HOLDOUT_SUMMARY,
        "holdout_results": HOLDOUT_RESULTS,
        "original_evaluator": ORIGINAL_DIR / "d0_cam1_eval.py",
        "original_protocol": ORIGINAL_PROTOCOL,
        "original_cohort": cohort_path,
        "original_cohort_freeze": ORIGINAL_OUTPUT / "d0_cam1_freeze.json",
        "model_code": Path(protocol["restoration"]["model_code"]["path"]),
        "operational_code": Path(protocol["restoration"]["operational_code"]["path"]),
    }
    for key, record in protocol["restoration"]["checkpoints"].items():
        paths[f"checkpoint:{key}"] = Path(record["path"])
    return paths


def validate_release_prerequisites() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Path]]:
    contract, _, raw_cohort, _, execution_hash = raw_runner.load_frozen_inputs(
        raw_runner.DEFAULT_CONTRACT,
        raw_runner.DEFAULT_FREEZE,
        RAW_EXECUTION_FREEZE,
    )
    development = read_json(DEVELOPMENT_SUMMARY)
    holdout = read_json(HOLDOUT_SUMMARY)
    if development.get("gate") != "PASS" or not all(development["gate_checks"].values()):
        raise ProtocolError("Development release gate is not PASS")
    if holdout.get("gate") != "PASS" or not all(holdout["gate_checks"].values()):
        raise ProtocolError("Official holdout release gate is not PASS")
    if holdout["inputs"]["execution_freeze_sha256"] != execution_hash:
        raise ProtocolError("Holdout does not bind the active RAW execution freeze")
    if holdout["artifacts"]["roi_results_sha256"] != sha256_file(HOLDOUT_RESULTS):
        raise ProtocolError("Holdout result CSV hash differs from its summary")
    if development["artifacts"]["roi_results_sha256"] != sha256_file(DEVELOPMENT_RESULTS):
        raise ProtocolError("Development result CSV hash differs from its summary")
    protocol, original_freeze, cohort = D0.load_frozen_cohort(
        ORIGINAL_PROTOCOL, ORIGINAL_OUTPUT
    )
    cohort_path = Path(original_freeze["cohort_path"])
    if sha256_file(cohort_path) != contract["inputs"]["cohort_sha256"]:
        raise ProtocolError("Original D0 cohort differs from the RAW-v2 cohort")
    if len(cohort) != 1737 or cohort["frame_id"].nunique() != 479:
        raise ProtocolError("Frozen cohort must contain 1,737 ROIs across 479 frames")
    if cohort["condition"].nunique() != 50:
        raise ProtocolError("Frozen cohort must contain exactly 50 condition cells")
    if float(read_json(raw_runner.DEFAULT_FREEZE)["selected_candidate"]["localization_gaussian_sigma_px"]) != 0.75:
        raise ProtocolError("Released localization sigma must remain 0.75 px")
    return protocol, original_freeze, cohort, expected_release_paths(protocol, cohort_path)


def _read_restoration_freeze(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ProtocolError("Restoration execution freeze or sidecar is missing")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    actual = sha256_file(path)
    if len(fields) != 2 or fields[1] != path.name or fields[0].lower() != actual:
        raise ProtocolError("Restoration execution freeze sidecar mismatch")
    payload = read_json(path)
    if payload.get("schema_version") != "MERIT-C057-MTF50-V2-RESTORATION-FREEZE-v1":
        raise ProtocolError("Unexpected restoration execution freeze schema")
    if payload.get("state") != "FROZEN_AFTER_RAW_HOLDOUT_PASS_BEFORE_RESTORATION_OUTPUTS":
        raise ProtocolError("Restoration execution freeze state is invalid")
    return payload, actual


def load_verified_environment(
    freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, str]:
    protocol, original_freeze, cohort, expected = validate_release_prerequisites()
    frozen, freeze_hash = _read_restoration_freeze(freeze_path)
    records = frozen.get("files", {})
    if set(records) != set(expected):
        raise ProtocolError("Restoration execution freeze file labels changed")
    for label, expected_path in expected.items():
        record = records[label]
        recorded_path = resolve_record_path(str(record.get("path", ""))).resolve()
        if recorded_path != expected_path.resolve():
            raise ProtocolError(f"Restoration freeze path mismatch: {label}")
        if str(record.get("sha256", "")).lower() != sha256_file(expected_path):
            raise ProtocolError(f"Restoration freeze hash mismatch: {label}")
    if float(frozen["selected_candidate"]["localization_gaussian_sigma_px"]) != 0.75:
        raise ProtocolError("Restoration freeze sigma changed")
    expected_counts = frozen["expected_counts"]
    if expected_counts != {
        "frames": 479,
        "roi_identities": 1737,
        "condition_cells": 50,
        "runs_including_raw": 7,
        "primary_condition_cells": 42,
    }:
        raise ProtocolError("Restoration freeze expected counts changed")
    return protocol, original_freeze, cohort, freeze_hash


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ProtocolError(f"Could not decode image: {path}")
    return image


def measure_rois(
    image: np.ndarray,
    frame: pd.DataFrame,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> list[dict[str, Any]]:
    measured: list[dict[str, Any]] = []
    for row in frame.sort_values("roi_id").itertuples(index=False):
        x1, x2 = int(row.x1) - x_offset, int(row.x2) - x_offset
        y1, y2 = int(row.y1) - y_offset, int(row.y2) - y_offset
        result = estimate_mtf50_v2(
            image[y1:y2, x1:x2],
            str(row.orientation),
            localization_sigma_px=0.75,
        )
        measured.append(
            {
                "roi_id": int(row.roi_id),
                "orientation": str(row.orientation),
                "mtf50": float(result.mtf50) if result.valid else None,
                "valid": result.valid,
                "reason": result.reason,
                "polarity": result.polarity,
                "line_support_rows": result.first_stage_support_rows,
                "total_rows": result.total_rows,
            }
        )
    return measured


def evaluate_frame(
    image: np.ndarray,
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    models: dict[tuple[str, str], torch.nn.Module],
    tiled_restore,
    device: torch.device,
) -> list[dict[str, Any]]:
    events = [
        {
            "condition": "raw",
            "seed": "none",
            "rois": measure_rois(image, frame),
            "inference_seconds": 0.0,
        }
    ]
    config = protocol["restoration"]
    crop, x0, y0 = D0.union_context(image, frame, int(config["context_margin_px"]))
    tiling = config["tiling"]
    for run in protocol["runs"]:
        if run["condition"] == "raw":
            continue
        started = time.perf_counter()
        model = models[(run["condition"], str(run["seed"]))]
        restored = tiled_restore(
            crop,
            model,
            device,
            tile_size=int(tiling["tile_size"]),
            overlap=int(tiling["overlap"]),
            batch_size=int(tiling["batch_size"]),
            blend_floor=float(tiling["positive_hann_floor"]),
            amp_dtype=config["amp_dtype"],
        )
        restored_u8 = np.rint(restored * np.float32(255.0)).astype(np.uint8)
        events.append(
            {
                "condition": run["condition"],
                "seed": str(run["seed"]),
                "rois": measure_rois(restored_u8, frame, x_offset=x0, y_offset=y0),
                "inference_seconds": time.perf_counter() - started,
            }
        )
        del restored, restored_u8
    return events


def progress_path(output: Path, index: int, frame_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", frame_id)
    return output / "progress" / f"{index:04d}_{safe}.json"


def validate_progress(
    path: Path,
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    freeze_hash: str,
) -> None:
    payload = read_json(path)
    if payload.get("schema_version") != SCHEMA:
        raise ProtocolError(f"Progress schema mismatch: {path}")
    if payload.get("restoration_freeze_sha256") != freeze_hash:
        raise ProtocolError(f"Progress freeze mismatch: {path}")
    if payload.get("frame_id") != str(frame.iloc[0]["frame_id"]):
        raise ProtocolError(f"Progress frame mismatch: {path}")
    if payload.get("image_sha256") != str(frame.iloc[0]["image_sha256"]):
        raise ProtocolError(f"Progress image mismatch: {path}")
    keys = [f"{event['condition']}_{event['seed']}" for event in payload["events"]]
    if keys != RUN_KEYS:
        raise ProtocolError(f"Progress run order/coverage mismatch: {path}")
    expected_rois = len(frame)
    if any(len(event["rois"]) != expected_rois for event in payload["events"]):
        raise ProtocolError(f"Progress ROI coverage mismatch: {path}")
    expected_identity = [
        (int(row.roi_id), str(row.orientation))
        for row in frame.sort_values("roi_id").itertuples(index=False)
    ]
    for event in payload["events"]:
        observed_identity = [
            (int(result["roi_id"]), str(result["orientation"]))
            for result in event["rois"]
        ]
        if observed_identity != expected_identity:
            raise ProtocolError(f"Progress ROI identity/orientation mismatch: {path}")


def run_smoke(freeze_path: Path) -> None:
    protocol, _, cohort, freeze_hash = load_verified_environment(freeze_path)
    device = D0.require_cuda()
    models, tiled_restore = D0.load_models(protocol, device)
    _, frame = D0.frame_records(cohort)[0]
    image = D0.verify_row_assets(frame)
    events = evaluate_frame(image, frame, protocol, models, tiled_restore, device)
    if [f"{event['condition']}_{event['seed']}" for event in events] != RUN_KEYS:
        raise ProtocolError("Smoke run coverage changed")
    if any(len(event["rois"]) != len(frame) for event in events):
        raise ProtocolError("Smoke ROI coverage changed")
    print(
        json.dumps(
            {
                "state": "PASS",
                "frame_id": str(frame.iloc[0]["frame_id"]),
                "roi_count": len(frame),
                "runs": RUN_KEYS,
                "restoration_freeze_sha256": freeze_hash,
                "gpu": torch.cuda.get_device_name(device),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def run_inference(freeze_path: Path, output: Path) -> None:
    protocol, _, cohort, freeze_hash = load_verified_environment(freeze_path)
    frames = D0.frame_records(cohort)
    already = 0
    for index, (_, frame) in enumerate(frames, start=1):
        path = progress_path(output, index, str(frame.iloc[0]["frame_id"]))
        if path.exists():
            validate_progress(path, frame, protocol, freeze_hash)
            already += 1
    if (output / "restoration_v2_final.json").exists():
        raise FileExistsError("Final output already exists")
    device = D0.require_cuda()
    models, tiled_restore = D0.load_models(protocol, device)
    started = time.perf_counter()
    processed = 0
    print(
        json.dumps(
            {
                "stage": "start",
                "frames": len(frames),
                "already_complete": already,
                "gpu": torch.cuda.get_device_name(device),
                "restoration_freeze_sha256": freeze_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    for index, (_, frame) in enumerate(frames, start=1):
        frame_id = str(frame.iloc[0]["frame_id"])
        path = progress_path(output, index, frame_id)
        if path.exists():
            continue
        frame_started = time.perf_counter()
        image = D0.verify_row_assets(frame)
        torch.cuda.reset_peak_memory_stats(device)
        events = evaluate_frame(image, frame, protocol, models, tiled_restore, device)
        payload = {
            "schema_version": SCHEMA,
            "frame_id": frame_id,
            "condition_cell": str(frame.iloc[0]["condition"]),
            "image_sha256": str(frame.iloc[0]["image_sha256"]),
            "restoration_freeze_sha256": freeze_hash,
            "events": events,
            "frame_elapsed_seconds": time.perf_counter() - frame_started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "committed_utc": utc_now(),
        }
        write_json_atomic(path, payload, overwrite=False)
        processed += 1
        completed = already + processed
        elapsed = time.perf_counter() - started
        average = elapsed / processed
        status = {
            "schema_version": SCHEMA,
            "state": "RUNNING" if completed < len(frames) else "INFERENCE_COMPLETE",
            "completed_frames": completed,
            "total_frames": len(frames),
            "remaining_frames": len(frames) - completed,
            "last_frame_id": frame_id,
            "session_average_seconds": average,
            "projected_remaining_seconds": average * (len(frames) - completed),
            "updated_utc": utc_now(),
            "restoration_freeze_sha256": freeze_hash,
        }
        write_json_atomic(output / "status.json", status, overwrite=True)
        if processed == 1 or processed % 10 == 0 or completed == len(frames):
            print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)


def bootstrap_mean_ci(values: np.ndarray, seed: int, count: int = 10000) -> list[float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(count, values.size))
    sampled = values[indices].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def complete_case_coverage(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = set(RUN_KEYS) - set(wide.columns)
    if missing:
        raise ProtocolError(f"Missing fixed run columns: {sorted(missing)}")
    complete = wide.dropna(subset=RUN_KEYS).copy()
    per_distance = []
    for distance in sorted(wide["distance_m"].unique()):
        total = int((wide["distance_m"] == distance).sum())
        valid = int((complete["distance_m"] == distance).sum())
        per_distance.append(
            {
                "distance_m": float(distance),
                "roi_identities": total,
                "complete_case_roi_identities": valid,
                "complete_case_fraction": valid / total,
            }
        )
    overall = len(complete) / len(wide)
    minimum_distance = min(row["complete_case_fraction"] for row in per_distance)
    checks = {
        "overall_complete_case_fraction": overall >= 0.95,
        "each_distance_complete_case_fraction": minimum_distance >= 0.90,
    }
    report = {
        "all_roi_identities": int(len(wide)),
        "complete_case_roi_identities": int(len(complete)),
        "complete_case_fraction": float(overall),
        "minimum_distance_complete_case_fraction": float(minimum_distance),
        "per_distance": per_distance,
        "gate_checks": checks,
        "gate": "PASS" if all(checks.values()) else "FAIL",
    }
    return complete, report


def _run_summary(condition: pd.DataFrame, arm: str, seed: str) -> dict[str, Any]:
    selected = condition[
        (condition["run_condition"] == arm)
        & (condition["seed"] == seed)
        & (condition["primary_condition"] == 1)
    ].copy()
    delta = (selected["abs_error"] - selected["raw_abs_error"]).to_numpy(float)
    return {
        "condition": arm,
        "seed": seed,
        "condition_cells": int(len(selected)),
        "mean_abs_error_cy_px": float(selected["abs_error"].mean()),
        "median_abs_error_cy_px": float(selected["abs_error"].median()),
        "mean_restored_minus_raw_abs_error_cy_px": float(delta.mean()),
        "median_restored_minus_raw_abs_error_cy_px": float(np.median(delta)),
        "mean_delta_95pct_condition_bootstrap_ci": bootstrap_mean_ci(
            delta, 20261020 + int(seed) + (100 if arm == "off" else 200)
        ),
        "conditions_improved": int(np.sum(delta < 0)),
        "conditions_tied": int(np.sum(delta == 0)),
        "conditions_worsened": int(np.sum(delta > 0)),
    }


def verify_raw_reproduction(roi: pd.DataFrame) -> dict[str, Any]:
    reference = pd.concat(
        [pd.read_csv(DEVELOPMENT_RESULTS), pd.read_csv(HOLDOUT_RESULTS)],
        ignore_index=True,
    )[["frame_id", "roi_id", "v2_mtf50", "v2_valid"]]
    raw = roi[roi["run_condition"] == "raw"][["frame_id", "roi_id", "mtf50"]]
    merged = raw.merge(reference, on=["frame_id", "roi_id"], validate="one_to_one")
    if len(merged) != 1737:
        raise ProtocolError("RAW reproduction does not cover all 1,737 frozen ROIs")
    left = merged["mtf50"].to_numpy(float)
    right = merged["v2_mtf50"].to_numpy(float)
    same_nan = np.isnan(left) == np.isnan(right)
    finite = np.isfinite(left) & np.isfinite(right)
    differences = np.abs(left[finite] - right[finite])
    maximum = float(differences.max()) if differences.size else 0.0
    if not bool(np.all(same_nan)) or maximum > 1e-12:
        raise ProtocolError("RAW restoration pass does not reproduce frozen RAW-v2 values")
    return {
        "roi_identities": 1737,
        "valid_roi_identities": int(np.isfinite(left).sum()),
        "maximum_absolute_difference_cy_px": maximum,
        "pass": True,
    }


def aggregate(freeze_path: Path, output: Path) -> None:
    protocol, _, cohort, freeze_hash = load_verified_environment(freeze_path)
    if (output / "restoration_v2_final.json").exists():
        raise FileExistsError("Final output already exists")
    frames = D0.frame_records(cohort)
    rows: list[dict[str, Any]] = []
    manifest = []
    for index, (_, frame) in enumerate(frames, start=1):
        frame_id = str(frame.iloc[0]["frame_id"])
        path = progress_path(output, index, frame_id)
        if not path.is_file():
            raise ProtocolError(f"Missing progress: {path}")
        validate_progress(path, frame, protocol, freeze_hash)
        payload = read_json(path)
        manifest.append(
            {
                "frame_id": frame_id,
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
        )
        metadata = {int(row.roi_id): row for row in frame.itertuples(index=False)}
        for event in payload["events"]:
            for result in event["rois"]:
                source = metadata[int(result["roi_id"])]
                rows.append(
                    {
                        "frame_id": frame_id,
                        "condition_cell": source.condition,
                        "speed_kmh": int(source.speed_kmh),
                        "distance_m": float(source.distance_m),
                        "iso": int(source.iso),
                        "frame": source.frame,
                        "roi_id": int(result["roi_id"]),
                        "orientation": result["orientation"],
                        "run_condition": event["condition"],
                        "seed": str(event["seed"]),
                        "mtf50": np.nan if result["mtf50"] is None else float(result["mtf50"]),
                        "valid": bool(result["valid"]),
                        "reason": result["reason"],
                        "polarity": result["polarity"],
                        "line_support_rows": int(result["line_support_rows"]),
                        "total_rows": int(result["total_rows"]),
                        "development_condition": int(source.development_condition),
                        "reference_condition": int(source.reference_condition),
                        "primary_condition": int(source.primary_condition),
                    }
                )
    roi = pd.DataFrame(rows)
    if len(roi) != 1737 * 7:
        raise ProtocolError(f"ROI result count {len(roi)} != 12,159")
    raw_reproduction = verify_raw_reproduction(roi)
    roi["run_key"] = roi["run_condition"] + "_" + roi["seed"]
    identity = [
        "frame_id",
        "condition_cell",
        "speed_kmh",
        "distance_m",
        "iso",
        "frame",
        "roi_id",
        "orientation",
        "development_condition",
        "reference_condition",
        "primary_condition",
    ]
    wide = roi.pivot(index=identity, columns="run_key", values="mtf50").reset_index()
    if len(wide) != 1737:
        raise ProtocolError("ROI identities changed during seven-run pivot")
    complete, coverage = complete_case_coverage(wide)
    complete["roi_key"] = (
        complete["frame_id"] + "/roi" + complete["roi_id"].astype(str)
    )
    long = complete.melt(
        id_vars=identity + ["roi_key"],
        value_vars=RUN_KEYS,
        var_name="run_key",
        value_name="mtf50",
    )
    long[["run_condition", "seed"]] = long["run_key"].str.split(
        "_", n=1, expand=True
    )
    frame_results = (
        long.groupby(
            [
                "frame_id",
                "condition_cell",
                "speed_kmh",
                "distance_m",
                "iso",
                "run_condition",
                "seed",
                "development_condition",
                "reference_condition",
                "primary_condition",
            ],
            as_index=False,
        )
        .agg(mtf50=("mtf50", "mean"), complete_rois=("roi_key", "count"))
    )
    condition = (
        frame_results.groupby(
            [
                "condition_cell",
                "speed_kmh",
                "distance_m",
                "iso",
                "run_condition",
                "seed",
                "development_condition",
                "reference_condition",
                "primary_condition",
            ],
            as_index=False,
        )
        .agg(
            median_mtf50=("mtf50", "median"),
            mean_mtf50=("mtf50", "mean"),
            frames=("frame_id", "count"),
        )
    )
    reference = condition[
        (condition["run_condition"] == "raw")
        & (condition["reference_condition"] == 1)
    ][["distance_m", "median_mtf50"]].rename(
        columns={"median_mtf50": "reference_mtf50"}
    )
    if len(reference) != 5 or reference["distance_m"].nunique() != 5:
        raise ProtocolError("Distance-specific RAW reference coverage changed")
    condition = condition.merge(reference, on="distance_m", validate="many_to_one")
    condition["abs_error"] = np.abs(
        condition["median_mtf50"] - condition["reference_mtf50"]
    )
    raw_error = condition[condition["run_condition"] == "raw"][
        ["condition_cell", "abs_error"]
    ].rename(columns={"abs_error": "raw_abs_error"})
    condition = condition.merge(raw_error, on="condition_cell", validate="many_to_one")
    run_summaries = [
        _run_summary(condition, arm, seed)
        for seed in ("0", "1", "2")
        for arm in ("off", "on")
    ]
    arm_summaries = []
    for arm in ("off", "on"):
        selected = condition[
            (condition["run_condition"] == arm)
            & (condition["primary_condition"] == 1)
        ]
        averaged = selected.groupby("condition_cell", as_index=False).agg(
            abs_error=("abs_error", "mean"),
            raw_abs_error=("raw_abs_error", "first"),
        )
        delta = (averaged["abs_error"] - averaged["raw_abs_error"]).to_numpy(float)
        arm_summaries.append(
            {
                "arm": arm,
                "condition_cells": int(len(averaged)),
                "mean_abs_error_cy_px": float(averaged["abs_error"].mean()),
                "mean_restored_minus_raw_abs_error_cy_px": float(delta.mean()),
                "mean_delta_95pct_condition_bootstrap_ci": bootstrap_mean_ci(
                    delta, 20261120 + (100 if arm == "off" else 200)
                ),
                "conditions_improved": int(np.sum(delta < 0)),
                "conditions_tied": int(np.sum(delta == 0)),
                "conditions_worsened": int(np.sum(delta > 0)),
            }
        )
    off = condition[
        (condition["run_condition"] == "off")
        & (condition["primary_condition"] == 1)
    ][["condition_cell", "seed", "abs_error"]].rename(
        columns={"abs_error": "off_abs_error"}
    )
    on = condition[
        (condition["run_condition"] == "on")
        & (condition["primary_condition"] == 1)
    ][["condition_cell", "seed", "abs_error"]].rename(
        columns={"abs_error": "on_abs_error"}
    )
    task_pairs = on.merge(off, on=["condition_cell", "seed"], validate="one_to_one")
    task_pairs["on_minus_off_abs_error"] = (
        task_pairs["on_abs_error"] - task_pairs["off_abs_error"]
    )
    if len(task_pairs) != 42 * 3:
        raise ProtocolError("Task ON/OFF primary pair count changed")

    manifest_payload = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "records": manifest,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "roi_results": output / "restoration_v2_roi_results.csv",
        "complete_case_rois": output / "restoration_v2_complete_case_rois.csv",
        "frame_results": output / "restoration_v2_frame_results.csv",
        "condition_results": output / "restoration_v2_condition_results.csv",
        "task_pairs": output / "restoration_v2_task_pairs.csv",
    }
    write_csv_atomic(output_paths["roi_results"], roi)
    write_csv_atomic(output_paths["complete_case_rois"], complete)
    write_csv_atomic(output_paths["frame_results"], frame_results)
    write_csv_atomic(output_paths["condition_results"], condition)
    write_csv_atomic(output_paths["task_pairs"], task_pairs)
    manifest_path = output / "restoration_v2_progress_manifest.json"
    write_json_atomic(manifest_path, manifest_payload, overwrite=False)
    final = {
        "schema_version": SCHEMA,
        "state": "COMPLETE" if coverage["gate"] == "PASS" else "COMPLETE_COVERAGE_GATE_FAILED",
        "created_utc": utc_now(),
        "analysis_status": "retrospective frozen measurement-method re-evaluation; not prospective confirmation",
        "counts": {
            "frames": 479,
            "roi_identities": 1737,
            "condition_cells": 50,
            "runs_including_raw": 7,
            "primary_condition_cells": 42,
        },
        "raw_reproduction": raw_reproduction,
        "complete_case_coverage": coverage,
        "fixed_run_summaries": run_summaries,
        "arm_three_seed_summaries": arm_summaries,
        "task_on_minus_off": {
            "pairs": int(len(task_pairs)),
            "condition_cells": int(task_pairs["condition_cell"].nunique()),
            "seeds": int(task_pairs["seed"].nunique()),
            "mean_on_minus_off_abs_error_cy_px": float(
                task_pairs["on_minus_off_abs_error"].mean()
            ),
            "median_on_minus_off_abs_error_cy_px": float(
                task_pairs["on_minus_off_abs_error"].median()
            ),
        },
        "interpretation_gate": (
            "The identical seven-run complete-case coverage gate passed; report the frozen positive or negative efficacy result."
            if coverage["gate"] == "PASS"
            else "Do not interpret restoration efficacy because identical seven-run complete-case coverage failed."
        ),
        "claim_limits": [
            "The original v1 86.82% failure and negative sensitivity remain preserved.",
            "MTF50 v2 is not claimed as ISO 12233 compliance or optical calibration.",
            "This retrospective analysis does not establish natural-crack detection or physical-width efficacy.",
        ],
        "inputs": {
            "restoration_freeze_sha256": freeze_hash,
            "raw_execution_freeze_sha256": sha256_file(RAW_EXECUTION_FREEZE),
            "holdout_summary_sha256": sha256_file(HOLDOUT_SUMMARY),
            "holdout_results_sha256": sha256_file(HOLDOUT_RESULTS),
            "original_protocol_sha256": sha256_file(ORIGINAL_PROTOCOL),
            "original_cohort_sha256": sha256_file(Path(read_json(ORIGINAL_OUTPUT / "d0_cam1_freeze.json")["cohort_path"])),
            "progress_manifest_sha256": sha256_file(manifest_path),
            "artifacts": {
                label: {
                    "path": relative_or_absolute(path),
                    "sha256": sha256_file(path),
                }
                for label, path in output_paths.items()
            },
        },
    }
    write_json_atomic(output / "restoration_v2_final.json", final, overwrite=False)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("command", choices=("smoke", "inference", "aggregate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "smoke":
        run_smoke(args.freeze)
    elif args.command == "inference":
        run_inference(args.freeze, args.output)
    else:
        aggregate(args.freeze, args.output)


if __name__ == "__main__":
    main()
