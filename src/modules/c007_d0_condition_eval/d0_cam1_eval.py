#!/usr/bin/env python3
"""Frozen D0 cam1 50-condition actual-degradation evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.torch_version import TorchVersion


SCHEMA = "MERIT-D0-CAM1-50COND-v1"


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


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ProtocolError(f"Could not decode image: {path}")
    return image


def parse_condition(name: str) -> tuple[int, float, int]:
    matched = re.fullmatch(r"(60|80)km_(2\.5|3\.5|4\.5|5\.5|6\.5)m_ISO(100|200|400|800|1600)", name)
    if not matched:
        raise ProtocolError(f"Unexpected condition name: {name}")
    return int(matched.group(1)), float(matched.group(2)), int(matched.group(3))


def parse_multi_csv(path: Path) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    geometry_header = next(
        (index for index, line in enumerate(lines) if line.startswith("N,Distance %")),
        None,
    )
    primary_header = next(
        (index for index, line in enumerate(lines) if line.startswith("N,MTF50 (Cy/Pxl)")),
        None,
    )
    if geometry_header is None or primary_header is None:
        raise ProtocolError(f"Could not locate Imatest ROI sections: {path}")
    geometry: dict[int, tuple[int, int, int, int]] = {}
    for line in lines[geometry_header + 1 :]:
        if not line.strip():
            break
        row = next(csv.reader([line]))
        try:
            roi_id = int(row[0].strip())
        except (ValueError, IndexError):
            continue
        geometry[roi_id] = tuple(int(row[index].strip()) for index in range(3, 7))
    imatest: dict[int, float] = {}
    for line in lines[primary_header + 1 :]:
        if not line.strip() or line.startswith("Summary"):
            break
        row = next(csv.reader([line]))
        try:
            imatest[int(row[0].strip())] = float(row[1].strip())
        except (ValueError, IndexError):
            continue
    size_line = next((line for line in lines if line.startswith("nwid_save,")), None)
    height_line = next((line for line in lines if line.startswith("nht_save,")), None)
    if size_line is None or height_line is None:
        raise ProtocolError(f"Missing saved image dimensions: {path}")
    width = int(next(csv.reader([size_line]))[1].strip())
    height = int(next(csv.reader([height_line]))[1].strip())
    if not geometry or sorted(geometry) != sorted(imatest):
        raise ProtocolError(f"Imatest geometry/result ROI sets differ or are empty: {path}")
    records = []
    for roi_id in sorted(geometry):
        x1, y1, x2, y2 = geometry[roi_id]
        roi_height, roi_width = y2 - y1, x2 - x1
        records.append(
            {
                "roi_id": roi_id,
                "orientation": "vertical_edge" if roi_height >= roi_width else "horizontal_edge",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "imatest_mtf50": imatest[roi_id],
            }
        )
    return records, (height, width)


def locate_frame_assets(root: Path, condition: str, frame: str) -> tuple[Path, Path]:
    directory = root / condition
    candidates = []
    for child in directory.iterdir():
        if not child.is_dir() or child.name == "Results":
            continue
        image = child / frame
        multi = child / "Results" / f"{Path(frame).stem}_Y_multi.csv"
        if image.is_file() and multi.is_file():
            candidates.append((image, multi))
    if len(candidates) != 1:
        raise ProtocolError(
            f"Expected one chart extract + multi CSV for {condition}/{frame}, found {len(candidates)}"
        )
    return candidates[0]


def verify_static_assets(protocol_path: Path, protocol: dict[str, Any]) -> dict[str, str]:
    verified = {"protocol": sha256_file(protocol_path)}
    historical = protocol["source"]["historical_summary_csv"]
    verified["historical_summary_csv"] = require_hash(
        Path(historical["path"]), historical["sha256"], "historical cam1 summary"
    )
    for label in ("model_code", "operational_code"):
        record = protocol["restoration"][label]
        verified[label] = require_hash(Path(record["path"]), record["sha256"], label)
    for key, record in protocol["restoration"]["checkpoints"].items():
        verified[f"checkpoint:{key}"] = require_hash(
            Path(record["path"]), record["sha256"], key
        )
    return verified


def build_cohort(protocol_path: Path, output_root: Path) -> None:
    protocol = load_protocol(protocol_path)
    verified = verify_static_assets(protocol_path, protocol)
    root = Path(protocol["source"]["root"])
    development = set(protocol["development_conditions"])
    records: list[dict[str, Any]] = []
    image_hash_cache: dict[Path, str] = {}
    multi_hash_cache: dict[Path, str] = {}
    seen_frames: set[str] = set()
    multi_candidates = sorted(root.glob("*/*/Results/*_Y_multi.csv"))
    for multi_path in multi_candidates:
        condition = multi_path.parents[2].name
        frame = multi_path.stem.removesuffix("_Y_multi") + ".png"
        speed, distance, iso = parse_condition(condition)
        image_path = multi_path.parent.parent / frame
        if not image_path.is_file():
            continue
        frame_id = f"{condition}/{frame}"
        if frame_id in seen_frames:
            raise ProtocolError(f"Duplicate complete multi-ROI frame: {frame_id}")
        seen_frames.add(frame_id)
        image = read_gray(image_path)
        roi_records, expected_shape = parse_multi_csv(multi_path)
        if image.shape != expected_shape:
            raise ProtocolError(
                f"Image/multi dimensions differ for {image_path}: {image.shape} vs {expected_shape}"
            )
        image_hash = image_hash_cache.setdefault(image_path, sha256_file(image_path))
        multi_hash = multi_hash_cache.setdefault(multi_path, sha256_file(multi_path))
        reference_condition = f"60km_{distance:.1f}m_ISO100"
        external_frame_mean = float(np.mean([roi["imatest_mtf50"] for roi in roi_records]))
        for roi in roi_records:
            if not (0 <= roi["x1"] < roi["x2"] <= image.shape[1]):
                raise ProtocolError(f"ROI x out of bounds: {image_path} {roi}")
            if not (0 <= roi["y1"] < roi["y2"] <= image.shape[0]):
                raise ProtocolError(f"ROI y out of bounds: {image_path} {roi}")
            records.append(
                {
                    "frame_id": frame_id,
                    "condition": condition,
                    "speed_kmh": speed,
                    "distance_m": f"{distance:.1f}",
                    "iso": iso,
                    "frame": frame,
                    "roi_id": roi["roi_id"],
                    "orientation": roi["orientation"],
                    "x1": roi["x1"],
                    "y1": roi["y1"],
                    "x2": roi["x2"],
                    "y2": roi["y2"],
                    "imatest_roi_mtf50": f"{roi['imatest_mtf50']:.10g}",
                    "preexisting_frame_mtf50": f"{external_frame_mean:.10g}",
                    "image_path": str(image_path),
                    "image_sha256": image_hash,
                    "multi_path": str(multi_path),
                    "multi_sha256": multi_hash,
                    "development_condition": int(condition in development),
                    "reference_condition": int(condition == reference_condition),
                    "primary_condition": int(
                        condition not in development and condition != reference_condition
                    ),
                }
            )
    if len(seen_frames) != 479 or len(records) < 479:
        raise ProtocolError(
            f"Expected all 479 complete multi-ROI frames, found {len(seen_frames)} frames / {len(records)} ROIs"
        )
    if len(records) < len(seen_frames):
        raise ProtocolError("Every frozen frame must retain at least one historical Imatest ROI")
    condition_counts = Counter(record["condition"] for record in records if record["roi_id"] == 1)
    if min(condition_counts.values()) != 5 or max(condition_counts.values()) != 13 or len(condition_counts) != 50:
        raise ProtocolError("Unexpected per-condition frame coverage")
    primary_count = len({record["condition"] for record in records if record["primary_condition"]})
    expected_primary = int(protocol["primary_endpoint"]["expected_condition_cells"])
    if primary_count != expected_primary:
        raise ProtocolError(f"Primary cell count {primary_count} != {expected_primary}")
    output_root.mkdir(parents=True, exist_ok=True)
    cohort_path = output_root / "d0_cam1_cohort.csv"
    if cohort_path.exists():
        raise FileExistsError(cohort_path)
    with cohort_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    freeze = {
        "schema_version": SCHEMA,
        "state": "COHORT_FROZEN_BEFORE_MODEL_OUTPUTS",
        "created_utc": utc_now(),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": verified["protocol"],
        "evaluation_code_path": str(Path(__file__).resolve()),
        "evaluation_code_sha256": sha256_file(Path(__file__)),
        "cohort_path": str(cohort_path.resolve()),
        "cohort_sha256": sha256_file(cohort_path),
        "frames": len(seen_frames),
        "roi_records": len(records),
        "roi_count_distribution": dict(
            sorted(Counter(Counter(record["frame_id"] for record in records).values()).items())
        ),
        "condition_cells": len(condition_counts),
        "condition_frame_counts": dict(sorted(condition_counts.items())),
        "primary_condition_cells": primary_count,
        "verified_static_assets": verified,
    }
    write_json_atomic(output_root / "d0_cam1_freeze.json", freeze, overwrite=False)
    print(json.dumps(freeze, ensure_ascii=False, sort_keys=True))


def load_frozen_cohort(
    protocol_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    protocol = load_protocol(protocol_path)
    verify_static_assets(protocol_path, protocol)
    freeze_path = output_root / "d0_cam1_freeze.json"
    freeze = read_json(freeze_path)
    if freeze.get("state") != "COHORT_FROZEN_BEFORE_MODEL_OUTPUTS":
        raise ProtocolError("Cohort freeze state is invalid")
    require_hash(protocol_path, freeze["protocol_sha256"], "frozen protocol")
    require_hash(Path(__file__), freeze["evaluation_code_sha256"], "frozen evaluation code")
    cohort_path = Path(freeze["cohort_path"])
    require_hash(cohort_path, freeze["cohort_sha256"], "frozen cohort")
    cohort = pd.read_csv(cohort_path, dtype={"seed": str} if False else None)
    if cohort["frame_id"].nunique() != int(freeze["frames"]):
        raise ProtocolError("Frozen cohort frame count changed")
    return protocol, freeze, cohort


def inhouse_mtf50(roi: np.ndarray, orientation: str) -> float:
    if roi.ndim != 2 or min(roi.shape) < 32:
        raise ValueError("ROI must be a 2D edge crop of at least 32 px per axis")
    data = roi.astype(np.float64)
    if orientation == "horizontal_edge":
        data = data.T
    elif orientation != "vertical_edge":
        raise ValueError(f"Unknown edge orientation: {orientation}")
    height, width = data.shape
    rows = np.arange(height, dtype=np.float64)
    gradient = np.abs(np.gradient(data, axis=1))
    edge = gradient.argmax(axis=1).astype(np.float64)
    slope, intercept = np.polyfit(rows, edge, 1)
    residual = np.abs(edge - (slope * rows + intercept))
    good = residual < 6.0
    if int(good.sum()) < max(20, int(0.5 * height)):
        return math.nan
    slope, intercept = np.polyfit(rows[good], edge[good], 1)
    oversample = 4
    columns = np.arange(width, dtype=np.float64)
    position = (columns[None, :] - (slope * rows[:, None] + intercept)) * oversample
    selected = np.abs(position) < 30 * oversample
    positions = position[selected].ravel()
    values = data[selected].ravel()
    if positions.size < 1000 or np.ptp(values) < 5.0:
        return math.nan
    order = np.argsort(positions)
    positions, values = positions[order], values[order]
    bins = np.arange(positions.min(), positions.max(), 1.0)
    indices = np.digitize(positions, bins)
    esf = np.asarray(
        [
            values[indices == index].mean() if np.any(indices == index) else np.nan
            for index in range(1, len(bins))
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(esf)
    if int(valid.sum()) < 32:
        return math.nan
    esf = np.interp(np.arange(len(esf)), np.flatnonzero(valid), esf[valid])
    lsf = np.gradient(esf) * np.hanning(len(esf))
    spectrum = np.abs(np.fft.rfft(lsf))
    if not np.isfinite(spectrum[0]) or spectrum[0] <= 1e-12:
        return math.nan
    mtf = spectrum / spectrum[0]
    frequency = np.fft.rfftfreq(len(lsf), d=1.0 / oversample)
    band = frequency <= 0.6
    frequency, mtf = frequency[band], mtf[band]
    crossings = np.flatnonzero(mtf < 0.5)
    if crossings.size == 0 or crossings[0] == 0:
        return math.nan
    index = int(crossings[0])
    value = float(
        np.interp(0.5, [mtf[index], mtf[index - 1]], [frequency[index], frequency[index - 1]])
    )
    return value if 0.0 < value <= 0.6 else math.nan


def import_inference_components(protocol: dict[str, Any]):
    execution_dir = Path(protocol["restoration"]["model_code"]["path"]).parent
    if str(execution_dir) not in sys.path:
        sys.path.insert(0, str(execution_dir))
    from c006_model import NAFNetUNet
    from c006_operational_inference import tiled_restore_gray

    return NAFNetUNet, tiled_restore_gray


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ProtocolError("D0 50-condition model evaluation requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def load_models(protocol: dict[str, Any], device: torch.device):
    model_class, tiled_restore = import_inference_components(protocol)
    models: dict[tuple[str, str], torch.nn.Module] = {}
    with torch.serialization.safe_globals([TorchVersion]):
        for run in protocol["runs"]:
            if run["condition"] == "raw":
                continue
            record = protocol["restoration"]["checkpoints"][run["checkpoint_key"]]
            checkpoint = torch.load(Path(record["path"]), map_location="cpu", weights_only=True)
            model = model_class(width=int(protocol["restoration"]["model_width"]))
            model.load_state_dict(checkpoint["state"], strict=True)
            models[(run["condition"], run["seed"])] = model.to(device).eval()
    return models, tiled_restore


def frame_records(cohort: pd.DataFrame):
    return list(cohort.sort_values(["condition", "frame", "roi_id"]).groupby("frame_id", sort=True))


def verify_row_assets(frame: pd.DataFrame) -> np.ndarray:
    image_path = Path(frame.iloc[0]["image_path"])
    require_hash(image_path, str(frame.iloc[0]["image_sha256"]), "cohort image")
    multi_path = Path(frame.iloc[0]["multi_path"])
    require_hash(multi_path, str(frame.iloc[0]["multi_sha256"]), "cohort multi CSV")
    return read_gray(image_path)


def union_context(image: np.ndarray, frame: pd.DataFrame, margin: int) -> tuple[np.ndarray, int, int]:
    x0 = max(0, int(frame["x1"].min()) - margin)
    y0 = max(0, int(frame["y1"].min()) - margin)
    x1 = min(image.shape[1], int(frame["x2"].max()) + margin)
    y1 = min(image.shape[0], int(frame["y2"].max()) + margin)
    crop = image[y0:y1, x0:x1]
    pad_bottom = max(0, 448 - crop.shape[0])
    pad_right = max(0, 448 - crop.shape[1])
    if pad_bottom or pad_right:
        crop = cv2.copyMakeBorder(
            crop, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_REFLECT_101
        )
    return crop, x0, y0


def measure_rois(
    image: np.ndarray, frame: pd.DataFrame, *, x_offset: int = 0, y_offset: int = 0
) -> list[dict[str, Any]]:
    measured = []
    for _, row in frame.sort_values("roi_id").iterrows():
        x1, x2 = int(row["x1"]) - x_offset, int(row["x2"]) - x_offset
        y1, y2 = int(row["y1"]) - y_offset, int(row["y2"]) - y_offset
        roi = image[y1:y2, x1:x2]
        value = inhouse_mtf50(roi, str(row["orientation"]))
        measured.append(
            {
                "roi_id": int(row["roi_id"]),
                "orientation": str(row["orientation"]),
                "mtf50": None if not math.isfinite(value) else value,
                "imatest_roi_mtf50": float(row["imatest_roi_mtf50"]),
            }
        )
    return measured


def progress_path(output_root: Path, index: int, frame_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", frame_id)
    return output_root / "progress" / f"{index:04d}_{safe}.json"


def expected_run_keys(protocol: dict[str, Any]) -> list[tuple[str, str]]:
    return [(run["condition"], str(run["seed"])) for run in protocol["runs"]]


def validate_progress(path: Path, frame: pd.DataFrame, protocol: dict[str, Any]) -> None:
    payload = read_json(path)
    if payload.get("frame_id") != str(frame.iloc[0]["frame_id"]):
        raise ProtocolError(f"Progress frame mismatch: {path}")
    if payload.get("image_sha256") != str(frame.iloc[0]["image_sha256"]):
        raise ProtocolError(f"Progress image mismatch: {path}")
    keys = [(event["condition"], str(event["seed"])) for event in payload["events"]]
    if keys != expected_run_keys(protocol):
        raise ProtocolError(f"Progress run coverage mismatch: {path}")
    expected_rois = len(frame)
    if any(len(event["rois"]) != expected_rois for event in payload["events"]):
        raise ProtocolError(f"Progress ROI coverage mismatch: {path}")


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
    crop, x0, y0 = union_context(image, frame, int(config["context_margin_px"]))
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


def run_evaluation(protocol_path: Path, output_root: Path) -> None:
    protocol, freeze, cohort = load_frozen_cohort(protocol_path, output_root)
    frames = frame_records(cohort)
    already = 0
    for index, (_, frame) in enumerate(frames, start=1):
        path = progress_path(output_root, index, str(frame.iloc[0]["frame_id"]))
        if path.exists():
            validate_progress(path, frame, protocol)
            already += 1
    if (output_root / "d0_cam1_final.json").exists():
        raise FileExistsError("Final output already exists")
    device = require_cuda()
    models, tiled_restore = load_models(protocol, device)
    session_started = time.perf_counter()
    processed = 0
    print(
        json.dumps(
            {"stage": "start", "frames": len(frames), "already_complete": already, "gpu": torch.cuda.get_device_name(device)},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    for index, (_, frame) in enumerate(frames, start=1):
        frame_id = str(frame.iloc[0]["frame_id"])
        path = progress_path(output_root, index, frame_id)
        if path.exists():
            continue
        frame_started = time.perf_counter()
        image = verify_row_assets(frame)
        torch.cuda.reset_peak_memory_stats(device)
        events = evaluate_frame(image, frame, protocol, models, tiled_restore, device)
        payload = {
            "schema_version": SCHEMA,
            "frame_id": frame_id,
            "condition_cell": str(frame.iloc[0]["condition"]),
            "image_sha256": str(frame.iloc[0]["image_sha256"]),
            "events": events,
            "frame_elapsed_seconds": time.perf_counter() - frame_started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "committed_utc": utc_now(),
        }
        write_json_atomic(path, payload, overwrite=False)
        processed += 1
        completed = already + processed
        elapsed = time.perf_counter() - session_started
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
            "freeze_sha256": sha256_file(output_root / "d0_cam1_freeze.json"),
        }
        write_json_atomic(output_root / "status.json", status, overwrite=True)
        if processed == 1 or processed % 10 == 0 or completed == len(frames):
            print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return math.nan
    return float(spearmanr(left[mask], right[mask]).statistic)


def bootstrap_mean_ci(values: np.ndarray, count: int, seed: int) -> list[float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(count, values.size))
    sampled = values[indices].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def run_summary(
    merged: pd.DataFrame, protocol: dict[str, Any], condition: str, seed: str
) -> dict[str, Any]:
    subset = merged[
        (merged["run_condition"] == condition)
        & (merged["seed"].astype(str) == str(seed))
        & (merged["primary_condition"] == 1)
    ].copy()
    delta = (subset["abs_error"] - subset["raw_abs_error"]).to_numpy(float)
    bootstrap = protocol["primary_endpoint"]["bootstrap"]
    return {
        "condition": condition,
        "seed": str(seed),
        "condition_cells": int(len(subset)),
        "mean_abs_error_cy_px": float(subset["abs_error"].mean()),
        "median_abs_error_cy_px": float(subset["abs_error"].median()),
        "mean_restored_minus_raw_abs_error_cy_px": float(np.mean(delta)),
        "median_restored_minus_raw_abs_error_cy_px": float(np.median(delta)),
        "mean_delta_95pct_cluster_bootstrap_ci": bootstrap_mean_ci(
            delta, int(bootstrap["resamples"]), int(bootstrap["seed"]) + int(seed)
        ),
        "conditions_improved": int(np.sum(delta < 0)),
        "conditions_tied": int(np.sum(delta == 0)),
        "conditions_worsened": int(np.sum(delta > 0)),
        "overshoot_conditions": int(np.sum(subset["overshoot"])),
    }


def aggregate(protocol_path: Path, output_root: Path) -> None:
    protocol, freeze, cohort = load_frozen_cohort(protocol_path, output_root)
    frames = frame_records(cohort)
    roi_rows: list[dict[str, Any]] = []
    manifest = []
    for index, (_, frame) in enumerate(frames, start=1):
        frame_id = str(frame.iloc[0]["frame_id"])
        path = progress_path(output_root, index, frame_id)
        if not path.is_file():
            raise ProtocolError(f"Missing progress: {path}")
        validate_progress(path, frame, protocol)
        payload = read_json(path)
        manifest.append({"frame_id": frame_id, "path": str(path.relative_to(output_root)), "sha256": sha256_file(path)})
        metadata = frame.iloc[0]
        roi_meta = {int(row["roi_id"]): row for _, row in frame.iterrows()}
        for event in payload["events"]:
            for result in event["rois"]:
                source = roi_meta[int(result["roi_id"])]
                roi_rows.append(
                    {
                        "frame_id": frame_id,
                        "condition_cell": source["condition"],
                        "speed_kmh": int(source["speed_kmh"]),
                        "distance_m": float(source["distance_m"]),
                        "iso": int(source["iso"]),
                        "frame": source["frame"],
                        "roi_id": int(result["roi_id"]),
                        "orientation": result["orientation"],
                        "run_condition": event["condition"],
                        "seed": str(event["seed"]),
                        "mtf50": np.nan if result["mtf50"] is None else float(result["mtf50"]),
                        "imatest_roi_mtf50": float(source["imatest_roi_mtf50"]),
                        "preexisting_frame_mtf50": float(source["preexisting_frame_mtf50"]),
                        "development_condition": int(source["development_condition"]),
                        "reference_condition": int(source["reference_condition"]),
                        "primary_condition": int(source["primary_condition"]),
                    }
                )
    roi = pd.DataFrame(roi_rows)
    expected_rows = len(cohort) * len(protocol["runs"])
    if len(roi) != expected_rows:
        raise ProtocolError(f"ROI result count {len(roi)} != {expected_rows}")
    raw = roi[roi["run_condition"] == "raw"].copy()
    raw_valid = float(raw["mtf50"].notna().mean())
    roi_spearman = finite_spearman(raw["mtf50"].to_numpy(), raw["imatest_roi_mtf50"].to_numpy())
    frame = (
        roi.groupby(
            [
                "frame_id", "condition_cell", "speed_kmh", "distance_m", "iso",
                "run_condition", "seed", "development_condition", "reference_condition", "primary_condition"
            ],
            as_index=False,
        )
        .agg(mtf50=("mtf50", "mean"), valid_rois=("mtf50", "count"), preexisting_frame_mtf50=("preexisting_frame_mtf50", "first"))
    )
    raw_frame = frame[frame["run_condition"] == "raw"]
    frame_spearman = finite_spearman(
        raw_frame["mtf50"].to_numpy(), raw_frame["preexisting_frame_mtf50"].to_numpy()
    )
    gates = protocol["measurement"]["raw_validation_gate"]
    validation_pass = (
        raw_valid >= float(gates["minimum_valid_roi_fraction"])
        and roi_spearman >= float(gates["minimum_roi_spearman_vs_imatest"])
        and frame_spearman >= float(gates["minimum_frame_spearman_vs_preexisting_imatest_mean"])
    )
    condition = (
        frame.groupby(
            [
                "condition_cell", "speed_kmh", "distance_m", "iso", "run_condition", "seed",
                "development_condition", "reference_condition", "primary_condition"
            ],
            as_index=False,
        )
        .agg(median_mtf50=("mtf50", "median"), mean_mtf50=("mtf50", "mean"), valid_frames=("mtf50", "count"), total_frames=("frame_id", "count"))
    )
    raw_reference = condition[
        (condition["run_condition"] == "raw") & (condition["reference_condition"] == 1)
    ][["distance_m", "median_mtf50"]].rename(columns={"median_mtf50": "reference_mtf50"})
    if len(raw_reference) != 5 or raw_reference["distance_m"].nunique() != 5:
        raise ProtocolError("Distance-specific raw reference coverage changed")
    condition = condition.merge(raw_reference, on="distance_m", how="left", validate="many_to_one")
    condition["abs_error"] = np.abs(condition["median_mtf50"] - condition["reference_mtf50"])
    raw_errors = condition[condition["run_condition"] == "raw"][["condition_cell", "abs_error"]].rename(columns={"abs_error": "raw_abs_error"})
    condition = condition.merge(raw_errors, on="condition_cell", how="left", validate="many_to_one")
    tolerance = float(protocol["reference_rule"]["overshoot_tolerance_cy_px"])
    condition["overshoot"] = condition["median_mtf50"] > condition["reference_mtf50"] + tolerance
    model_summaries = [
        run_summary(condition, protocol, run["condition"], str(run["seed"]))
        for run in protocol["runs"]
        if run["condition"] != "raw"
    ]
    arm_rows = []
    for arm in ("off", "on"):
        subset = condition[(condition["run_condition"] == arm) & (condition["primary_condition"] == 1)]
        arm_condition = subset.groupby("condition_cell", as_index=False).agg(
            abs_error=("abs_error", "mean"),
            raw_abs_error=("raw_abs_error", "first"),
            median_mtf50=("median_mtf50", "mean"),
            reference_mtf50=("reference_mtf50", "first"),
        )
        delta = (arm_condition["abs_error"] - arm_condition["raw_abs_error"]).to_numpy(float)
        bootstrap = protocol["primary_endpoint"]["bootstrap"]
        arm_rows.append(
            {
                "arm": arm,
                "condition_cells": int(len(arm_condition)),
                "mean_abs_error_cy_px": float(arm_condition["abs_error"].mean()),
                "mean_restored_minus_raw_abs_error_cy_px": float(np.mean(delta)),
                "mean_delta_95pct_cluster_bootstrap_ci": bootstrap_mean_ci(
                    delta,
                    int(bootstrap["resamples"]),
                    int(bootstrap["seed"]) + (100 if arm == "off" else 200),
                ),
                "conditions_improved": int(np.sum(delta < 0)),
                "conditions_worsened": int(np.sum(delta > 0)),
            }
        )
    off = condition[(condition["run_condition"] == "off") & (condition["primary_condition"] == 1)][["condition_cell", "seed", "abs_error"]].rename(columns={"abs_error": "off_abs_error"})
    on = condition[(condition["run_condition"] == "on") & (condition["primary_condition"] == 1)][["condition_cell", "seed", "abs_error"]].rename(columns={"abs_error": "on_abs_error"})
    task_pairs = on.merge(off, on=["condition_cell", "seed"], validate="one_to_one")
    task_pairs["on_minus_off_abs_error"] = task_pairs["on_abs_error"] - task_pairs["off_abs_error"]
    task_effect = {
        "pairs": int(len(task_pairs)),
        "condition_cells": int(task_pairs["condition_cell"].nunique()),
        "seeds": int(task_pairs["seed"].nunique()),
        "mean_on_minus_off_abs_error_cy_px": float(task_pairs["on_minus_off_abs_error"].mean()),
        "median_on_minus_off_abs_error_cy_px": float(task_pairs["on_minus_off_abs_error"].median()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    roi.to_csv(output_root / "d0_cam1_roi_results.csv", index=False)
    frame.to_csv(output_root / "d0_cam1_frame_results.csv", index=False)
    condition.to_csv(output_root / "d0_cam1_condition_results.csv", index=False)
    task_pairs.to_csv(output_root / "d0_cam1_task_on_off_pairs.csv", index=False)
    manifest_payload = {"schema_version": SCHEMA, "created_utc": utc_now(), "records": manifest}
    write_json_atomic(output_root / "d0_cam1_progress_manifest.json", manifest_payload, overwrite=False)
    final = {
        "schema_version": SCHEMA,
        "state": "COMPLETE" if validation_pass else "COMPLETE_ESTIMATOR_GATE_FAILED",
        "created_utc": utc_now(),
        "timing_status": protocol["analysis_timing"],
        "counts": {
            "frames": int(freeze["frames"]),
            "condition_cells": 50,
            "roi_records": int(len(cohort)),
            "roi_per_frame_distribution": freeze["roi_count_distribution"],
            "runs_including_raw": len(protocol["runs"]),
            "primary_condition_cells": int(protocol["primary_endpoint"]["expected_condition_cells"]),
        },
        "raw_estimator_validation": {
            "valid_roi_fraction": raw_valid,
            "roi_spearman_vs_imatest": roi_spearman,
            "frame_spearman_vs_preexisting_imatest_mean": frame_spearman,
            "thresholds": gates,
            "pass": validation_pass,
        },
        "primary_run_summaries": model_summaries,
        "arm_three_seed_summaries": arm_rows,
        "task_on_minus_off": task_effect,
        "interpretation_gate": (
            "Paired actual-degradation MTF surrogate may be interpreted within the frozen D0 acquisition envelope."
            if validation_pass
            else "Do not interpret restored MTF results; the in-house estimator failed its predeclared raw Imatest concordance gate."
        ),
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "freeze_sha256": sha256_file(output_root / "d0_cam1_freeze.json"),
            "cohort_sha256": freeze["cohort_sha256"],
            "progress_manifest_sha256": sha256_file(output_root / "d0_cam1_progress_manifest.json"),
        },
    }
    write_json_atomic(output_root / "d0_cam1_final.json", final, overwrite=False)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))


def self_test(protocol_path: Path, output_root: Path) -> None:
    protocol, _, cohort = load_frozen_cohort(protocol_path, output_root)
    device = require_cuda()
    models, tiled_restore = load_models(protocol, device)
    _, frame = frame_records(cohort)[0]
    image = verify_row_assets(frame)
    events = evaluate_frame(image, frame, protocol, models, tiled_restore, device)
    expected_per_event = len(frame)
    valid = sum(result["mtf50"] is not None for event in events for result in event["rois"])
    payload = {
        "schema_version": SCHEMA,
        "pass": len(events) == 7 and all(len(event["rois"]) == expected_per_event for event in events) and valid >= max(1, 6 * expected_per_event),
        "frame_id": str(frame.iloc[0]["frame_id"]),
        "events": len(events),
        "valid_mtf_measurements": valid,
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not payload["pass"]:
        raise ProtocolError("GPU self-test failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "self-test", "run", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        build_cohort(args.protocol, args.output_root)
    elif args.command == "self-test":
        self_test(args.protocol, args.output_root)
    elif args.command == "run":
        run_evaluation(args.protocol, args.output_root)
    else:
        aggregate(args.protocol, args.output_root)


if __name__ == "__main__":
    main()
