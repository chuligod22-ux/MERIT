"""Evaluate printed nominal apparent-width preservation for one WP1 model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
REVISION = C009.parent.parent
C006 = REVISION / "40_수정본" / "c006_execution"
DEFAULT_OUTPUT = C009 / "evaluation" / "wp8_printed_width"
SOURCE_ROOT = Path(
    r"<LOCAL_PATH_REDACTED>"
)
SIGMA_H = 2.6
SIGMA_V = 2.4
NOISE_STD_8BIT = 2.5
BASE_SEED = 20260924


def load_measurement_modules():
    root = str(HERE.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    v1 = common.import_module("c009_width_v1", HERE / "measure_width_chart_raw.py")
    sys.modules["measure_width_chart_raw"] = v1
    v2 = common.import_module("c009_width_v2", HERE / "measure_width_chart_raw_v2.py")
    return v1, v2


def load_operations():
    root = str(C006.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return common.import_module("c009_width_ops", C006 / "c006_operational_inference.py")


def load_inputs(contract: dict[str, Any]) -> tuple[list[dict[str, str]], dict[tuple[str, float], dict[str, Any]]]:
    ledger_path = common.verify_asset(contract["assets"]["width_roi"], "width ROI")
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    if len(ledger) != 500:
        raise common.EvaluationError(f"Expected 500 width ROI rows, found {len(ledger)}")
    trace_path = (
        C009
        / "artifacts"
        / "width_raw_measurement_v2"
        / "width_chart_raw_traces_v2.jsonl"
    )
    traces: dict[tuple[str, float], dict[str, Any]] = {}
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["source_id"], float(record["nominal_width_mm"]))
            if key in traces:
                raise common.EvaluationError(f"Duplicate width trace: {key}")
            traces[key] = record
    if len(traces) != 500:
        raise common.EvaluationError(f"Expected 500 width traces, found {len(traces)}")
    return ledger, traces


def deterministic_rng(source_id: str) -> np.random.RandomState:
    digest = hashlib.sha256(f"{BASE_SEED}:{source_id}".encode("utf-8")).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "big"))


def degrade(gray: np.ndarray, source_id: str) -> np.ndarray:
    image = gray.astype(np.float32) / np.float32(255.0)
    output = gaussian_filter(image, sigma=(SIGMA_V, SIGMA_H), mode="reflect").astype(np.float32)
    rng = deterministic_rng(source_id)
    output += rng.randn(*output.shape).astype(np.float32) * np.float32(NOISE_STD_8BIT / 255.0)
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)


@torch.inference_mode()
def tiled_restore(
    gray: np.ndarray,
    model: torch.nn.Module,
    specification: dict[str, Any],
    device: torch.device,
    operations: Any,
    batch_size: int,
) -> np.ndarray:
    tile_size = 448
    overlap = 112
    positions = [
        (y, x)
        for y in operations.axis_starts(gray.shape[0], tile_size, overlap)
        for x in operations.axis_starts(gray.shape[1], tile_size, overlap)
    ]
    image = gray.astype(np.float32) / np.float32(255.0)
    accumulator = np.zeros_like(image, dtype=np.float32)
    weights = np.zeros_like(image, dtype=np.float32)
    window = operations.positive_hann_window(tile_size, 0.05)
    p_b = (SIGMA_H + SIGMA_V) / 6.0 if specification["arm"] == "merit_full" else 0.0
    p_n = NOISE_STD_8BIT / 3.0 if specification["arm"] == "merit_full" else 0.0
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        degraded = np.stack(
            [image[y : y + tile_size, x : x + tile_size] for y, x in chunk]
        ).astype(np.float32)
        inputs = common.prompt_tensor(degraded, p_b, p_n)
        with common.autocast(device):
            restored = model(torch.from_numpy(inputs).to(device))[:, 0].float().cpu().numpy()
        if not np.all(np.isfinite(restored)):
            raise common.EvaluationError("Non-finite width restoration output")
        for tile, (y, x) in zip(restored, chunk):
            accumulator[y : y + tile_size, x : x + tile_size] += tile * window
            weights[y : y + tile_size, x : x + tile_size] += window
    if not np.all(weights > 0):
        raise common.EvaluationError("Width tiling left uncovered pixels")
    return np.rint(np.clip(accumulator / weights, 0.0, 1.0) * 255.0).astype(np.uint8)


def expanded_trace(trace: dict[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    orientation = trace["trace_orientation"]
    if orientation == "HORIZONTAL":
        sampled_axis = np.asarray(trace["trace_x_rectified"], dtype=np.float64)
        sampled_path = np.asarray(trace["trace_y_rectified"], dtype=np.float64)
    elif orientation == "VERTICAL":
        sampled_axis = np.asarray(trace["trace_y_rectified"], dtype=np.float64)
        sampled_path = np.asarray(trace["trace_x_rectified"], dtype=np.float64)
    else:
        raise common.EvaluationError(f"Unknown trace orientation: {orientation}")
    axis = np.arange(int(math.ceil(sampled_axis[0])), int(math.floor(sampled_axis[-1])) + 1, dtype=np.int32)
    path = np.interp(axis.astype(np.float64), sampled_axis, sampled_path)
    return orientation, axis, path


def measure_variant(
    patch: Any,
    trace: dict[str, Any],
    source_id: str,
    nominal_width_mm: float,
    v1: Any,
    v2: Any,
) -> dict[str, Any]:
    residual = v1.local_dark_residual(patch.gray)
    orientation, axis, path = expanded_trace(trace)
    slopes = np.gradient(path, axis.astype(np.float64))
    widths: list[float] = []
    amplitudes: list[float] = []
    snrs: list[float] = []
    valid_axis: list[int] = []
    candidate_count = 0
    for index in range(0, len(axis), 2):
        position = int(axis[index])
        center = float(path[index])
        slope = float(slopes[index])
        candidate_count += 1
        profile = residual[:, position] if orientation == "HORIZONTAL" else residual[position, :]
        result = v1.half_max_crossing_width(profile, center)
        if result is None:
            continue
        profile_width, amplitude, snr = result
        rectified_width = profile_width / math.sqrt(1.0 + slope * slope)
        if orientation == "HORIZONTAL":
            x, y, nx, ny = float(position), center, -slope, 1.0
        else:
            x, y, nx, ny = center, float(position), 1.0, -slope
        scale = v2.rectified_vector_to_sensor_scale(
            patch.rectified_to_original, x, y, nx, ny
        )
        widths.append(rectified_width * scale)
        amplitudes.append(float(amplitude))
        snrs.append(float(snr))
        valid_axis.append(position)
    width_array = np.asarray(widths, dtype=np.float64)
    axis_array = np.asarray(valid_axis, dtype=np.int32)
    valid_count = len(width_array)
    valid_fraction = valid_count / max(candidate_count, 1)
    if valid_count:
        centre = float(np.median(width_array))
        scale = v1.robust_scale(width_array)
        retained = np.abs(width_array - centre) <= v1.OUTLIER_MAD_MULTIPLIER * scale
        width_array = width_array[retained]
        axis_array = axis_array[retained]
    blocks = [
        float(np.median(width_array[axis_array // v1.BOOTSTRAP_BLOCK_WIDTH_PX == block]))
        for block in np.unique(axis_array // v1.BOOTSTRAP_BLOCK_WIDTH_PX)
    ] if len(width_array) else []
    result: dict[str, Any] = {
        "source_id": source_id,
        "nominal_width_mm": nominal_width_mm,
        "trace_orientation": orientation,
        "trace_source": "FROZEN_RAW_V2",
        "candidate_profiles": candidate_count,
        "valid_profiles_before_outlier": valid_count,
        "retained_profiles": len(width_array),
        "valid_profile_fraction": valid_fraction,
        "median_amplitude_dn": float(np.median(amplitudes)) if amplitudes else None,
        "median_profile_snr": float(np.median(snrs)) if snrs else None,
    }
    if nominal_width_mm in v1.PREDECLARED_SUBPIXEL_WIDTHS_MM:
        result["status"] = "PREDECLARED_SUBPIXEL_NON_ESTIMABLE"
        return result
    qc_pass = (
        valid_count >= v1.MIN_VALID_PROFILE_COUNT
        and valid_fraction >= v1.MIN_VALID_PROFILE_FRACTION
        and len(width_array) >= v1.MIN_VALID_PROFILE_COUNT
        and len(blocks) >= 2
    )
    if not qc_pass:
        result["status"] = "QC_NON_ESTIMABLE"
        return result
    point_px = float(np.median(width_array))
    low_px, high_px = v1.bootstrap_ci(
        np.asarray(blocks, dtype=np.float64), v1.deterministic_seed(source_id, nominal_width_mm)
    )
    point_mm = point_px * v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL
    low_mm = low_px * v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL
    high_mm = high_px * v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL
    if high_mm - low_mm > 0.5 * nominal_width_mm:
        result["status"] = "UNCERTAINTY_NON_ESTIMABLE"
        return result
    result.update(
        {
            "status": "ESTIMABLE",
            "apparent_width_sensor_px": point_px,
            "apparent_width_mm_design_gsd": point_mm,
            "stat_ci95_low_mm": low_mm,
            "stat_ci95_high_mm": high_mm,
            "absolute_error_mm": abs(point_mm - nominal_width_mm),
            "relative_error": abs(point_mm - nominal_width_mm) / nominal_width_mm,
        }
    )
    return result


def run(specification: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if specification["kind"] != "wp1":
        raise common.EvaluationError("Width matrix is restricted to 18 WP1 models")
    contract = common.load_contract()
    ledger, traces = load_inputs(contract)
    v1, v2 = load_measurement_modules()
    operations = load_operations()
    model_name = common.model_id(specification)
    output_dir = output_root / model_name
    marker_path = output_dir / "RUN_COMPLETE.json"
    if marker_path.exists():
        marker = common.load_json(marker_path)
        marker["status"] = "ALREADY_COMPLETE"
        return marker
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise common.EvaluationError("CUDA is required")
    model, provenance = common.build_and_load_model(specification, device)
    batch_size = {"nafnet": 32, "restormer": 16, "mprnet": 16}[specification["family"]]
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in ledger:
        by_source.setdefault(row["source_id"], []).append(row)
    if len(by_source) != 50 or any(len(rows) != 10 for rows in by_source.values()):
        raise common.EvaluationError("Width source/ROI coverage mismatch")
    started = time.perf_counter()
    completed = 0
    for source_id in sorted(by_source):
        rows = by_source[source_id]
        path = output_dir / "progress" / f"{source_id}.json"
        if path.exists():
            payload = common.load_json(path)
            if payload.get("status") != "PASS" or payload.get("model_id") != model_name:
                raise common.EvaluationError(f"Invalid width progress: {path}")
            completed += 1
            continue
        source = SOURCE_ROOT / rows[0]["filename"]
        if common.sha256_file(source) != rows[0]["source_sha256"]:
            raise common.EvaluationError(f"Width source hash mismatch: {source}")
        image = v1.read_image(source)
        raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        degraded_gray = degrade(raw_gray, source_id)
        torch.cuda.reset_peak_memory_stats(device)
        restored_gray = tiled_restore(
            degraded_gray, model, specification, device, operations, batch_size
        )
        degraded_bgr = cv2.cvtColor(degraded_gray, cv2.COLOR_GRAY2BGR)
        restored_bgr = cv2.cvtColor(restored_gray, cv2.COLOR_GRAY2BGR)
        measurements: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda value: float(value["nominal_width_mm"])):
            nominal = float(row["nominal_width_mm"])
            key = (source_id, nominal)
            polygon = v1.polygon(row)
            degraded_patch = v1.rectify(degraded_bgr, polygon)
            restored_patch = v1.rectify(restored_bgr, polygon)
            for condition, patch in (
                ("controlled_degraded", degraded_patch),
                ("restored", restored_patch),
            ):
                measurement = measure_variant(patch, traces[key], source_id, nominal, v1, v2)
                measurement.update(
                    {
                        "condition": condition,
                        "filename": row["filename"],
                        "source_sha256": row["source_sha256"],
                        "row_index": int(row["row_index"]),
                        "col_index": int(row["col_index"]),
                    }
                )
                measurements.append(measurement)
        if len(measurements) != 20:
            raise common.EvaluationError("Width per-source measurement count mismatch")
        payload: dict[str, Any] = {
            "schema_version": "MERIT-C009-WIDTH-PROGRESS-v1",
            "status": "PASS",
            "model_id": model_name,
            "source_id": source_id,
            "source_path": str(source.resolve()),
            "source_sha256": rows[0]["source_sha256"],
            "degradation": {
                "sigma_h_px": SIGMA_H,
                "sigma_v_px": SIGMA_V,
                "noise_std_8bit": NOISE_STD_8BIT,
                "deterministic_seed_basis": f"sha256({BASE_SEED}:{source_id})[:4]",
            },
            "measurements": measurements,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        payload["payload_sha256"] = hashlib.sha256(common.canonical_bytes(payload)).hexdigest()
        common.write_json_exclusive(path, payload)
        completed += 1
        elapsed = time.perf_counter() - started
        common.write_json_atomic(
            output_dir / "status.json",
            {
                "schema_version": "MERIT-C009-WIDTH-STATUS-v1",
                "status": "running" if completed < 50 else "measurement_complete",
                "model_id": model_name,
                "completed_sources": completed,
                "total_sources": 50,
                "projected_remaining_seconds": elapsed / completed * (50 - completed),
            },
        )
        print(
            json.dumps({"model_id": model_name, "completed": completed, "total": 50}),
            flush=True,
        )
    progress = sorted((output_dir / "progress").glob("W*.json"))
    if len(progress) != 50:
        raise common.EvaluationError("Width final coverage mismatch")
    digest = hashlib.sha256()
    for path in progress:
        digest.update(path.name.encode("utf-8"))
        digest.update(common.sha256_file(path).encode("ascii"))
    marker: dict[str, Any] = {
        "schema_version": "MERIT-C009-WIDTH-RUN-COMPLETE-v1",
        "status": "PASS",
        "model_id": model_name,
        "specification": specification,
        "provenance": provenance,
        "contract_sha256": common.EXPECTED_CONTRACT_SHA256,
        "source_frames": 50,
        "roi_per_frame": 10,
        "conditions_per_roi": 2,
        "progress_manifest_sha256": digest.hexdigest(),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    marker["payload_sha256"] = hashlib.sha256(common.canonical_bytes(marker)).hexdigest()
    common.write_json_atomic(marker_path, marker)
    return marker


def find_specification(model_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    matches = [spec for spec in common.wp1_specifications(contract) if common.model_id(spec) == model_name]
    if len(matches) != 1:
        raise common.EvaluationError(f"Unknown WP1 model id: {model_name}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = common.load_contract()
    result = run(find_specification(args.model_id, contract), args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
