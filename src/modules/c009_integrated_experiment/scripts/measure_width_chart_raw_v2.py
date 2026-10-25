"""Orientation-aware RAW apparent-width measurement (v2).

Version 1 assumed that every printed line ran left-to-right.  Raw-only trace
contact sheets showed that five acquisition frames contain the same target in
a 90-degree orientation.  Version 2 evaluates horizontal and vertical trace
candidates using RAW residual signal only and freezes the stronger continuous
trace.  No nominal width, restoration output, or downstream endpoint is used
to select orientation.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import measure_width_chart_raw as v1


DEFAULT_ARTIFACT_DIR = (
    v1.MERIT_ROOT
    / "40_수정본"
    / "c009_integrated_experiment"
    / "artifacts"
    / "width_raw_measurement_v2"
)


def select_orientation(
    residual: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, float]]:
    horizontal_axis, horizontal_path, horizontal_metrics = v1.trace_centerline(residual)
    vertical_axis, vertical_path, vertical_metrics = v1.trace_centerline(residual.T)
    horizontal_score = float(horizontal_metrics["trace_median_snr"])
    vertical_score = float(vertical_metrics["trace_median_snr"])
    if horizontal_score >= vertical_score:
        orientation = "HORIZONTAL"
        axis = horizontal_axis
        path = horizontal_path
        selected = horizontal_metrics
    else:
        orientation = "VERTICAL"
        axis = vertical_axis
        path = vertical_path
        selected = vertical_metrics
    metrics = {
        **selected,
        "horizontal_trace_median_snr": horizontal_score,
        "vertical_trace_median_snr": vertical_score,
        "orientation_score_margin": abs(horizontal_score - vertical_score),
    }
    return orientation, axis, path, metrics


def rectified_vector_to_sensor_scale(
    inverse_homography: np.ndarray,
    x: float,
    y: float,
    vector_x: float,
    vector_y: float,
) -> float:
    vector = np.asarray([vector_x, vector_y], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    p0 = v1.perspective_map(inverse_homography, x, y)
    p1 = v1.perspective_map(
        inverse_homography, x + float(vector[0]), y + float(vector[1])
    )
    return float(np.linalg.norm(p1 - p0))


def measure_patch(
    patch: v1.RectifiedPatch,
    source_id: str,
    nominal_width_mm: float,
) -> tuple[dict[str, object], dict[str, object]]:
    residual = v1.local_dark_residual(patch.gray)
    orientation, axis, path, trace_metrics = select_orientation(residual)
    slopes = np.gradient(path, axis.astype(np.float64))

    widths_sensor_px: list[float] = []
    amplitudes: list[float] = []
    snrs: list[float] = []
    valid_axis: list[int] = []
    candidate_count = 0
    for index in range(0, len(axis), 2):
        axis_position = int(axis[index])
        path_position = float(path[index])
        slope = float(slopes[index])
        candidate_count += 1
        if orientation == "HORIZONTAL":
            result = v1.half_max_crossing_width(residual[:, axis_position], path_position)
            x = float(axis_position)
            y = path_position
            normal_x, normal_y = -slope, 1.0
        else:
            result = v1.half_max_crossing_width(residual[axis_position, :], path_position)
            x = path_position
            y = float(axis_position)
            normal_x, normal_y = 1.0, -slope
        if result is None:
            continue
        profile_width, amplitude, snr = result
        normal_width_rectified = profile_width / np.sqrt(1.0 + slope * slope)
        sensor_scale = rectified_vector_to_sensor_scale(
            patch.rectified_to_original, x, y, normal_x, normal_y
        )
        widths_sensor_px.append(normal_width_rectified * sensor_scale)
        amplitudes.append(amplitude)
        snrs.append(snr)
        valid_axis.append(axis_position)

    valid_count = len(widths_sensor_px)
    valid_fraction = valid_count / max(candidate_count, 1)
    width_array = np.asarray(widths_sensor_px, dtype=np.float64)
    axis_array = np.asarray(valid_axis, dtype=np.int32)
    if valid_count:
        centre = float(np.median(width_array))
        mad_scale = v1.robust_scale(width_array)
        retained = (
            np.abs(width_array - centre) <= v1.OUTLIER_MAD_MULTIPLIER * mad_scale
        )
        width_array = width_array[retained]
        axis_array = axis_array[retained]
    retained_count = len(width_array)

    blocks: list[float] = []
    if retained_count:
        block_ids = axis_array // v1.BOOTSTRAP_BLOCK_WIDTH_PX
        for block_id in np.unique(block_ids):
            blocks.append(float(np.median(width_array[block_ids == block_id])))
    block_values = np.asarray(blocks, dtype=np.float64)

    predeclared_subpixel = nominal_width_mm in v1.PREDECLARED_SUBPIXEL_WIDTHS_MM
    qc_pass = (
        valid_count >= v1.MIN_VALID_PROFILE_COUNT
        and valid_fraction >= v1.MIN_VALID_PROFILE_FRACTION
        and retained_count >= v1.MIN_VALID_PROFILE_COUNT
        and len(block_values) >= 2
    )
    output: dict[str, object] = {
        "source_id": source_id,
        "nominal_width_mm": nominal_width_mm,
        "design_nominal_span_sensor_px": nominal_width_mm
        / v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL,
        "trace_orientation": orientation,
        "status": "",
        "apparent_width_sensor_px": "",
        "apparent_width_mm_design_gsd": "",
        "stat_ci95_low_mm": "",
        "stat_ci95_high_mm": "",
        "gsd_sensitivity_low_mm": "",
        "gsd_sensitivity_high_mm": "",
        "absolute_error_mm": "",
        "relative_error": "",
        "candidate_profiles": candidate_count,
        "valid_profiles_before_outlier": valid_count,
        "retained_profiles": retained_count,
        "valid_profile_fraction": valid_fraction,
        "median_amplitude_dn": float(np.median(amplitudes)) if amplitudes else "",
        "median_profile_snr": float(np.median(snrs)) if snrs else "",
        **trace_metrics,
    }
    if predeclared_subpixel:
        output["status"] = "PREDECLARED_SUBPIXEL_NON_ESTIMABLE"
    elif not qc_pass:
        output["status"] = "QC_NON_ESTIMABLE"
    else:
        point_px = float(np.median(width_array))
        ci_low_px, ci_high_px = v1.bootstrap_ci(
            block_values, v1.deterministic_seed(source_id, nominal_width_mm)
        )
        point_mm = point_px * v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL
        ci_low_mm = ci_low_px * v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL
        ci_high_mm = ci_high_px * v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL
        if ci_high_mm - ci_low_mm > 0.5 * nominal_width_mm:
            output["status"] = "UNCERTAINTY_NON_ESTIMABLE"
        else:
            output.update(
                {
                    "status": "ESTIMABLE",
                    "apparent_width_sensor_px": point_px,
                    "apparent_width_mm_design_gsd": point_mm,
                    "stat_ci95_low_mm": ci_low_mm,
                    "stat_ci95_high_mm": ci_high_mm,
                    "gsd_sensitivity_low_mm": point_mm
                    * (1.0 - v1.GSD_RELATIVE_SENSITIVITY),
                    "gsd_sensitivity_high_mm": point_mm
                    * (1.0 + v1.GSD_RELATIVE_SENSITIVITY),
                    "absolute_error_mm": abs(point_mm - nominal_width_mm),
                    "relative_error": abs(point_mm - nominal_width_mm) / nominal_width_mm,
                }
            )

    if orientation == "HORIZONTAL":
        trace_x = axis
        trace_y = path
    else:
        trace_x = path
        trace_y = axis
    trace = {
        "source_id": source_id,
        "nominal_width_mm": nominal_width_mm,
        "trace_orientation": orientation,
        "orientation_selection": "maximum RAW-only continuous-trace median SNR",
        "rectified_shape": list(patch.gray.shape),
        "original_to_rectified": patch.original_to_rectified.tolist(),
        "rectified_to_original": patch.rectified_to_original.tolist(),
        "trace_stride_px": v1.TRACE_SAMPLE_STRIDE_PX,
        "trace_x_rectified": np.round(trace_x[:: v1.TRACE_SAMPLE_STRIDE_PX], 4).tolist(),
        "trace_y_rectified": np.round(trace_y[:: v1.TRACE_SAMPLE_STRIDE_PX], 4).tolist(),
        "trace_source": "RAW_ONLY",
    }
    return output, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=v1.DEFAULT_LEDGER)
    parser.add_argument("--source-root", type=Path, default=v1.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    with args.ledger.open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    if len(ledger) != 500:
        raise RuntimeError(f"Expected 500 frozen ROI rows, found {len(ledger)}")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    image_cache: dict[str, np.ndarray] = {}
    for row_index, row in enumerate(ledger, start=1):
        filename = row["filename"]
        if filename not in image_cache:
            image_cache.clear()
            image_cache[filename] = v1.read_image(args.source_root / filename)
        patch = v1.rectify(image_cache[filename], v1.polygon(row))
        result, trace = measure_patch(
            patch,
            source_id=row["source_id"],
            nominal_width_mm=float(row["nominal_width_mm"]),
        )
        result.update(
            {
                "filename": filename,
                "source_sha256": row["source_sha256"],
                "row_index": int(row["row_index"]),
                "col_index": int(row["col_index"]),
            }
        )
        ordered = {
            "source_id": result.pop("source_id"),
            "filename": result.pop("filename"),
            "source_sha256": result.pop("source_sha256"),
            "row_index": result.pop("row_index"),
            "col_index": result.pop("col_index"),
            **result,
        }
        results.append(ordered)
        traces.append(trace)
        if row_index % 50 == 0:
            print(f"processed {row_index}/500")

    result_path = args.artifact_dir / "width_chart_raw_measurements_v2.csv"
    trace_path = args.artifact_dir / "width_chart_raw_traces_v2.jsonl"
    summary_path = args.artifact_dir / "width_chart_raw_measurements_v2.json"
    v1.write_csv(result_path, results)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")

    statuses: dict[str, int] = {}
    orientations: dict[str, int] = {}
    per_width: dict[str, dict[str, int]] = {}
    for row in results:
        status = str(row["status"])
        orientation = str(row["trace_orientation"])
        statuses[status] = statuses.get(status, 0) + 1
        orientations[orientation] = orientations.get(orientation, 0) + 1
        width = f"{float(row['nominal_width_mm']):.1f}"
        per_width.setdefault(width, {})[status] = per_width.setdefault(width, {}).get(status, 0) + 1

    summary = {
        "schema_version": "width-raw-measurement-v2",
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "supersedes": "v1 measurement for orientation handling; v1 retained as failed diagnostic",
        "change_reason": "Five raw frames contain vertically oriented target lines; v1 assumed horizontal only.",
        "scope": "RAW-only printed nominal apparent-width measurement",
        "claim_limit": "Not traceable physical metrology and not natural-crack width accuracy",
        "calibration": {
            "type": "acquisition-system design GSD",
            "mm_per_sensor_pixel": v1.DESIGN_GSD_MM_PER_SENSOR_PIXEL,
            "relative_sensitivity": v1.GSD_RELATIVE_SENSITIVITY,
            "nominal_widths_are_not_used_to_calibrate_scale": True,
        },
        "orientation_selection": {
            "candidates": ["HORIZONTAL", "VERTICAL"],
            "selector": "maximum RAW-only continuous-trace median SNR",
            "counts": orientations,
            "does_not_use": ["nominal width", "restoration output", "endpoint outcome"],
        },
        "predeclared_subpixel_non_estimable_mm": sorted(v1.PREDECLARED_SUBPIXEL_WIDTHS_MM),
        "input": {
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": v1.sha256(args.ledger),
            "source_root": str(args.source_root.resolve()),
            "rows": len(ledger),
        },
        "counts": {
            "rows": len(results),
            "status": statuses,
            "orientation": orientations,
            "per_width": per_width,
        },
        "outputs": {
            "measurements_csv": str(result_path.resolve()),
            "measurements_csv_sha256": v1.sha256(result_path),
            "traces_jsonl": str(trace_path.resolve()),
            "traces_jsonl_sha256": v1.sha256(trace_path),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": v1.sha256(Path(__file__))},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
