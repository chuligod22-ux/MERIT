"""Measure RAW apparent widths on the frozen printed-width chart ROIs.

Scientific scope
----------------
* The calibration is the acquisition-system design GSD (0.3 mm/sensor pixel),
  not a traceable metrology calibration of the printed target.
* The output is printed nominal *apparent* width, not natural-crack physical
  width accuracy.
* Nominal 0.1 and 0.2 mm lines are predeclared non-estimable because they span
  0.33 and 0.67 sensor pixel at the design GSD.
* The centreline is derived once from RAW pixels.  The frozen trace can later
  be reused unchanged for co-registered degraded/restored variants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
MERIT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_DIR = MERIT_ROOT / "40_수정본" / "c009_integrated_experiment" / "protocol"
DEFAULT_LEDGER = PROTOCOL_DIR / "width_chart_roi_final_v1.csv"
DEFAULT_SOURCE_ROOT = Path(
    r"<LOCAL_PATH_REDACTED>"
)
DEFAULT_ARTIFACT_DIR = (
    MERIT_ROOT
    / "40_수정본"
    / "c009_integrated_experiment"
    / "artifacts"
    / "width_raw_measurement_v1"
)

DESIGN_GSD_MM_PER_SENSOR_PIXEL = 0.3
GSD_RELATIVE_SENSITIVITY = 0.05
PREDECLARED_SUBPIXEL_WIDTHS_MM = {0.1, 0.2}

INNER_X_FRACTION = (0.12, 0.88)
INNER_Y_FRACTION = (0.15, 0.85)
BACKGROUND_SIGMA_SIDE_DIVISOR = 22.0
RIDGE_SIGMA_PX = 0.8
DP_MAX_STEP_PX = 4
DP_SMOOTHNESS_PENALTY = 0.22
PROFILE_HALF_RANGE_PX = 16
PROFILE_BASELINE_INNER_PX = 10
PROFILE_MIN_AMPLITUDE_DN = 1.5
PROFILE_MIN_SNR = 3.0
MIN_VALID_PROFILE_COUNT = 50
MIN_VALID_PROFILE_FRACTION = 0.50
OUTLIER_MAD_MULTIPLIER = 3.5
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_BLOCK_WIDTH_PX = 16
TRACE_SAMPLE_STRIDE_PX = 4


@dataclass
class RectifiedPatch:
    gray: np.ndarray
    original_to_rectified: np.ndarray
    rectified_to_original: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def polygon(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"p{i}_x"]), float(row[f"p{i}_y"])] for i in range(4)],
        dtype=np.float32,
    )


def rectify(image: np.ndarray, points: np.ndarray) -> RectifiedPatch:
    side_lengths = np.asarray(
        [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[3] - points[2]),
            np.linalg.norm(points[0] - points[3]),
        ],
        dtype=np.float64,
    )
    side = max(128, int(round(float(np.mean(side_lengths)))))
    target = np.asarray(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    forward = cv2.getPerspectiveTransform(points, target)
    inverse = np.linalg.inv(forward)
    rectified = cv2.warpPerspective(
        image,
        forward,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return RectifiedPatch(gray=gray, original_to_rectified=forward, rectified_to_original=inverse)


def local_dark_residual(gray: np.ndarray) -> np.ndarray:
    sigma = max(7.0, min(gray.shape) / BACKGROUND_SIGMA_SIDE_DIVISOR)
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    return background - gray


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, 0.25)


def trace_centerline(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    height, width = residual.shape
    x0 = int(round(INNER_X_FRACTION[0] * width))
    x1 = int(round(INNER_X_FRACTION[1] * width))
    y0 = int(round(INNER_Y_FRACTION[0] * height))
    y1 = int(round(INNER_Y_FRACTION[1] * height))
    xs = np.arange(x0, x1, dtype=np.int32)

    ridge = cv2.GaussianBlur(
        residual,
        (0, 0),
        sigmaX=RIDGE_SIGMA_PX,
        sigmaY=RIDGE_SIGMA_PX,
        borderType=cv2.BORDER_REFLECT101,
    )
    roi = ridge[y0:y1, x0:x1].astype(np.float64)
    location = float(np.median(roi))
    scale = robust_scale(roi)
    score = np.clip((roi - location) / scale, -4.0, 12.0)

    n_y, n_x = score.shape
    previous = score[:, 0].copy()
    back_steps = np.zeros((n_x, n_y), dtype=np.int8)
    offsets = np.arange(-DP_MAX_STEP_PX, DP_MAX_STEP_PX + 1, dtype=np.int16)

    for x_index in range(1, n_x):
        candidates = np.full((len(offsets), n_y), -np.inf, dtype=np.float64)
        for offset_index, offset in enumerate(offsets):
            penalty = DP_SMOOTHNESS_PENALTY * float(offset * offset)
            if offset < 0:
                candidates[offset_index, :offset] = previous[-offset:] - penalty
            elif offset > 0:
                candidates[offset_index, offset:] = previous[:-offset] - penalty
            else:
                candidates[offset_index, :] = previous
        best_index = np.argmax(candidates, axis=0)
        previous = score[:, x_index] + candidates[best_index, np.arange(n_y)]
        back_steps[x_index, :] = offsets[best_index]

    path = np.empty(n_x, dtype=np.float64)
    current = int(np.argmax(previous))
    path[-1] = current
    for x_index in range(n_x - 1, 0, -1):
        current -= int(back_steps[x_index, current])
        current = min(max(current, 0), n_y - 1)
        path[x_index - 1] = current
    path += y0

    smoothing_window = min(31, n_x if n_x % 2 == 1 else n_x - 1)
    if smoothing_window >= 5:
        path = cv2.GaussianBlur(
            path.reshape(1, -1).astype(np.float32),
            (0, 0),
            sigmaX=3.0,
            borderType=cv2.BORDER_REFLECT101,
        ).reshape(-1).astype(np.float64)

    sampled_y = np.clip(np.rint(path).astype(np.int32), 0, height - 1)
    ridge_values = ridge[sampled_y, xs]
    trace_metrics = {
        "trace_median_residual_dn": float(np.median(ridge_values)),
        "trace_residual_robust_scale_dn": float(scale),
        "trace_median_snr": float(np.median((ridge_values - location) / scale)),
        "trace_max_abs_step_px": float(np.max(np.abs(np.diff(path)))) if len(path) > 1 else 0.0,
    }
    return xs, path, trace_metrics


def half_max_crossing_width(profile: np.ndarray, center: float) -> tuple[float, float, float] | None:
    profile = cv2.GaussianBlur(
        profile.reshape(-1, 1).astype(np.float32),
        (0, 0),
        sigmaX=0.0,
        sigmaY=0.60,
        borderType=cv2.BORDER_REFLECT101,
    ).reshape(-1).astype(np.float64)
    center_index = int(round(center))
    lo = max(0, center_index - PROFILE_HALF_RANGE_PX)
    hi = min(len(profile), center_index + PROFILE_HALF_RANGE_PX + 1)
    if hi - lo < 2 * PROFILE_HALF_RANGE_PX:
        return None

    local = profile[lo:hi]
    expected = center_index - lo
    peak_lo = max(0, expected - 3)
    peak_hi = min(len(local), expected + 4)
    peak_index = peak_lo + int(np.argmax(local[peak_lo:peak_hi]))

    left_baseline = local[: max(1, expected - PROFILE_BASELINE_INNER_PX)]
    right_baseline = local[min(len(local), expected + PROFILE_BASELINE_INNER_PX + 1) :]
    if len(left_baseline) == 0 or len(right_baseline) == 0:
        return None
    baseline_values = np.concatenate([left_baseline, right_baseline])
    baseline = float(np.median(baseline_values))
    noise = robust_scale(baseline_values)
    peak = float(local[peak_index])
    amplitude = peak - baseline
    if amplitude < PROFILE_MIN_AMPLITUDE_DN or amplitude / noise < PROFILE_MIN_SNR:
        return None
    half = baseline + 0.5 * amplitude

    left_index = peak_index
    while left_index > 0 and local[left_index] > half:
        left_index -= 1
    right_index = peak_index
    while right_index < len(local) - 1 and local[right_index] > half:
        right_index += 1
    if left_index == 0 or right_index == len(local) - 1:
        return None

    def interpolate(index_a: int, index_b: int) -> float:
        value_a = float(local[index_a])
        value_b = float(local[index_b])
        if abs(value_b - value_a) < 1e-9:
            return float(index_a + index_b) / 2.0
        return index_a + (half - value_a) / (value_b - value_a)

    left = interpolate(left_index, left_index + 1)
    right = interpolate(right_index - 1, right_index)
    width = right - left
    if not np.isfinite(width) or width <= 0.0:
        return None
    return float(width), float(amplitude), float(amplitude / noise)


def perspective_map(matrix: np.ndarray, x: float, y: float) -> np.ndarray:
    point = np.asarray([x, y, 1.0], dtype=np.float64)
    mapped = matrix @ point
    return mapped[:2] / mapped[2]


def rectified_normal_to_sensor_scale(
    inverse_homography: np.ndarray,
    x: float,
    y: float,
    slope: float,
) -> float:
    normal = np.asarray([-slope, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    p0 = perspective_map(inverse_homography, x, y)
    p1 = perspective_map(inverse_homography, x + normal[0], y + normal[1])
    return float(np.linalg.norm(p1 - p0))


def bootstrap_ci(block_values: np.ndarray, seed: int) -> tuple[float, float]:
    if len(block_values) < 2:
        value = float(np.median(block_values))
        return value, value
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.choice(block_values, size=len(block_values), replace=True)
        draws[index] = np.median(sample)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def deterministic_seed(source_id: str, nominal_width_mm: float) -> int:
    token = f"{source_id}:{nominal_width_mm:.1f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def measure_patch(
    patch: RectifiedPatch,
    source_id: str,
    nominal_width_mm: float,
) -> tuple[dict[str, object], dict[str, object]]:
    residual = local_dark_residual(patch.gray)
    xs, path, trace_metrics = trace_centerline(residual)
    slopes = np.gradient(path, xs.astype(np.float64))

    widths_sensor_px: list[float] = []
    amplitudes: list[float] = []
    snrs: list[float] = []
    valid_xs: list[int] = []
    candidate_count = 0
    for index in range(0, len(xs), 2):
        x = int(xs[index])
        y = float(path[index])
        candidate_count += 1
        result = half_max_crossing_width(residual[:, x], y)
        if result is None:
            continue
        vertical_width, amplitude, snr = result
        slope = float(slopes[index])
        normal_width_rectified = vertical_width / np.sqrt(1.0 + slope * slope)
        sensor_scale = rectified_normal_to_sensor_scale(
            patch.rectified_to_original, float(x), y, slope
        )
        widths_sensor_px.append(normal_width_rectified * sensor_scale)
        amplitudes.append(amplitude)
        snrs.append(snr)
        valid_xs.append(x)

    valid_count = len(widths_sensor_px)
    valid_fraction = valid_count / max(candidate_count, 1)
    width_array = np.asarray(widths_sensor_px, dtype=np.float64)
    x_array = np.asarray(valid_xs, dtype=np.int32)
    retained = np.zeros(valid_count, dtype=bool)
    if valid_count:
        centre = float(np.median(width_array))
        mad_scale = robust_scale(width_array)
        retained = np.abs(width_array - centre) <= OUTLIER_MAD_MULTIPLIER * mad_scale
        width_array = width_array[retained]
        x_array = x_array[retained]
    retained_count = len(width_array)

    blocks: list[float] = []
    if retained_count:
        block_ids = x_array // BOOTSTRAP_BLOCK_WIDTH_PX
        for block_id in np.unique(block_ids):
            blocks.append(float(np.median(width_array[block_ids == block_id])))
    block_values = np.asarray(blocks, dtype=np.float64)

    predeclared_subpixel = nominal_width_mm in PREDECLARED_SUBPIXEL_WIDTHS_MM
    qc_pass = (
        valid_count >= MIN_VALID_PROFILE_COUNT
        and valid_fraction >= MIN_VALID_PROFILE_FRACTION
        and retained_count >= MIN_VALID_PROFILE_COUNT
        and len(block_values) >= 2
    )

    output: dict[str, object] = {
        "source_id": source_id,
        "nominal_width_mm": nominal_width_mm,
        "design_nominal_span_sensor_px": nominal_width_mm / DESIGN_GSD_MM_PER_SENSOR_PIXEL,
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
        ci_low_px, ci_high_px = bootstrap_ci(
            block_values, deterministic_seed(source_id, nominal_width_mm)
        )
        point_mm = point_px * DESIGN_GSD_MM_PER_SENSOR_PIXEL
        ci_low_mm = ci_low_px * DESIGN_GSD_MM_PER_SENSOR_PIXEL
        ci_high_mm = ci_high_px * DESIGN_GSD_MM_PER_SENSOR_PIXEL
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
                    "gsd_sensitivity_low_mm": point_mm * (1.0 - GSD_RELATIVE_SENSITIVITY),
                    "gsd_sensitivity_high_mm": point_mm * (1.0 + GSD_RELATIVE_SENSITIVITY),
                    "absolute_error_mm": abs(point_mm - nominal_width_mm),
                    "relative_error": abs(point_mm - nominal_width_mm) / nominal_width_mm,
                }
            )

    trace = {
        "source_id": source_id,
        "nominal_width_mm": nominal_width_mm,
        "rectified_shape": list(patch.gray.shape),
        "original_to_rectified": patch.original_to_rectified.tolist(),
        "rectified_to_original": patch.rectified_to_original.tolist(),
        "trace_stride_px": TRACE_SAMPLE_STRIDE_PX,
        "trace_x_rectified": xs[::TRACE_SAMPLE_STRIDE_PX].tolist(),
        "trace_y_rectified": np.round(path[::TRACE_SAMPLE_STRIDE_PX], 4).tolist(),
        "trace_source": "RAW_ONLY",
    }
    return output, trace


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
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
            image_cache[filename] = read_image(args.source_root / filename)
        patch = rectify(image_cache[filename], polygon(row))
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

    result_path = args.artifact_dir / "width_chart_raw_measurements_v1.csv"
    trace_path = args.artifact_dir / "width_chart_raw_traces_v1.jsonl"
    summary_path = args.artifact_dir / "width_chart_raw_measurements_v1.json"
    write_csv(result_path, results)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")

    statuses: dict[str, int] = {}
    for row in results:
        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1
    per_width: dict[str, dict[str, int]] = {}
    for row in results:
        width = f"{float(row['nominal_width_mm']):.1f}"
        status = str(row["status"])
        per_width.setdefault(width, {})[status] = per_width.setdefault(width, {}).get(status, 0) + 1

    summary = {
        "schema_version": "width-raw-measurement-v1",
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "scope": "RAW-only printed nominal apparent-width measurement",
        "claim_limit": "Not traceable physical metrology and not natural-crack width accuracy",
        "calibration": {
            "type": "acquisition-system design GSD",
            "mm_per_sensor_pixel": DESIGN_GSD_MM_PER_SENSOR_PIXEL,
            "relative_sensitivity": GSD_RELATIVE_SENSITIVITY,
            "nominal_widths_are_not_used_to_calibrate_scale": True,
        },
        "predeclared_subpixel_non_estimable_mm": sorted(PREDECLARED_SUBPIXEL_WIDTHS_MM),
        "input": {
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": sha256(args.ledger),
            "source_root": str(args.source_root.resolve()),
            "rows": len(ledger),
        },
        "algorithm": {
            "trace_source": "RAW_ONLY",
            "apparent_width_definition": "FWHM between linearly interpolated half-maximum edge crossings",
            "same_trace_required_for_future_variants": True,
            "parameters": {
                "inner_x_fraction": INNER_X_FRACTION,
                "inner_y_fraction": INNER_Y_FRACTION,
                "background_sigma_side_divisor": BACKGROUND_SIGMA_SIDE_DIVISOR,
                "ridge_sigma_px": RIDGE_SIGMA_PX,
                "dp_max_step_px": DP_MAX_STEP_PX,
                "dp_smoothness_penalty": DP_SMOOTHNESS_PENALTY,
                "profile_half_range_px": PROFILE_HALF_RANGE_PX,
                "profile_min_amplitude_dn": PROFILE_MIN_AMPLITUDE_DN,
                "profile_min_snr": PROFILE_MIN_SNR,
                "min_valid_profile_count": MIN_VALID_PROFILE_COUNT,
                "min_valid_profile_fraction": MIN_VALID_PROFILE_FRACTION,
                "outlier_mad_multiplier": OUTLIER_MAD_MULTIPLIER,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_block_width_px": BOOTSTRAP_BLOCK_WIDTH_PX,
            },
        },
        "counts": {"rows": len(results), "status": statuses, "per_width": per_width},
        "outputs": {
            "measurements_csv": str(result_path.resolve()),
            "measurements_csv_sha256": sha256(result_path),
            "traces_jsonl": str(trace_path.resolve()),
            "traces_jsonl_sha256": sha256(trace_path),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
