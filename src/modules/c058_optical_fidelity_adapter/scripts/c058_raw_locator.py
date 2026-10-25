"""Deterministic RAW-only locator for the MERIT quadrant chart.

The locator itself consumes pixels only.  Existing Imatest crops, ROI boxes,
and MTF50 values are loaded exclusively by the development-validation CLI and
never enter :func:`locate_chart` or candidate ranking.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


WORKING_SIZE = (1024, 683)
SEARCH_WIDTH = 1024
SEARCH_HEIGHT = 280
TOP_PADDING = 120
RADIUS_GRID = (25, 32, 40, 50, 62, 76, 92, 110)
ANGLE_GRID = (-15, -10, -5, 0, 5, 10, 15)
MIN_ACCEPTED_SCORE = 0.45
CANONICAL_SIZE = 1024
CANONICAL_RADIUS = 320

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
REPO = BASE.parents[2]
DEFAULT_COHORT = (
    REPO
    / "05_MERIT_revision"
    / "40_수정본"
    / "c007_d0_condition_eval"
    / "artifacts"
    / "cam1_v1"
    / "d0_cam1_cohort.csv"
)
DEFAULT_RAW_ROOT = Path(
    r"<LOCAL_PATH_REDACTED>"
)
DEFAULT_OUTPUT = BASE / "artifacts" / "locator_development_v1"
MTF_ESTIMATOR = (
    REPO
    / "05_MERIT_revision"
    / "40_수정본"
    / "c057_mtf50_estimator_v2"
    / "mtf50_estimator.py"
)


class LocatorFailure(RuntimeError):
    """Raised whenever the finite locator cannot satisfy its frozen gates."""


class LocatorResult(NamedTuple):
    center_x: float
    center_y: float
    radius: float
    angle_deg: float
    score: float
    original_width: int
    original_height: int


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise LocatorFailure(f"unreadable image: {path}")
    return image


def _checker_template(radius: int, angle_deg: int) -> np.ndarray:
    size = round(2.2 * radius)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    x = xx - (size - 1) / 2
    y = yy - (size - 1) / 2
    radians = math.radians(angle_deg)
    cosine, sine = math.cos(radians), math.sin(radians)
    xr = cosine * x + sine * y
    yr = -sine * x + cosine * y
    distance = np.hypot(x, y)
    template = np.zeros((size, size), dtype=np.float32)
    inside = distance < 0.88 * radius
    template[inside] = np.where(xr[inside] * yr[inside] >= 0, -1.0, 1.0)
    uncertain_axis = (np.abs(xr) < 0.08 * radius) | (np.abs(yr) < 0.08 * radius)
    template[inside & uncertain_axis] = 0.0
    return template


TEMPLATES = tuple(
    (radius, angle, _checker_template(radius, angle))
    for radius in RADIUS_GRID
    for angle in ANGLE_GRID
)


def locate_chart(
    gray: np.ndarray, *, working_size: tuple[int, int] = WORKING_SIZE
) -> LocatorResult:
    """Locate the chart using only a finite bank of synthetic templates.

    The search window is the fixed upper-left acquisition region.  Edge
    padding allows the physical target to be partially outside the sensor.
    No image-derived template, reference ROI, MTF score, or condition metadata
    is accepted by this function.
    """

    if gray.ndim != 2 or min(gray.shape) < 64:
        raise LocatorFailure("expected a two-dimensional RAW grayscale image")
    work_width, work_height = working_size
    working = cv2.resize(
        gray, (work_width, work_height), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    padded = np.pad(working, ((TOP_PADDING, 0), (0, 0)), mode="edge")
    search = padded[: TOP_PADDING + SEARCH_HEIGHT, :SEARCH_WIDTH]
    best_score = -math.inf
    best: tuple[float, float, int, int] | None = None
    for radius, angle, template in TEMPLATES:
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(np.abs(response))
        if score > best_score:
            best_score = float(score)
            center_x = location[0] + (template.shape[1] - 1) / 2
            center_y = location[1] + (template.shape[0] - 1) / 2 - TOP_PADDING
            best = (center_x, center_y, radius, angle)
    if best is None or not math.isfinite(best_score) or best_score < MIN_ACCEPTED_SCORE:
        raise LocatorFailure(f"checker score {best_score:.6f} below gate")

    center_x, center_y, radius, angle = best
    scale_x = gray.shape[1] / work_width
    scale_y = gray.shape[0] / work_height
    scale = math.sqrt(scale_x * scale_y)
    return LocatorResult(
        center_x=float(center_x * scale_x),
        center_y=float(center_y * scale_y),
        radius=float(radius * scale),
        angle_deg=float(angle),
        score=best_score,
        original_width=int(gray.shape[1]),
        original_height=int(gray.shape[0]),
    )


def _canonical_rois() -> list[dict[str, int | str]]:
    center, radius = CANONICAL_SIZE / 2, float(CANONICAL_RADIUS)
    specifications = (
        (1, "vertical_edge", 0.00, -0.50, 0.36, 0.62),
        (2, "vertical_edge", 0.00, +0.50, 0.36, 0.62),
        (3, "horizontal_edge", -0.53, 0.00, 0.62, 0.36),
        (4, "horizontal_edge", +0.53, 0.00, 0.62, 0.36),
    )
    rois: list[dict[str, int | str]] = []
    for roi_id, orientation, dx, dy, width, height in specifications:
        cx, cy = center + dx * radius, center + dy * radius
        x1, x2 = round(cx - width * radius / 2), round(cx + width * radius / 2)
        y1, y2 = round(cy - height * radius / 2), round(cy + height * radius / 2)
        rois.append(
            {
                "roi_id": roi_id,
                "orientation": orientation,
                "x1": max(0, x1),
                "y1": max(0, y1),
                "x2": min(CANONICAL_SIZE, x2),
                "y2": min(CANONICAL_SIZE, y2),
            }
        )
    return rois


def normalization_matrix(result: LocatorResult) -> np.ndarray:
    scale = CANONICAL_RADIUS / result.radius
    matrix = cv2.getRotationMatrix2D(
        (result.center_x, result.center_y), result.angle_deg, scale
    )
    matrix[0, 2] += CANONICAL_SIZE / 2 - result.center_x
    matrix[1, 2] += CANONICAL_SIZE / 2 - result.center_y
    return matrix


def normalize_chart(
    gray: np.ndarray, result: LocatorResult
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    normalized = cv2.warpAffine(
        gray,
        normalization_matrix(result),
        (CANONICAL_SIZE, CANONICAL_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return normalized, _canonical_rois()


def find_exact_crop_origin(raw: np.ndarray, extract: np.ndarray) -> tuple[int, int, float]:
    """Find an exact pre-existing crop for validation, never for localization."""

    if raw.ndim != 2 or extract.ndim != 2:
        raise ValueError("grayscale images required")
    if extract.shape[0] > raw.shape[0] or extract.shape[1] > raw.shape[1]:
        raise ValueError("extract is larger than raw")
    scale = min(1.0, 1024.0 / raw.shape[1])
    if scale < 1.0:
        small_raw = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_extract = cv2.resize(
            extract, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        small_raw, small_extract = raw, extract
    response = cv2.matchTemplate(small_raw, small_extract, cv2.TM_CCOEFF_NORMED)
    _, _, _, rough = cv2.minMaxLoc(response)
    rough_x, rough_y = round(rough[0] / scale), round(rough[1] / scale)

    step_y = max(1, extract.shape[0] // 96)
    step_x = max(1, extract.shape[1] // 96)
    sample = extract[::step_y, ::step_x].astype(np.int16)
    best = (math.inf, -1, -1)
    for y in range(max(0, rough_y - 8), min(raw.shape[0] - extract.shape[0], rough_y + 8) + 1):
        for x in range(max(0, rough_x - 8), min(raw.shape[1] - extract.shape[1], rough_x + 8) + 1):
            candidate = raw[
                y : y + extract.shape[0] : step_y,
                x : x + extract.shape[1] : step_x,
            ].astype(np.int16)
            mae = float(np.mean(np.abs(candidate - sample)))
            if (mae, x, y) < best:
                best = (mae, x, y)
    return int(best[1]), int(best[2]), float(best[0])


def _load_estimator():
    specification = importlib.util.spec_from_file_location("c057_mtf", MTF_ESTIMATOR)
    if specification is None or specification.loader is None:
        raise RuntimeError("C057 estimator import failed")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _reference_geometry(rows: pd.DataFrame) -> tuple[float, float, float]:
    vertical = rows[rows["orientation"] == "vertical_edge"]
    horizontal = rows[rows["orientation"] == "horizontal_edge"]
    if vertical.empty or horizontal.empty:
        raise LocatorFailure("both Imatest ROI orientations required for validation")
    center_x = float(((vertical["x1"] + vertical["x2"]) / 2).median())
    center_y = float(((horizontal["y1"] + horizontal["y2"]) / 2).median())
    radius_candidates: list[float] = []
    if len(vertical) >= 2:
        centers = ((vertical["y1"] + vertical["y2"]) / 2).sort_values()
        radius_candidates.append(float(centers.iloc[-1] - centers.iloc[0]))
    if len(horizontal) >= 2:
        centers = ((horizontal["x1"] + horizontal["x2"]) / 2).sort_values()
        radius_candidates.append(float(centers.iloc[-1] - centers.iloc[0]))
    if not radius_candidates:
        vertical_scale = float((vertical["y2"] - vertical["y1"]).median() / 0.62)
        horizontal_scale = float((horizontal["x2"] - horizontal["x1"]).median() / 0.62)
        radius_candidates.extend((vertical_scale, horizontal_scale))
    radius = float(np.median(radius_candidates))
    if not math.isfinite(radius) or radius < 100:
        raise LocatorFailure("invalid reference radius")
    return center_x, center_y, radius


def _extract_path(raw_root: Path, condition: str, frame: str) -> Path:
    condition_root = raw_root / condition
    candidates = sorted(
        directory / frame
        for directory in condition_root.iterdir()
        if directory.is_dir() and (directory / frame).is_file()
    )
    if len(candidates) != 1:
        raise LocatorFailure(
            f"expected one pre-existing extract for {condition}/{frame}, found {len(candidates)}"
        )
    return candidates[0]


def _validate_frame(
    rows: pd.DataFrame,
    raw_root: Path,
    estimator,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    first = rows.iloc[0]
    raw_path = raw_root / str(first["condition"]) / str(first["frame"])
    extract_path = _extract_path(raw_root, str(first["condition"]), str(first["frame"]))
    raw = read_gray(raw_path)
    extract = read_gray(extract_path)
    result = locate_chart(raw)
    crop_x, crop_y, crop_mae = find_exact_crop_origin(raw, extract)
    reference_x, reference_y, reference_radius = _reference_geometry(rows)
    detected_x = result.center_x - crop_x
    detected_y = result.center_y - crop_y
    center_error = math.hypot(detected_x - reference_x, detected_y - reference_y)
    radius_error = abs(result.radius - reference_radius)

    normalized, predicted_rois = normalize_chart(raw, result)
    roi_records: list[dict[str, object]] = []
    reference_order = {
        "vertical_edge": rows[rows["orientation"] == "vertical_edge"].sort_values("y1"),
        "horizontal_edge": rows[rows["orientation"] == "horizontal_edge"].sort_values("x1"),
    }
    orientation_index = {"vertical_edge": 0, "horizontal_edge": 0}
    for roi in predicted_rois:
        orientation = str(roi["orientation"])
        reference_group = reference_order[orientation]
        index = orientation_index[orientation]
        orientation_index[orientation] += 1
        if index >= len(reference_group):
            continue
        reference = reference_group.iloc[index]
        patch = normalized[
            int(roi["y1"]) : int(roi["y2"]), int(roi["x1"]) : int(roi["x2"])
        ]
        estimate = estimator.estimate_mtf50_v2(
            patch, orientation, localization_sigma_px=0.75
        )
        scale = CANONICAL_RADIUS / result.radius
        raw_mtf50 = float(estimate.mtf50 * scale) if estimate.valid else math.nan
        roi_records.append(
            {
                "frame_id": first["frame_id"],
                "condition": first["condition"],
                "roi_id": int(roi["roi_id"]),
                "orientation": orientation,
                "valid": bool(estimate.valid),
                "reason": estimate.reason,
                "raw_scale_adjusted_mtf50": raw_mtf50 if math.isfinite(raw_mtf50) else None,
                "imatest_roi_mtf50": float(reference["imatest_roi_mtf50"]),
            }
        )
    valid_count = sum(bool(record["valid"]) for record in roi_records)
    frame = {
        "frame_id": first["frame_id"],
        "condition": first["condition"],
        "speed_kmh": int(first["speed_kmh"]),
        "distance_m": float(first["distance_m"]),
        "iso": int(first["iso"]),
        "raw_width": int(raw.shape[1]),
        "raw_height": int(raw.shape[0]),
        "extract_width": int(extract.shape[1]),
        "extract_height": int(extract.shape[0]),
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_sparse_mae": crop_mae,
        "score": result.score,
        "center_x_raw": result.center_x,
        "center_y_raw": result.center_y,
        "radius_raw": result.radius,
        "angle_deg": result.angle_deg,
        "reference_center_x_extract": reference_x,
        "reference_center_y_extract": reference_y,
        "reference_radius": reference_radius,
        "center_error_px": center_error,
        "center_error_radii": center_error / reference_radius,
        "radius_error_fraction": radius_error / reference_radius,
        "roi_attempts": len(roi_records),
        "roi_valid": valid_count,
    }
    return frame, roi_records


def _summary(frames: pd.DataFrame, rois: pd.DataFrame, total: int) -> dict[str, object]:
    successful = frames[frames["status"] == "ok"]
    finite = rois[rois["valid"] == True] if not rois.empty else rois  # noqa: E712
    if len(finite) >= 3:
        rho = spearmanr(
            finite["raw_scale_adjusted_mtf50"].astype(float),
            finite["imatest_roi_mtf50"].astype(float),
        ).statistic
        mae = float(
            np.mean(
                np.abs(
                    finite["raw_scale_adjusted_mtf50"].astype(float)
                    - finite["imatest_roi_mtf50"].astype(float)
                )
            )
        )
    else:
        rho, mae = math.nan, math.nan
    center = (
        successful["center_error_radii"].astype(float).to_numpy()
        if len(successful)
        else np.asarray([], dtype=float)
    )
    radius = (
        successful["radius_error_fraction"].astype(float).to_numpy()
        if len(successful)
        else np.asarray([], dtype=float)
    )
    exact_crop = (
        successful["crop_sparse_mae"].astype(float).to_numpy()
        if len(successful)
        else np.asarray([], dtype=float)
    )
    return {
        "schema_version": "MERIT-C058-RAW-LOCATOR-DEVELOPMENT-v2",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "479 exposed C057 frames only; same-acquisition development validation",
        "raw_only_locator": True,
        "reference_used_only_after_localization": True,
        "bounded_correction": "search width only: 420 to 1024 working pixels",
        "frames_expected": total,
        "frames_located": int(len(successful)),
        "localization_coverage": float(len(successful) / total),
        "condition_cells_located": int(successful["condition"].nunique()) if len(successful) else 0,
        "extract_shape_counts": (
            successful.groupby(["extract_width", "extract_height"]).size().rename("count").reset_index().to_dict("records")
            if len(successful)
            else []
        ),
        "crop_origin": {
            "x_min": int(successful["crop_x"].min()) if len(successful) else None,
            "x_max": int(successful["crop_x"].max()) if len(successful) else None,
            "y_min": int(successful["crop_y"].min()) if len(successful) else None,
            "y_max": int(successful["crop_y"].max()) if len(successful) else None,
            "maximum_sparse_mae": float(exact_crop.max()) if len(exact_crop) else None,
            "exact_fraction": float(np.mean(exact_crop == 0.0)) if len(exact_crop) else None,
        },
        "center_error_radii": {
            "median": float(np.median(center)) if len(center) else None,
            "p95": float(np.percentile(center, 95)) if len(center) else None,
            "maximum": float(center.max()) if len(center) else None,
        },
        "radius_error_fraction": {
            "median": float(np.median(radius)) if len(radius) else None,
            "p95": float(np.percentile(radius, 95)) if len(radius) else None,
        },
        "mtf50_cross_validation": {
            "roi_attempts": int(len(rois)),
            "roi_valid": int(rois["valid"].sum()) if len(rois) else 0,
            "valid_fraction": float(rois["valid"].mean()) if len(rois) else None,
            "spearman_rho_vs_imatest": float(rho) if math.isfinite(rho) else None,
            "mae_cy_px_vs_imatest": mae if math.isfinite(mae) else None,
        },
        "gates": {
            "coverage_at_least_0_95": bool(len(successful) / total >= 0.95),
            "condition_cells_equal_50": bool(successful["condition"].nunique() == 50) if len(successful) else False,
            "median_center_error_at_most_0_15_radius": bool(len(center) and np.median(center) <= 0.15),
            "p95_center_error_at_most_0_35_radius": bool(len(center) and np.percentile(center, 95) <= 0.35),
            "roi_valid_fraction_at_least_0_95": bool(len(rois) and rois["valid"].mean() >= 0.95),
            "exact_crop_fraction_at_least_0_95": bool(
                len(exact_crop) and np.mean(exact_crop == 0.0) >= 0.95
            ),
            "mtf_spearman_at_least_0_80": bool(math.isfinite(rho) and rho >= 0.80),
            "mtf_mae_at_most_0_03_cy_px": bool(math.isfinite(mae) and mae <= 0.03),
        },
    }


def validate_exposed(
    cohort_path: Path,
    raw_root: Path,
    output: Path,
    *,
    limit: int | None = None,
    representative: bool = False,
) -> dict[str, object]:
    cohort = pd.read_csv(cohort_path)
    grouped = list(cohort.groupby("frame_id", sort=True))
    if representative:
        frame_table = cohort.drop_duplicates("frame_id").sort_values(
            ["condition", "frame_id"]
        )
        keys = set(frame_table.groupby("condition", sort=True).head(1)["frame_id"])
        grouped = [(key, rows) for key, rows in cohort.groupby("frame_id", sort=True) if key in keys]
    if limit is not None:
        grouped = grouped[:limit]
    estimator = _load_estimator()
    frame_records: list[dict[str, object]] = []
    roi_records: list[dict[str, object]] = []
    for index, (frame_id, rows) in enumerate(grouped, start=1):
        try:
            frame, rois = _validate_frame(rows, raw_root, estimator)
            frame["status"] = "ok"
            frame["reason"] = "ok"
            frame_records.append(frame)
            roi_records.extend(rois)
        except Exception as exc:  # fail closed while preserving the audit row
            first = rows.iloc[0]
            frame_records.append(
                {
                    "frame_id": frame_id,
                    "condition": first["condition"],
                    "speed_kmh": int(first["speed_kmh"]),
                    "distance_m": float(first["distance_m"]),
                    "iso": int(first["iso"]),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if index % 10 == 0 or index == len(grouped):
            print(f"locator validation {index}/{len(grouped)}", flush=True)
    frames = pd.DataFrame(frame_records)
    rois = pd.DataFrame(roi_records)
    output.mkdir(parents=True, exist_ok=True)
    frames.to_csv(output / "locator_frame_results.csv", index=False)
    rois.to_csv(output / "locator_roi_results.csv", index=False)
    summary = _summary(frames, rois, len(grouped))
    (output / "locator_development_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--representative", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    summary = validate_exposed(
        arguments.cohort,
        arguments.raw_root,
        arguments.output,
        limit=arguments.limit,
        representative=arguments.representative,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
