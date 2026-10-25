"""Scale-checked RAW quadrant-chart locator for MERIT C058 v3h development.

The v1 locator ranked self-similar checker templates across a very broad scale
range.  Small brick or crack-board patterns could therefore win and their
radius was then used in the native MTF50 conversion.  V2 separates the tasks:
large templates locate the chart centre and angle, a radial-gradient stage
estimates the circular target radius, and an inverse-affine footprint check
rejects partial charts before any MTF50 value is produced.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


WORKING_SIZE = (1024, 683)
SEARCH_HEIGHT = 280
TOP_PADDING = 120
CANONICAL_SIZE = 1024
CANONICAL_RADIUS = 320.0
TEMPLATE_RADII = (68, 76, 84, 92, 100)
TEMPLATE_ANGLES = (-5, 0, 5, 10, 15, 20)
RADIAL_RADII = np.arange(60, 103, dtype=np.float32)
RADIAL_ANGLES = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False, dtype=np.float32)
MIN_TEMPLATE_SCORE = 0.55
MIN_SOURCE_COVERAGE = 0.995
MIN_RADIAL_COVERAGE = 0.75


class LocatorFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class LocatorV2Result:
    center_x: float
    center_y: float
    radius: float
    angle_deg: float
    template_score: float
    radial_score: float
    radial_peak_margin: float
    radial_coverage: float
    minimum_roi_source_coverage: float
    eligible: bool
    rejection_reason: str
    original_width: int
    original_height: int

    @property
    def score(self) -> float:
        return self.template_score

    def to_dict(self) -> dict[str, bool | float | int | str]:
        return asdict(self)


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
    uncertain = (np.abs(xr) < 0.08 * radius) | (np.abs(yr) < 0.08 * radius)
    template[inside & uncertain] = 0.0
    return template


TEMPLATES = tuple(
    (radius, angle, _checker_template(radius, angle))
    for radius in TEMPLATE_RADII
    for angle in TEMPLATE_ANGLES
)


def _coarse_candidate(working: np.ndarray) -> tuple[float, float, float, float]:
    padded = np.pad(working, ((TOP_PADDING, 0), (0, 0)), mode="edge")
    search = padded[: TOP_PADDING + SEARCH_HEIGHT, : WORKING_SIZE[0]]
    best_score = -math.inf
    best: tuple[float, float, float] | None = None
    for radius, angle, template in TEMPLATES:
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(np.abs(response))
        if float(score) > best_score:
            best_score = float(score)
            best = (
                location[0] + (template.shape[1] - 1) / 2,
                location[1] + (template.shape[0] - 1) / 2 - TOP_PADDING,
                float(angle),
            )
    if best is None:
        raise LocatorFailure("no coarse chart candidate")
    return best[0], best[1], best[2], best_score


def _radial_profile(
    gx: np.ndarray, gy: np.ndarray, center_x: float, center_y: float
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(RADIAL_ANGLES)[None, :]
    sine = np.sin(RADIAL_ANGLES)[None, :]
    radii = RADIAL_RADII[:, None]
    map_x = (center_x + radii * cosine).astype(np.float32)
    map_y = (center_y + radii * sine).astype(np.float32)
    valid = (
        (map_x >= 0)
        & (map_x <= gx.shape[1] - 1)
        & (map_y >= 0)
        & (map_y <= gx.shape[0] - 1)
    )
    sampled_x = cv2.remap(gx, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sampled_y = cv2.remap(gy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    radial = np.abs(sampled_x * cosine + sampled_y * sine)
    radial[~valid] = np.nan
    scores = np.nanpercentile(radial, 65, axis=1)
    coverage = np.mean(valid, axis=1)
    return scores.astype(np.float32), coverage.astype(np.float32)


def _refine_circle(working: np.ndarray, center_x: float, center_y: float):
    smooth = cv2.GaussianBlur(working, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT_101)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    candidates = []
    for dy in (-6.0, 0.0, 6.0):
        for dx in (-6.0, 0.0, 6.0):
            scores, coverage = _radial_profile(gx, gy, center_x + dx, center_y + dy)
            eligible = np.where(coverage >= 0.75, scores, -np.inf)
            index = int(np.argmax(eligible))
            candidates.append(
                (
                    float(eligible[index]),
                    center_x + dx,
                    center_y + dy,
                    float(RADIAL_RADII[index]),
                    float(coverage[index]),
                    scores,
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, refined_x, refined_y, radius, coverage, profile = candidates[0]
    peak_index = int(round(radius - float(RADIAL_RADII[0])))
    separated = profile.copy()
    separated[max(0, peak_index - 3) : peak_index + 4] = -np.inf
    second = float(np.max(separated))
    margin = float((score - second) / max(abs(score), 1e-6))
    return refined_x, refined_y, radius, score, margin, coverage


def _canonical_rois() -> tuple[tuple[int, int, int, int], ...]:
    center, radius = CANONICAL_SIZE / 2, CANONICAL_RADIUS
    specs = (
        (0.00, -0.50, 0.36, 0.62),
        (0.00, +0.50, 0.36, 0.62),
        (-0.53, 0.00, 0.62, 0.36),
        (+0.53, 0.00, 0.62, 0.36),
    )
    rois = []
    for dx, dy, width, height in specs:
        cx, cy = center + dx * radius, center + dy * radius
        rois.append(
            (
                round(cx - width * radius / 2),
                round(cy - height * radius / 2),
                round(cx + width * radius / 2),
                round(cy + height * radius / 2),
            )
        )
    return tuple(rois)


def normalization_matrix(result: LocatorV2Result) -> np.ndarray:
    scale = CANONICAL_RADIUS / result.radius
    matrix = cv2.getRotationMatrix2D(
        (result.center_x, result.center_y), result.angle_deg, scale
    )
    matrix[0, 2] += CANONICAL_SIZE / 2 - result.center_x
    matrix[1, 2] += CANONICAL_SIZE / 2 - result.center_y
    return matrix


def _roi_source_coverages_from_geometry(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float,
    angle: float,
) -> tuple[float, float, float, float]:
    provisional = LocatorV2Result(
        center_x, center_y, radius, angle, 0, 0, 0, 0, 0, False, "", width, height
    )
    inverse = cv2.invertAffineTransform(normalization_matrix(provisional))
    fractions = []
    for x1, y1, x2, y2 in _canonical_rois():
        xs = np.linspace(x1, x2 - 1, 17, dtype=np.float32)
        ys = np.linspace(y1, y2 - 1, 17, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        source_x = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
        source_y = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
        valid = (source_x >= 0) & (source_x < width) & (source_y >= 0) & (source_y < height)
        fractions.append(float(np.mean(valid)))
    return tuple(fractions)


def roi_source_coverages(result: LocatorV2Result) -> tuple[float, float, float, float]:
    """Return the actual-source fraction for each canonical MTF ROI."""

    return _roi_source_coverages_from_geometry(
        result.original_width,
        result.original_height,
        result.center_x,
        result.center_y,
        result.radius,
        result.angle_deg,
    )


def assess_chart(gray: np.ndarray) -> LocatorV2Result:
    if gray.ndim != 2 or min(gray.shape) < 64:
        raise LocatorFailure("expected a two-dimensional grayscale image")
    working = cv2.resize(gray, WORKING_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    coarse_x, coarse_y, angle, template_score = _coarse_candidate(working)
    center_x, center_y, radius, radial_score, margin, radial_coverage = _refine_circle(
        working, coarse_x, coarse_y
    )
    scale_x = gray.shape[1] / WORKING_SIZE[0]
    scale_y = gray.shape[0] / WORKING_SIZE[1]
    scale = math.sqrt(scale_x * scale_y)
    raw_x, raw_y, raw_radius = center_x * scale_x, center_y * scale_y, radius * scale
    source_coverages = _roi_source_coverages_from_geometry(
        gray.shape[1], gray.shape[0], raw_x, raw_y, raw_radius, angle
    )
    source_coverage = min(source_coverages)
    if template_score < MIN_TEMPLATE_SCORE:
        reason = "low_template_score"
    elif source_coverage < MIN_SOURCE_COVERAGE:
        reason = "partial_chart"
    elif radial_coverage < MIN_RADIAL_COVERAGE:
        reason = "insufficient_radial_coverage"
    else:
        reason = "ok"
    return LocatorV2Result(
        center_x=float(raw_x),
        center_y=float(raw_y),
        radius=float(raw_radius),
        angle_deg=float(angle),
        template_score=float(template_score),
        radial_score=float(radial_score),
        radial_peak_margin=float(margin),
        radial_coverage=float(radial_coverage),
        minimum_roi_source_coverage=float(source_coverage),
        eligible=reason == "ok",
        rejection_reason=reason,
        original_width=int(gray.shape[1]),
        original_height=int(gray.shape[0]),
    )


def locate_chart(gray: np.ndarray) -> LocatorV2Result:
    result = assess_chart(gray)
    if not result.eligible:
        raise LocatorFailure(f"{result.rejection_reason}: score={result.template_score:.6f}")
    return result


def normalize_chart(gray: np.ndarray, result: LocatorV2Result):
    normalized = cv2.warpAffine(
        gray,
        normalization_matrix(result),
        (CANONICAL_SIZE, CANONICAL_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rois = []
    orientations = ("vertical_edge", "vertical_edge", "horizontal_edge", "horizontal_edge")
    coverages = roi_source_coverages(result)
    for index, ((x1, y1, x2, y2), orientation, source_coverage) in enumerate(
        zip(_canonical_rois(), orientations, coverages), start=1
    ):
        rois.append(
            {
                "roi_id": index,
                "orientation": orientation,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "source_coverage": float(source_coverage),
            }
        )
    return normalized, rois
