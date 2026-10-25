"""Frozen slanted-edge MTF50 estimators for MERIT C057.

Version 2 changes edge *localization* only.  A Gaussian-smoothed copy is used
to choose the signed-gradient polarity and fit the edge line; the ESF, LSF and
MTF are always computed from the original input pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class MTF50Result:
    """MTF50 value and fail-closed diagnostics for one ROI."""

    mtf50: float
    reason: str
    orientation: str
    polarity: str | None
    first_stage_support_rows: int
    total_rows: int
    slope: float | None
    intercept: float | None
    localization_sigma_px: float
    esf_uses_original_pixels: bool = True

    @property
    def valid(self) -> bool:
        return math.isfinite(self.mtf50)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mtf50"] = self.mtf50 if self.valid else None
        payload["valid"] = self.valid
        return payload


def _canonicalize(roi: np.ndarray, orientation: str) -> np.ndarray:
    if roi.ndim != 2 or min(roi.shape) < 32:
        raise ValueError("ROI must be a 2D edge crop of at least 32 px per axis")
    data = roi.astype(np.float64)
    if orientation == "horizontal_edge":
        return data.T
    if orientation != "vertical_edge":
        raise ValueError(f"Unknown edge orientation: {orientation}")
    return data


def _fit_two_stage_line(edge: np.ndarray) -> tuple[int, float, float] | None:
    """Fit the frozen two-stage OLS line and return support, slope, intercept."""

    rows = np.arange(edge.size, dtype=np.float64)
    slope, intercept = np.polyfit(rows, edge.astype(np.float64), 1)
    residual = np.abs(edge - (slope * rows + intercept))
    good = residual < 6.0
    support = int(good.sum())
    if support < max(20, int(0.5 * edge.size)):
        return None
    slope, intercept = np.polyfit(rows[good], edge[good], 1)
    return support, float(slope), float(intercept)


def _compute_mtf50_from_line(
    original: np.ndarray,
    *,
    slope: float,
    intercept: float,
) -> tuple[float, str]:
    """Compute MTF50 from unmodified pixels using the frozen v1 ESF path."""

    height, width = original.shape
    rows = np.arange(height, dtype=np.float64)
    columns = np.arange(width, dtype=np.float64)
    oversample = 4
    position = (
        columns[None, :] - (slope * rows[:, None] + intercept)
    ) * oversample
    selected = np.abs(position) < 30 * oversample
    positions = position[selected].ravel()
    values = original[selected].ravel()
    if positions.size < 1000:
        return math.nan, "insufficient_esf_samples"
    if np.ptp(values) < 5.0:
        return math.nan, "insufficient_roi_contrast"

    order = np.argsort(positions)
    positions, values = positions[order], values[order]
    bins = np.arange(positions.min(), positions.max(), 1.0)
    indices = np.digitize(positions, bins)
    esf = np.asarray(
        [
            values[indices == index].mean()
            if np.any(indices == index)
            else np.nan
            for index in range(1, len(bins))
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(esf)
    if int(valid.sum()) < 32:
        return math.nan, "insufficient_valid_esf_bins"
    esf = np.interp(np.arange(len(esf)), np.flatnonzero(valid), esf[valid])
    lsf = np.gradient(esf) * np.hanning(len(esf))
    spectrum = np.abs(np.fft.rfft(lsf))
    if not np.isfinite(spectrum[0]) or spectrum[0] <= 1e-12:
        return math.nan, "invalid_mtf_dc"
    mtf = spectrum / spectrum[0]
    frequency = np.fft.rfftfreq(len(lsf), d=1.0 / oversample)
    band = frequency <= 0.6
    frequency, mtf = frequency[band], mtf[band]
    crossings = np.flatnonzero(mtf < 0.5)
    if crossings.size == 0:
        return math.nan, "no_mtf50_crossing"
    if crossings[0] == 0:
        return math.nan, "mtf_below_half_at_dc"
    index = int(crossings[0])
    value = float(
        np.interp(
            0.5,
            [mtf[index], mtf[index - 1]],
            [frequency[index], frequency[index - 1]],
        )
    )
    if not 0.0 < value <= 0.6:
        return math.nan, "mtf50_out_of_range"
    return value, "ok"


def estimate_mtf50_v1(roi: np.ndarray, orientation: str) -> MTF50Result:
    """Numerically reproduce the original C007 in-house estimator."""

    data = _canonicalize(roi, orientation)
    gradient = np.abs(np.gradient(data, axis=1))
    edge = gradient.argmax(axis=1).astype(np.float64)
    fitted = _fit_two_stage_line(edge)
    if fitted is None:
        return MTF50Result(
            math.nan,
            "insufficient_line_support",
            orientation,
            "absolute",
            0,
            data.shape[0],
            None,
            None,
            0.0,
        )
    support, slope, intercept = fitted
    value, reason = _compute_mtf50_from_line(
        data, slope=slope, intercept=intercept
    )
    return MTF50Result(
        value,
        reason,
        orientation,
        "absolute",
        support,
        data.shape[0],
        slope,
        intercept,
        0.0,
    )


def _signed_candidate(
    gradient: np.ndarray, polarity: str
) -> tuple[int, float, float, float] | None:
    if polarity == "positive":
        edge = gradient.argmax(axis=1).astype(np.float64)
        strength = gradient[np.arange(gradient.shape[0]), edge.astype(int)]
    elif polarity == "negative":
        edge = gradient.argmin(axis=1).astype(np.float64)
        strength = -gradient[np.arange(gradient.shape[0]), edge.astype(int)]
    else:  # pragma: no cover - private caller invariant
        raise ValueError(polarity)
    fitted = _fit_two_stage_line(edge)
    if fitted is None:
        return None
    support, slope, intercept = fitted
    return support, slope, intercept, float(np.sum(strength))


def estimate_mtf50_v2(
    roi: np.ndarray,
    orientation: str,
    *,
    localization_sigma_px: float = 0.75,
) -> MTF50Result:
    """Estimate MTF50 with frozen v2 localization and unaltered ESF pixels."""

    if localization_sigma_px <= 0:
        raise ValueError("localization_sigma_px must be positive")
    original = _canonicalize(roi, orientation)
    localization = cv2.GaussianBlur(
        original,
        (0, 0),
        sigmaX=localization_sigma_px,
        sigmaY=localization_sigma_px,
        borderType=cv2.BORDER_REFLECT_101,
    )
    signed_gradient = np.gradient(localization, axis=1)
    candidates = {
        polarity: _signed_candidate(signed_gradient, polarity)
        for polarity in ("positive", "negative")
    }
    available = [
        (fit[0], fit[3], polarity, fit)
        for polarity, fit in candidates.items()
        if fit is not None
    ]
    if not available:
        return MTF50Result(
            math.nan,
            "insufficient_line_support_both_polarities",
            orientation,
            None,
            0,
            original.shape[0],
            None,
            None,
            localization_sigma_px,
        )
    # Line support is primary. Edge strength only resolves an exact support tie.
    support, _, polarity, fitted = max(
        available, key=lambda item: (item[0], item[1])
    )
    _, slope, intercept, _ = fitted
    value, reason = _compute_mtf50_from_line(
        original, slope=slope, intercept=intercept
    )
    return MTF50Result(
        value,
        reason,
        orientation,
        polarity,
        support,
        original.shape[0],
        slope,
        intercept,
        localization_sigma_px,
    )
