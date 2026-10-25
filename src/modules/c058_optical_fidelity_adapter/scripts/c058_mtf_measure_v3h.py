"""Source-aware native MTF50 measurement for C058 v3h."""

from __future__ import annotations

import math
from typing import Any


MIN_ROI_SOURCE_COVERAGE = 0.995
CANONICAL_RADIUS = 320.0


def measure_native(
    image,
    located: Any,
    locator: Any,
    estimator: Any,
) -> list[dict[str, Any]]:
    canonical, rois = locator.normalize_chart(image, located)
    measured = []
    for roi in rois:
        source_coverage = float(roi.get("source_coverage", 1.0))
        base = {
            "roi_id": int(roi["roi_id"]),
            "orientation": str(roi["orientation"]),
            "source_coverage": source_coverage,
        }
        if source_coverage < MIN_ROI_SOURCE_COVERAGE:
            measured.append(
                {
                    **base,
                    "valid": False,
                    "reason": "partial_roi_source",
                    "mtf50_canonical_cy_px": None,
                    "mtf50_native_cy_px": None,
                }
            )
            continue
        patch = canonical[
            int(roi["y1"]) : int(roi["y2"]),
            int(roi["x1"]) : int(roi["x2"]),
        ]
        result = estimator.estimate_mtf50_v2(
            patch, str(roi["orientation"]), localization_sigma_px=0.75
        )
        canonical_value = float(result.mtf50) if result.valid else math.nan
        native_value = canonical_value * CANONICAL_RADIUS / float(located.radius)
        if not result.valid:
            valid, reason = False, str(result.reason)
        elif not math.isfinite(canonical_value) or canonical_value <= 0:
            valid, reason = False, "canonical_mtf50_nonfinite_or_nonpositive"
        elif canonical_value > 0.5:
            valid, reason = False, "canonical_nyquist_exceeded"
        elif not math.isfinite(native_value) or native_value <= 0:
            valid, reason = False, "native_mtf50_nonfinite_or_nonpositive"
        elif native_value > 0.5:
            valid, reason = False, "native_nyquist_exceeded"
        else:
            valid, reason = True, str(result.reason)
        measured.append(
            {
                **base,
                "valid": valid,
                "reason": reason,
                "mtf50_canonical_cy_px": canonical_value if valid else None,
                "mtf50_native_cy_px": native_value if valid else None,
            }
        )
    return measured
