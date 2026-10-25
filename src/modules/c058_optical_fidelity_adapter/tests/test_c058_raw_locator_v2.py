from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_raw_locator_v2.py"
SPEC = importlib.util.spec_from_file_location("c058_raw_locator_v2_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def chart(center=(240, 90), radius=78, angle=5):
    image = np.full((683, 1024), 128, dtype=np.uint8)
    yy, xx = np.mgrid[:683, :1024].astype(np.float32)
    x, y = xx - center[0], yy - center[1]
    radians = np.deg2rad(angle)
    xr = np.cos(radians) * x + np.sin(radians) * y
    yr = -np.sin(radians) * x + np.cos(radians) * y
    inside = np.hypot(x, y) <= radius
    image[inside & (xr * yr >= 0)] = 35
    image[inside & (xr * yr < 0)] = 220
    return cv2.GaussianBlur(image, (0, 0), 0.8)


def test_full_quadrant_chart_is_eligible_with_refined_radius():
    result = MODULE.assess_chart(chart())
    assert result.eligible
    assert abs(result.radius - 78) <= 8
    assert result.minimum_roi_source_coverage >= MODULE.MIN_SOURCE_COVERAGE


def test_partial_chart_is_rejected_before_mtf_measurement():
    result = MODULE.assess_chart(chart(center=(240, 15)))
    assert not result.eligible
    assert result.rejection_reason in {"partial_chart", "insufficient_radial_coverage"}


def test_repeated_square_distractor_is_not_accepted_as_chart():
    image = np.full((683, 1024), 180, dtype=np.uint8)
    for y in range(20, 260, 65):
        for x in range(40, 800, 90):
            cv2.rectangle(image, (x, y), (x + 50, y + 40), 60, -1)
    result = MODULE.assess_chart(image)
    assert not result.eligible
    assert result.rejection_reason == "low_template_score"
