from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("measure_width_chart_raw.py")
SPEC = importlib.util.spec_from_file_location("measure_width_chart_raw", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_patch(width_px: int, noise_sigma: float = 0.35) -> object:
    side = 256
    image = np.full((side, side), 165.0, dtype=np.float32)
    points = []
    for x in range(side):
        y = int(round(128.0 + 8.0 * np.sin(2.0 * np.pi * x / 180.0)))
        points.append((x, y))
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, 60.0, width_px)
    image = cv2.GaussianBlur(image, (0, 0), 0.75)
    rng = np.random.default_rng(1234)
    image += rng.normal(0.0, noise_sigma, image.shape).astype(np.float32)
    identity = np.eye(3, dtype=np.float64)
    return MODULE.RectifiedPatch(
        gray=np.clip(image, 0, 255),
        original_to_rectified=identity,
        rectified_to_original=identity,
    )


class WidthMeasurementTests(unittest.TestCase):
    def test_identity_normal_scale(self) -> None:
        scale = MODULE.rectified_normal_to_sensor_scale(np.eye(3), 100.0, 100.0, 0.2)
        self.assertAlmostEqual(scale, 1.0, places=8)

    def test_resolved_line_is_estimable(self) -> None:
        result, trace = MODULE.measure_patch(synthetic_patch(4), "SYN001", 1.0)
        self.assertEqual(result["status"], "ESTIMABLE")
        self.assertGreater(float(result["apparent_width_sensor_px"]), 3.0)
        self.assertLess(float(result["apparent_width_sensor_px"]), 7.0)
        self.assertGreater(int(result["retained_profiles"]), 50)
        self.assertEqual(trace["trace_source"], "RAW_ONLY")

    def test_subpixel_width_is_predeclared_non_estimable(self) -> None:
        result, _ = MODULE.measure_patch(synthetic_patch(2), "SYN002", 0.2)
        self.assertEqual(result["status"], "PREDECLARED_SUBPIXEL_NON_ESTIMABLE")
        self.assertEqual(result["apparent_width_mm_design_gsd"], "")

    def test_blank_patch_fails_qc(self) -> None:
        side = 256
        identity = np.eye(3, dtype=np.float64)
        patch = MODULE.RectifiedPatch(
            gray=np.full((side, side), 165.0, dtype=np.float32),
            original_to_rectified=identity,
            rectified_to_original=identity,
        )
        result, _ = MODULE.measure_patch(patch, "SYN003", 1.0)
        self.assertEqual(result["status"], "QC_NON_ESTIMABLE")


if __name__ == "__main__":
    unittest.main()
