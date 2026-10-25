from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_width_chart_raw as v1
import measure_width_chart_raw_v2 as v2


def synthetic_patch(orientation: str, width_px: int = 4) -> v1.RectifiedPatch:
    side = 256
    image = np.full((side, side), 165.0, dtype=np.float32)
    points = []
    for axis in range(side):
        path = int(round(128.0 + 8.0 * np.sin(2.0 * np.pi * axis / 180.0)))
        points.append((axis, path) if orientation == "HORIZONTAL" else (path, axis))
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, 60.0, width_px)
    image = cv2.GaussianBlur(image, (0, 0), 0.75)
    rng = np.random.default_rng(4321)
    image += rng.normal(0.0, 0.35, image.shape).astype(np.float32)
    identity = np.eye(3, dtype=np.float64)
    return v1.RectifiedPatch(
        gray=np.clip(image, 0, 255),
        original_to_rectified=identity,
        rectified_to_original=identity,
    )


class OrientationAwareWidthMeasurementTests(unittest.TestCase):
    def test_horizontal_orientation(self) -> None:
        result, trace = v2.measure_patch(synthetic_patch("HORIZONTAL"), "SYNH", 1.0)
        self.assertEqual(result["trace_orientation"], "HORIZONTAL")
        self.assertEqual(result["status"], "ESTIMABLE")
        self.assertEqual(trace["trace_orientation"], "HORIZONTAL")

    def test_vertical_orientation(self) -> None:
        result, trace = v2.measure_patch(synthetic_patch("VERTICAL"), "SYNV", 1.0)
        self.assertEqual(result["trace_orientation"], "VERTICAL")
        self.assertEqual(result["status"], "ESTIMABLE")
        self.assertEqual(trace["trace_orientation"], "VERTICAL")
        self.assertGreater(float(result["apparent_width_sensor_px"]), 3.0)
        self.assertLess(float(result["apparent_width_sensor_px"]), 7.0)

    def test_orientation_selection_does_not_depend_on_nominal_width(self) -> None:
        patch = synthetic_patch("VERTICAL")
        result_small, _ = v2.measure_patch(patch, "SYNA", 0.3)
        result_large, _ = v2.measure_patch(patch, "SYNB", 1.0)
        self.assertEqual(result_small["trace_orientation"], result_large["trace_orientation"])


if __name__ == "__main__":
    unittest.main()
