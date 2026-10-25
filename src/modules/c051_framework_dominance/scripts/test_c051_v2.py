"""Regression tests for the corrected C051 detector-input contract."""

from __future__ import annotations

import numpy as np

import c051_v2_common as v2


def test_to_detector_u8_clips_scales_and_rounds() -> None:
    restored = np.asarray([-0.1, 0.0, 1.0 / 255.0, 0.5, 1.0, 1.1], dtype=np.float32)
    converted = v2.to_detector_u8(restored)
    assert converted.dtype == np.uint8
    assert converted.tolist() == [0, 0, 1, 128, 255, 255]


def test_detector_roundtrip_preserves_unit_scale_with_quantization_bound() -> None:
    restored = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    detector_input = v2.to_detector_u8(restored).astype(np.float32) / 255.0
    assert float(np.max(np.abs(detector_input - restored))) <= (0.5 / 255.0 + 1e-7)
