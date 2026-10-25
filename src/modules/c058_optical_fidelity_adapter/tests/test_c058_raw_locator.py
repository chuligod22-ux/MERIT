from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "c058_raw_locator.py"
SPEC = importlib.util.spec_from_file_location("c058_raw_locator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_chart(
    *, center: tuple[int, int] = (145, 68), radius: int = 58, angle: float = 7.0
) -> np.ndarray:
    image = np.full((683, 1024), 150, dtype=np.uint8)
    image[:] += np.linspace(-35, 30, image.shape[1], dtype=np.int16)[None, :].astype(np.uint8)
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]].astype(np.float32)
    x = xx - center[0]
    y = yy - center[1]
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))
    xr = c * x + s * y
    yr = -s * x + c * y
    disk = x * x + y * y <= radius * radius
    image[disk & (xr * yr >= 0)] = 35
    image[disk & (xr * yr < 0)] = 205
    return cv2.GaussianBlur(image, (5, 5), 1.0)


def test_locator_finds_partial_synthetic_quadrant_chart() -> None:
    image = synthetic_chart(center=(138, 42), radius=62, angle=5.0)
    result = MODULE.locate_chart(image, working_size=(1024, 683))
    assert abs(result.center_x - 138) <= 8
    assert abs(result.center_y - 42) <= 8
    assert 40 <= result.radius <= 85
    assert result.score >= MODULE.MIN_ACCEPTED_SCORE


def test_locator_fails_closed_without_checker_pattern() -> None:
    image = np.tile(np.linspace(20, 220, 1024, dtype=np.uint8), (683, 1))
    with pytest.raises(MODULE.LocatorFailure):
        MODULE.locate_chart(image, working_size=(1024, 683))


def test_normalization_and_roi_contract() -> None:
    image = synthetic_chart()
    result = MODULE.locate_chart(image, working_size=(1024, 683))
    normalized, rois = MODULE.normalize_chart(image, result)
    assert normalized.shape == (MODULE.CANONICAL_SIZE, MODULE.CANONICAL_SIZE)
    assert [roi["roi_id"] for roi in rois] == [1, 2, 3, 4]
    assert [roi["orientation"] for roi in rois] == [
        "vertical_edge",
        "vertical_edge",
        "horizontal_edge",
        "horizontal_edge",
    ]
    for roi in rois:
        assert 0 <= roi["x1"] < roi["x2"] <= MODULE.CANONICAL_SIZE
        assert 0 <= roi["y1"] < roi["y2"] <= MODULE.CANONICAL_SIZE


def test_sparse_exact_crop_origin() -> None:
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 256, (420, 620), dtype=np.uint8)
    extract = raw[17:217, 93:393].copy()
    x, y, mae = MODULE.find_exact_crop_origin(raw, extract)
    assert (x, y) == (93, 17)
    assert mae == 0.0
