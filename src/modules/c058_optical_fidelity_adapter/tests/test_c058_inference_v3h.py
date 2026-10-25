from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/c058_inference_v3h.py"
SPEC = importlib.util.spec_from_file_location("c058_inference_v3h_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_small_native_crop_is_padded_then_restored_to_exact_shape():
    crop = np.arange(420 * 430, dtype=np.uint32).reshape(420, 430).astype(np.uint8)
    calls = []

    def tiled(image, model, device, **kwargs):
        calls.append((image.shape, kwargs))
        return image.astype(np.float32) / np.float32(255.0)

    output = MODULE.tiled_restore_native_u8(
        crop,
        object(),
        "cuda",
        tiled,
        tile_size=448,
        overlap=112,
        batch_size=1,
        blend_floor=0.05,
        amp_dtype="float16",
    )

    assert calls[0][0] == (448, 448)
    assert output.shape == crop.shape
    assert np.array_equal(output, crop)


def test_large_crop_is_not_padded():
    crop = np.full((500, 600), 127, dtype=np.uint8)

    def tiled(image, model, device, **kwargs):
        assert image is crop
        return image

    output = MODULE.tiled_restore_native_u8(
        crop,
        None,
        None,
        tiled,
        tile_size=448,
        overlap=112,
        batch_size=1,
        blend_floor=0.05,
        amp_dtype="float16",
    )
    assert np.array_equal(output, crop)
