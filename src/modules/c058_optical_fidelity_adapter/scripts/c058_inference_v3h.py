"""Boundary-safe native-crop inference helpers for C058 v3h."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np


def pad_to_minimum_tile(
    crop: np.ndarray, tile_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad a native crop to one tile and return its original shape."""

    if crop.ndim != 2 or crop.dtype != np.uint8:
        raise ValueError("crop must be 2D uint8")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    height, width = crop.shape
    bottom = max(0, int(tile_size) - height)
    right = max(0, int(tile_size) - width)
    if not bottom and not right:
        return crop, (height, width)
    border = cv2.BORDER_REFLECT_101 if min(height, width) > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(crop, 0, bottom, 0, right, borderType=border)
    return padded, (height, width)


def tiled_restore_native_u8(
    crop: np.ndarray,
    model: Any,
    device: Any,
    tiled_restore: Callable[..., np.ndarray],
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
    blend_floor: float,
    amp_dtype: str,
) -> np.ndarray:
    """Run frozen tiling on a safe padded crop and remove only added padding."""

    padded, (height, width) = pad_to_minimum_tile(crop, tile_size)
    restored = tiled_restore(
        padded,
        model,
        device,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        blend_floor=blend_floor,
        amp_dtype=amp_dtype,
    )
    if restored.shape != padded.shape:
        raise ValueError("tiled restoration changed padded spatial shape")
    restored = restored[:height, :width]
    if restored.dtype == np.uint8:
        return restored.copy()
    if not np.issubdtype(restored.dtype, np.floating):
        raise ValueError("tiled restoration returned unsupported dtype")
    if not np.isfinite(restored).all() or float(restored.min()) < 0 or float(restored.max()) > 1:
        raise ValueError("floating restoration must be finite and in [0, 1]")
    return np.rint(restored * np.float32(255.0)).astype(np.uint8)
