"""Shared C018 degradation-conditioned restoration helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
C018 = HERE.parent
REVISION = C018.parent
C009 = REVISION / "c009_integrated_experiment"
C010 = REVISION / "c010_merit_v2_development"
C006 = REVISION / "c006_execution"
DEVELOPMENT_PROTOCOL = C018 / "protocol" / "development_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_file(name: str, path: Path):
    parent = str(path.resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def psnr(clean: np.ndarray, candidate: np.ndarray) -> float:
    error = float(
        np.mean((clean.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    )
    return 99.0 if error <= 1e-14 else float(10.0 * math.log10(1.0 / error))


def gradient_mae(clean: np.ndarray, candidate: np.ndarray) -> float:
    clean_dx = np.diff(clean, axis=1)
    clean_dy = np.diff(clean, axis=0)
    candidate_dx = np.diff(candidate, axis=1)
    candidate_dy = np.diff(candidate, axis=0)
    return float(
        0.5
        * (
            np.mean(np.abs(clean_dx - candidate_dx))
            + np.mean(np.abs(clean_dy - candidate_dy))
        )
    )


def edge_normal_overshoot_p95(clean: np.ndarray, candidate: np.ndarray) -> float:
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 1e-6]
    if positive.size == 0:
        return 0.0
    threshold = float(np.percentile(positive, 75.0))
    mask = magnitude >= threshold
    yy, xx = np.indices(clean.shape, dtype=np.float32)
    nx = gx / np.maximum(magnitude, 1e-6)
    ny = gy / np.maximum(magnitude, 1e-6)
    plus_x, plus_y = xx + nx * 1.5, yy + ny * 1.5
    minus_x, minus_y = xx - nx * 1.5, yy - ny * 1.5
    remap = lambda image, x, y: cv2.remap(  # noqa: E731
        image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    clean_plus = remap(clean, plus_x, plus_y)
    clean_minus = remap(clean, minus_x, minus_y)
    candidate_plus = remap(candidate, plus_x, plus_y)
    candidate_minus = remap(candidate, minus_x, minus_y)
    lower = np.minimum(clean_plus, clean_minus)
    upper = np.maximum(clean_plus, clean_minus)
    overshoot = np.maximum.reduce(
        [
            candidate_plus - upper,
            lower - candidate_plus,
            candidate_minus - upper,
            lower - candidate_minus,
            np.zeros_like(clean),
        ]
    )
    return float(np.percentile(overshoot[mask], 95.0))


def motion_kernel(length: int, angle: float) -> np.ndarray:
    size = int(length) if int(length) % 2 == 1 else int(length) + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0
    )
    kernel = cv2.warpAffine(kernel, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return kernel / float(kernel.sum())


def restore_physics(
    degraded: np.ndarray,
    family: str,
    parameters: dict[str, Any],
    gain: float = 0.0,
) -> np.ndarray:
    source = degraded.astype(np.float32, copy=False)
    if family == "illumination_gradient_vignetting":
        height, width = source.shape
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        edge_gain = float(parameters["edge_gain"])
        vertical_gain = float(parameters["vertical_gain"])
        field = (1.0 - (1.0 - edge_gain) * x**2) * (
            1.0 - (1.0 - vertical_gain) * y
        )
        output = source / np.maximum(field, np.float32(1e-4))
    elif family == "veiling_scattering":
        output = (source - float(parameters["veil"])) / float(parameters["contrast"])
    elif family == "saturation_clipping":
        clip_high = float(parameters["clip_high"])
        amplification = float(parameters["gain"])
        output = source * np.float32(clip_high / amplification)
        plateau = source >= np.float32(1.0 - 1e-6)
        if np.any(plateau):
            lower_bound = np.float32(clip_high / amplification)
            local = cv2.GaussianBlur(output, (0, 0), 2.0)
            output[plateau] = np.maximum(local[plateau], lower_bound)
    elif family == "linear_motion_psf":
        kernel = motion_kernel(
            int(parameters["length_px"]), float(parameters["angle_deg"])
        )
        reblurred = cv2.filter2D(
            source, -1, kernel, borderType=cv2.BORDER_REFLECT_101
        )
        output = source + np.float32(gain) * (source - reblurred)
    elif family == "spatially_varying_blur":
        left = gaussian_filter(
            source, float(parameters["sigma_left_px"]), mode="reflect"
        )
        right = gaussian_filter(
            source, float(parameters["sigma_right_px"]), mode="reflect"
        )
        weight = np.linspace(0.0, 1.0, source.shape[1], dtype=np.float32)[None, :]
        reblurred = left * (1.0 - weight) + right * weight
        output = source + np.float32(gain) * (source - reblurred)
    elif family == "identity":
        output = source
    else:
        raise ValueError(f"Unsupported physics family: {family}")
    return np.clip(output, 0.0, 1.0).astype(np.float32)
