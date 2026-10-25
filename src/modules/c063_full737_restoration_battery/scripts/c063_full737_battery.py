#!/usr/bin/env python3
"""Freeze, execute, and aggregate the 737-frame controlled restoration battery.

The experiment extends the legacy 12-crop detector-response table without using
detector response to select frames or crops.  Every cam2 frame contributes its
fixed upper-40% lining region.  A deterministic synthetic degradation is
applied to each operational source frame and every restoration method receives
the identical degraded array.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage.restoration import (
    denoise_nl_means,
    denoise_tv_chambolle,
    richardson_lucy,
    unsupervised_wiener,
)
from torch.torch_version import TorchVersion


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve()
PROTOCOL_DIR = BASE / "protocol"
OUTPUT = BASE / "artifacts" / "full737_battery_v1"
MANIFEST = PROTOCOL_DIR / "full737_battery_manifest_v1.csv"
FREEZE = PROTOCOL_DIR / "full737_battery_freeze_v1.json"
STATUS = OUTPUT / "status.json"

C062_MANIFEST = ROOT / (
    "05_MERIT_revision/40_수정본/c062_cam2_full_cohort_audit/"
    "protocol/full737_manifest_v2.csv"
)
MODEL_CODE = ROOT / "05_MERIT_revision/40_수정본/c006_execution/c006_model.py"
INFERENCE_CODE = ROOT / (
    "05_MERIT_revision/40_수정본/c006_execution/c006_operational_inference.py"
)
TASK_OFF_CHECKPOINT = ROOT / (
    "05_MERIT_revision/40_수정본/c006_execution/runs/final/task_off_seed1/final.pt"
)
TASK_ON_CHECKPOINT = ROOT / (
    "05_MERIT_revision/40_수정본/c006_execution/runs/final/task_on_seed1/final.pt"
)
DETECTOR_CODE = ROOT / (
    "05_MERIT_revision/40_수정본/c005_freeze/d2_detector/d2_detector.py"
)
DETECTOR_CHECKPOINT = ROOT / (
    "05_MERIT_revision/40_수정본/c005_freeze/d2_detector/artifacts/"
    "d2_training_run/best_d2_resnet34_unet.pt"
)
DETECTOR_PROTOCOL = ROOT / (
    "05_MERIT_revision/40_수정본/c005_freeze/d2_detector/protocol.json"
)
ADAPTER_CODE = ROOT / (
    "05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/"
    "c058_adapter_v2.py"
)
INDEPENDENT_VERIFIER = BASE / "scripts" / "independent_verify_c063.py"
GATE_TABLE = ROOT / (
    "05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/artifacts/"
    "safety_router_v3i_development/all50_margin_gated_v1/gate_table.csv"
)

SCHEMA = "MERIT-C063-FULL737-BATTERY-v1"
EXPECTED_FRAMES = 737
EXPECTED_CELLS = 50
PRIMARY_CELLS = 45
TOP_FRACTION = 0.40
CROP_HEIGHT = 896
CROP_WIDTH = 1344
THRESHOLD = 0.55
BLUR_H = 2.6
BLUR_V = 2.4
NOISE_DN = 2.5
CLASSICAL_TILE = 1024
CLASSICAL_OVERLAP = 64
BOOTSTRAP_COUNT = 10_000
PERMUTATION_COUNT = 100_000
RNG_SEED = 20261024
DEVELOPMENT_CONDITIONS = {
    "crack_d25_ISO100_V60",
    "crack_d25_ISO100_V80",
    "crack_d35_ISO100_V60",
    "crack_d35_ISO100_V80",
    "crack_d25_ISO200_V60",
}
METRICS = ("mean_probability", "maximum_probability", "area_fraction_at_0.55")
METHODS = (
    "degraded",
    "wiener_oracle",
    "richardson_lucy_oracle",
    "unsupervised_wiener_oracle",
    "wiener_wrong_psf",
    "tv_denoise",
    "nl_means",
    "unsharp",
    "nafnet_fidelity",
    "merit_core_adapter",
    "merit_routed_v3i",
)
PRIMARY_MERIT = "merit_core_adapter"


class BatteryError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise BatteryError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise BatteryError(f"cannot decode image: {path}")
    return image


def build_manifest() -> pd.DataFrame:
    source = pd.read_csv(C062_MANIFEST, dtype={"frame": str})
    if len(source) != EXPECTED_FRAMES or source["condition_cell"].nunique() != EXPECTED_CELLS:
        raise BatteryError("C062 manifest coverage changed")
    manifest = source.copy()
    manifest["primary_condition"] = (~manifest["condition_cell"].isin(DEVELOPMENT_CONDITIONS)).astype(int)
    if manifest.loc[manifest["primary_condition"] == 1, "condition_cell"].nunique() != PRIMARY_CELLS:
        raise BatteryError("primary condition-cell coverage changed")
    manifest["analysis_crop"] = "detector_blind_fixed_center_896x1344_within_upper40pct"
    return manifest


def freeze() -> dict[str, Any]:
    if FREEZE.exists() or (OUTPUT / "progress").exists():
        raise BatteryError("freeze or outputs already exist")
    manifest = build_manifest()
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8")
    bindings = {
        "script": SCRIPT,
        "c062_manifest": C062_MANIFEST,
        "manifest": MANIFEST,
        "model_code": MODEL_CODE,
        "inference_code": INFERENCE_CODE,
        "task_off_checkpoint": TASK_OFF_CHECKPOINT,
        "task_on_checkpoint": TASK_ON_CHECKPOINT,
        "detector_code": DETECTOR_CODE,
        "detector_checkpoint": DETECTOR_CHECKPOINT,
        "detector_protocol": DETECTOR_PROTOCOL,
        "adapter_code": ADAPTER_CODE,
        "gate_table": GATE_TABLE,
        "independent_verifier": INDEPENDENT_VERIFIER,
    }
    payload = {
        "schema_version": SCHEMA,
        "state": "FROZEN_BEFORE_FULL737_BATTERY_OUTPUTS",
        "created_utc": utc_now(),
        "cohort": {
            "frames": EXPECTED_FRAMES,
            "condition_cells": EXPECTED_CELLS,
            "primary_condition_cells": PRIMARY_CELLS,
            "selection": "all frames; detector-blind and content-blind",
            "analysis_region": (
                "fixed 896x1344 crop centred horizontally and vertically within the upper "
                "floor(0.40*height) lining band; no detector-dependent placement"
            ),
            "independent_unit": "acquisition condition cell",
            "frame_role": "repeated paired observation within condition cell",
        },
        "controlled_degradation": {
            "gaussian_sigma_horizontal_px": BLUR_H,
            "gaussian_sigma_vertical_px": BLUR_V,
            "gaussian_noise_sd_dn": NOISE_DN,
            "noise_seed": "first 64 bits of source image SHA-256 XOR 20261024",
            "source_interpretation": "operational source reference, not pristine physical truth",
        },
        "methods": list(METHODS),
        "primary_merit": PRIMARY_MERIT,
        "method_parameters": {
            "oracle_psf": [BLUR_V, BLUR_H],
            "wrong_psf_sigma_px": [1.5, 1.5],
            "wiener_K": 0.01,
            "richardson_lucy_iterations": 20,
            "tv_weight": 0.08,
            "nl_means": {"patch_size": 5, "patch_distance": 6, "h": 0.05, "fast_mode": True},
            "unsharp": {"sigma_px": 1.5, "amount": 1.5},
            "classical_tiling": {"tile": CLASSICAL_TILE, "overlap": CLASSICAL_OVERLAP},
            "deep_seed": 1,
            "safe_adapter": {"alpha": -0.75, "sigma_px": 0.75, "cap_dn": 8.0},
        },
        "detector": {
            "architecture": "ResNet34-UNet",
            "role": "external four-tunnel-trained response probe absent from MERIT training and loss",
            "threshold": THRESHOLD,
        },
        "statistics": {
            "primary_metric": "mean_probability absolute source-response error",
            "secondary_metrics": ["maximum_probability", "area_fraction_at_0.55"],
            "within_cell": "median across all frames",
            "method_effect": "method absolute error minus degraded absolute error; negative favours method",
            "recovery_percent": "100*(mean degraded error - mean method error)/mean degraded error",
            "merit_dominance": "baseline error minus MERIT error; positive favours MERIT",
            "bootstrap": {"count": BOOTSTRAP_COUNT, "seed": RNG_SEED, "unit": "condition cell"},
            "paired_sign_flip": {"count": PERMUTATION_COUNT, "seed": RNG_SEED + 100},
            "multiplicity": "Holm within the fixed method family for each comparison family",
            "image_level_analysis": "secondary sensitivity only; frames are not asserted independent",
        },
        "claim_boundary": (
            "This is a controlled detector-response restoration comparison against an operational source "
            "reference. It is not human-mask detection accuracy or proof that 737 frames are independent."
        ),
        "bindings": {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in bindings.items()
        },
    }
    write_json_atomic(FREEZE, payload)
    write_text_atomic(FREEZE.with_suffix(".json.sha256"), f"{sha256_file(FREEZE)}  {FREEZE.name}\n")
    return payload


def load_frozen() -> tuple[dict[str, Any], pd.DataFrame, str]:
    if not FREEZE.is_file() or not MANIFEST.is_file():
        raise BatteryError("execution freeze is missing")
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise BatteryError("freeze schema mismatch")
    for name, record in payload["bindings"].items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise BatteryError(f"frozen binding mismatch: {name}")
    manifest = pd.read_csv(MANIFEST, dtype={"frame": str})
    if len(manifest) != EXPECTED_FRAMES or manifest["condition_cell"].nunique() != EXPECTED_CELLS:
        raise BatteryError("manifest coverage mismatch")
    return payload, manifest, sha256_file(FREEZE)


def gaussian_psf(sigma_h: float, sigma_v: float) -> np.ndarray:
    width = int(2 * np.ceil(3 * sigma_h) + 1)
    height = int(2 * np.ceil(3 * sigma_v) + 1)
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    kernel = np.outer(np.exp(-(y**2) / (2 * sigma_v**2)), np.exp(-(x**2) / (2 * sigma_h**2)))
    return kernel / kernel.sum()


TRUE_PSF = gaussian_psf(BLUR_H, BLUR_V)
WRONG_PSF = gaussian_psf(1.5, 1.5)


def wiener_filter(image: np.ndarray, psf: np.ndarray, regularization: float = 0.01) -> np.ndarray:
    image64 = image.astype(np.float64, copy=False)
    pad = np.zeros_like(image64)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    transfer = np.fft.fft2(pad)
    restored = np.fft.ifft2(
        np.fft.fft2(image64) * np.conj(transfer) / (np.abs(transfer) ** 2 + regularization)
    ).real
    return restored.astype(np.float32)


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(max(height, 3))[:height]
    wx = np.hanning(max(width, 3))[:width]
    return np.maximum(np.outer(wy, wx).astype(np.float32), np.float32(0.05))


def tiled_classical(image_u8: np.ndarray, operation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    height, width = image_u8.shape
    accumulator = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for y0 in tile_starts(height, CLASSICAL_TILE, CLASSICAL_OVERLAP):
        for x0 in tile_starts(width, CLASSICAL_TILE, CLASSICAL_OVERLAP):
            tile = image_u8[y0:y0 + CLASSICAL_TILE, x0:x0 + CLASSICAL_TILE]
            restored = operation(tile)
            if restored.shape != tile.shape or not np.isfinite(restored).all():
                raise BatteryError("classical method returned an invalid tile")
            window = blend_window(*tile.shape)
            accumulator[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] += restored.astype(np.float32) * window
            weights[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] += window
    return np.clip(np.rint(accumulator / np.maximum(weights, 1e-8)), 0, 255).astype(np.uint8)


def restore_classical(method: str, degraded: np.ndarray) -> np.ndarray:
    normalized = lambda tile: tile.astype(np.float32) / np.float32(255.0)
    if method == "wiener_oracle":
        return tiled_classical(degraded, lambda tile: wiener_filter(tile, TRUE_PSF))
    if method == "richardson_lucy_oracle":
        return tiled_classical(
            degraded,
            lambda tile: richardson_lucy(normalized(tile), TRUE_PSF, num_iter=20, clip=True) * 255.0,
        )
    if method == "unsupervised_wiener_oracle":
        return tiled_classical(
            degraded,
            lambda tile: np.clip(unsupervised_wiener(normalized(tile), TRUE_PSF)[0], 0, 1) * 255.0,
        )
    if method == "wiener_wrong_psf":
        return tiled_classical(degraded, lambda tile: wiener_filter(tile, WRONG_PSF))
    if method == "tv_denoise":
        return tiled_classical(
            degraded,
            lambda tile: denoise_tv_chambolle(normalized(tile), weight=0.08, channel_axis=None) * 255.0,
        )
    if method == "nl_means":
        return tiled_classical(
            degraded,
            lambda tile: denoise_nl_means(
                normalized(tile), patch_size=5, patch_distance=6, h=0.05,
                fast_mode=True, channel_axis=None,
            ) * 255.0,
        )
    if method == "unsharp":
        image = degraded.astype(np.float32)
        smooth = gaussian_filter(image, sigma=1.5, mode="reflect")
        return np.clip(np.rint(image + 1.5 * (image - smooth)), 0, 255).astype(np.uint8)
    raise BatteryError(f"unknown classical method: {method}")


def load_models(device: torch.device):
    # c006_operational_inference imports this historical module name directly.
    model_module = load_module("c006_model", MODEL_CODE)
    inference_module = load_module("c063_inference", INFERENCE_CODE)
    detector_module = load_module("c063_detector", DETECTOR_CODE)
    adapter_module = load_module("c063_adapter", ADAPTER_CODE)
    models = {}
    with torch.serialization.safe_globals([TorchVersion]):
        for name, path in (("nafnet_fidelity", TASK_OFF_CHECKPOINT), ("merit", TASK_ON_CHECKPOINT)):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = model_module.NAFNetUNet(width=32)
            model.load_state_dict(checkpoint["state"], strict=True)
            models[name] = model.to(device).eval()
    detector_checkpoint = torch.load(DETECTOR_CHECKPOINT, map_location="cpu", weights_only=True)
    detector = detector_module.ResNet34UNet()
    detector.load_state_dict(detector_checkpoint["model_state"], strict=True)
    detector = detector.to(device).eval()
    candidate = adapter_module.SafeAdapterCandidate(
        route_id="c063_active_am0.75",
        alpha=-0.75,
        sigma=0.75,
        cap_dn=8.0,
        maximum_headroom_limited_fraction=0.01,
    )
    gate = pd.read_csv(GATE_TABLE).set_index(["speed_kmh", "distance_m", "iso"])
    return models, detector, inference_module, adapter_module, candidate, gate


def response_metrics(gray: np.ndarray, detector, inference_module, device: torch.device) -> dict[str, float]:
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    probability = inference_module.tiled_detect_rgb(
        rgb,
        detector,
        device,
        tile_size=512,
        batch_size=64,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    result = {
        "mean_probability": float(probability.mean(dtype=np.float64)),
        "maximum_probability": float(probability.max()),
        "area_fraction_at_0.55": float(np.mean(probability >= THRESHOLD)),
    }
    del rgb, probability
    return result


def deep_restore(gray: np.ndarray, model, inference_module, device: torch.device) -> np.ndarray:
    restored = inference_module.tiled_restore_gray(
        gray,
        model,
        device,
        tile_size=448,
        overlap=112,
        batch_size=64,
        blend_floor=0.05,
        amp_dtype="bfloat16",
    )
    return np.clip(np.rint(restored * np.float32(255.0)), 0, 255).astype(np.uint8)


def progress_path(ordinal: int, condition: str, frame: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in frame)
    return OUTPUT / "progress" / f"{ordinal:04d}_{condition}_{safe}.json"


def deterministic_degrade(source: np.ndarray, image_sha256: str) -> tuple[np.ndarray, int]:
    seed = (int(image_sha256[:16], 16) ^ RNG_SEED) & ((1 << 63) - 1)
    rng = np.random.default_rng(seed)
    blurred = gaussian_filter(source.astype(np.float32), sigma=(BLUR_V, BLUR_H), mode="reflect")
    degraded = blurred + rng.normal(0.0, NOISE_DN, size=source.shape).astype(np.float32)
    return np.clip(np.rint(degraded), 0, 255).astype(np.uint8), seed


def smoke() -> dict[str, Any]:
    """Exercise one real frame without saving or printing scientific endpoint values."""
    if not torch.cuda.is_available():
        raise BatteryError("CUDA is required")
    device = torch.device("cuda")
    models, detector, inference_module, adapter_module, candidate, _ = load_models(device)
    row = build_manifest().sort_values("ordinal").iloc[0]
    bgr = read_bgr(Path(row["image_path"]))
    crop_rows = int(np.floor(bgr.shape[0] * TOP_FRACTION))
    y0 = (crop_rows - CROP_HEIGHT) // 2
    x0 = (bgr.shape[1] - CROP_WIDTH) // 2
    source = cv2.cvtColor(
        bgr[y0:y0 + CROP_HEIGHT, x0:x0 + CROP_WIDTH], cv2.COLOR_BGR2GRAY
    )
    degraded, _ = deterministic_degrade(source, str(row["image_sha256"]))
    timings: dict[str, float] = {}
    for label, image in (("source_detector", source), ("degraded_detector", degraded)):
        started = time.perf_counter()
        result = response_metrics(image, detector, inference_module, device)
        if not all(np.isfinite(list(result.values()))):
            raise BatteryError(f"non-finite smoke metric: {label}")
        timings[label] = time.perf_counter() - started
    for method in METHODS[1:8]:
        started = time.perf_counter()
        restored = restore_classical(method, degraded)
        result = response_metrics(restored, detector, inference_module, device)
        if restored.shape != source.shape or not all(np.isfinite(list(result.values()))):
            raise BatteryError(f"invalid smoke result: {method}")
        timings[method] = time.perf_counter() - started
    for method, model_name in (
        ("nafnet_fidelity", "nafnet_fidelity"),
        ("merit_core_adapter", "merit"),
    ):
        started = time.perf_counter()
        restored = deep_restore(degraded, models[model_name], inference_module, device)
        if method == "merit_core_adapter":
            restored, _ = adapter_module.apply_safe_adapter(degraded, restored, candidate)
        result = response_metrics(restored, detector, inference_module, device)
        if restored.shape != source.shape or not all(np.isfinite(list(result.values()))):
            raise BatteryError(f"invalid smoke result: {method}")
        timings[method] = time.perf_counter() - started
    return {
        "schema_version": SCHEMA,
        "state": "SMOKE_PASS",
        "frame_shape": list(source.shape),
        "scientific_values_emitted": False,
        "timings_seconds": timings,
        "total_seconds": float(sum(timings.values())),
    }


def run() -> dict[str, Any]:
    frozen, manifest, freeze_hash = load_frozen()
    if not torch.cuda.is_available():
        raise BatteryError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    models, detector, inference_module, adapter_module, candidate, gate = load_models(device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = 0
    for row in manifest.itertuples(index=False):
        target = progress_path(int(row.ordinal), str(row.condition_cell), str(row.frame))
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("execution_freeze_sha256") != freeze_hash:
                raise BatteryError(f"stale progress: {target}")
            completed += 1
            continue
        started = time.perf_counter()
        image_path = Path(row.image_path)
        if sha256_file(image_path) != str(row.image_sha256):
            raise BatteryError(f"input hash mismatch: {image_path}")
        bgr = read_bgr(image_path)
        crop_rows = int(np.floor(bgr.shape[0] * TOP_FRACTION))
        if crop_rows < CROP_HEIGHT or bgr.shape[1] < CROP_WIDTH:
            raise BatteryError(f"image is smaller than the fixed analysis crop: {image_path}")
        crop_y0 = (crop_rows - CROP_HEIGHT) // 2
        crop_x0 = (bgr.shape[1] - CROP_WIDTH) // 2
        source = cv2.cvtColor(
            bgr[crop_y0:crop_y0 + CROP_HEIGHT, crop_x0:crop_x0 + CROP_WIDTH],
            cv2.COLOR_BGR2GRAY,
        )
        degraded, degradation_seed = deterministic_degrade(source, str(row.image_sha256))
        metrics = {
            "source": response_metrics(source, detector, inference_module, device),
            "degraded": response_metrics(degraded, detector, inference_module, device),
        }
        output_hashes = {"source": sha256_array(source), "degraded": sha256_array(degraded)}
        method_seconds: dict[str, float] = {}
        for method in METHODS[1:8]:
            method_started = time.perf_counter()
            restored = restore_classical(method, degraded)
            metrics[method] = response_metrics(restored, detector, inference_module, device)
            output_hashes[method] = sha256_array(restored)
            method_seconds[method] = time.perf_counter() - method_started
            del restored
        method_started = time.perf_counter()
        fidelity = deep_restore(degraded, models["nafnet_fidelity"], inference_module, device)
        metrics["nafnet_fidelity"] = response_metrics(fidelity, detector, inference_module, device)
        output_hashes["nafnet_fidelity"] = sha256_array(fidelity)
        method_seconds["nafnet_fidelity"] = time.perf_counter() - method_started
        del fidelity
        method_started = time.perf_counter()
        merit_raw = deep_restore(degraded, models["merit"], inference_module, device)
        merit_active, diagnostics = adapter_module.apply_safe_adapter(degraded, merit_raw, candidate)
        metrics["merit_core_adapter"] = response_metrics(
            merit_active, detector, inference_module, device
        )
        output_hashes["merit_core_adapter"] = sha256_array(merit_active)
        method_seconds["merit_core_adapter"] = time.perf_counter() - method_started
        gate_key = (int(row.speed_kmh), float(row.distance_m), int(row.iso))
        if gate_key not in gate.index:
            raise BatteryError(f"missing route gate: {gate_key}")
        route_active = bool(gate.loc[gate_key, "route_active"])
        merit_routed = merit_active if route_active else degraded.copy()
        metrics["merit_routed_v3i"] = response_metrics(
            merit_routed, detector, inference_module, device
        ) if route_active else dict(metrics["degraded"])
        output_hashes["merit_routed_v3i"] = sha256_array(merit_routed)
        payload = {
            "schema_version": SCHEMA,
            "execution_freeze_sha256": freeze_hash,
            "ordinal": int(row.ordinal),
            "condition_cell": str(row.condition_cell),
            "frame": str(row.frame),
            "image_sha256": str(row.image_sha256),
            "speed_kmh": int(row.speed_kmh),
            "distance_m": float(row.distance_m),
            "iso": int(row.iso),
            "primary_condition": bool(row.primary_condition),
            "crop_rows": crop_rows,
            "crop_y0": crop_y0,
            "crop_x0": crop_x0,
            "crop_height": CROP_HEIGHT,
            "crop_width": CROP_WIDTH,
            "degradation_seed": degradation_seed,
            "route_active": route_active,
            "metrics": metrics,
            "output_hashes": output_hashes,
            "adapter_diagnostics": diagnostics.to_dict(),
            "method_seconds": method_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json_atomic(target, payload)
        completed += 1
        write_json_atomic(
            STATUS,
            {
                "schema_version": SCHEMA,
                "state": "RUNNING",
                "updated_utc": utc_now(),
                "completed_frames": completed,
                "expected_frames": EXPECTED_FRAMES,
                "active_frame": f"{row.condition_cell}/{row.frame}",
                "execution_freeze_sha256": freeze_hash,
                "claim_boundary": frozen["claim_boundary"],
            },
        )
        del bgr, source, degraded, merit_raw, merit_active, merit_routed
        torch.cuda.empty_cache()
    result = {
        "schema_version": SCHEMA,
        "state": "INFERENCE_COMPLETE",
        "updated_utc": utc_now(),
        "completed_frames": completed,
        "expected_frames": EXPECTED_FRAMES,
        "execution_freeze_sha256": freeze_hash,
    }
    write_json_atomic(STATUS, result)
    return result


def bootstrap_method(
    degraded_error: np.ndarray,
    method_error: np.ndarray,
    seed: int,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    count = len(degraded_error)
    indices = rng.integers(0, count, size=(BOOTSTRAP_COUNT, count))
    degraded_means = degraded_error[indices].mean(axis=1)
    method_means = method_error[indices].mean(axis=1)
    delta = method_means - degraded_means
    recovery = 100.0 * (degraded_means - method_means) / np.maximum(degraded_means, 1e-15)
    return (
        [float(value) for value in np.percentile(delta, [2.5, 97.5])],
        [float(value) for value in np.percentile(recovery, [2.5, 97.5])],
    )


def bootstrap_mean(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_COUNT, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def sign_flip_pvalue(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    batch = 10_000
    for start in range(0, PERMUTATION_COUNT, batch):
        count = min(batch, PERMUTATION_COUNT - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed))
    return float((exceed + 1) / (PERMUTATION_COUNT + 1))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (total - rank) * pvalues[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def aggregate() -> dict[str, Any]:
    frozen, manifest, freeze_hash = load_frozen()
    records: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        path = progress_path(int(row.ordinal), str(row.condition_cell), str(row.frame))
        if not path.is_file():
            raise BatteryError(f"missing progress: {path}")
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("execution_freeze_sha256") != freeze_hash:
            raise BatteryError(f"progress freeze mismatch: {path}")
        base = {
            "ordinal": item["ordinal"],
            "condition_cell": item["condition_cell"],
            "frame": item["frame"],
            "image_sha256": item["image_sha256"],
            "speed_kmh": item["speed_kmh"],
            "distance_m": item["distance_m"],
            "iso": item["iso"],
            "primary_condition": item["primary_condition"],
            "route_active": item["route_active"],
            "elapsed_seconds": item["elapsed_seconds"],
        }
        for metric in METRICS:
            source_value = float(item["metrics"]["source"][metric])
            degraded_value = float(item["metrics"]["degraded"][metric])
            base[f"{metric}_source"] = source_value
            base[f"{metric}_degraded"] = degraded_value
            base[f"{metric}_degraded_abs_source_error"] = abs(degraded_value - source_value)
            for method in METHODS[1:]:
                method_value = float(item["metrics"][method][metric])
                base[f"{metric}_{method}"] = method_value
                base[f"{metric}_{method}_abs_source_error"] = abs(method_value - source_value)
        records.append(base)
    frame = pd.DataFrame(records).sort_values("ordinal").reset_index(drop=True)
    group_keys = [
        "condition_cell", "speed_kmh", "distance_m", "iso", "primary_condition", "route_active"
    ]
    error_columns = [column for column in frame.columns if column.endswith("_abs_source_error")]
    condition = frame.groupby(group_keys, as_index=False)[error_columns].median()
    condition["frames"] = frame.groupby(group_keys)["frame"].count().to_numpy()
    primary = condition[condition["primary_condition"] == True].copy()  # noqa: E712
    if len(primary) != PRIMARY_CELLS:
        raise BatteryError("primary-cell count changed")
    summaries: list[dict[str, Any]] = []
    dominance: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRICS):
        degraded = primary[f"{metric}_degraded_abs_source_error"].to_numpy(dtype=float)
        raw_p: dict[str, float] = {}
        raw_dom_p: dict[str, float] = {}
        method_cache: dict[str, np.ndarray] = {}
        for method_index, method in enumerate(METHODS[1:]):
            values = primary[f"{metric}_{method}_abs_source_error"].to_numpy(dtype=float)
            method_cache[method] = values
            delta = values - degraded
            raw_p[method] = sign_flip_pvalue(
                delta, RNG_SEED + 1000 * metric_index + method_index
            )
        adjusted = holm_adjust(raw_p)
        merit_error = method_cache[PRIMARY_MERIT]
        for method_index, method in enumerate(METHODS[1:]):
            values = method_cache[method]
            delta = values - degraded
            delta_ci, recovery_ci = bootstrap_method(
                degraded, values, RNG_SEED + 10_000 * metric_index + method_index
            )
            tolerance = 1e-15
            summaries.append(
                {
                    "metric": metric,
                    "method": method,
                    "condition_cells": len(values),
                    "mean_degraded_abs_source_error": float(degraded.mean()),
                    "mean_method_abs_source_error": float(values.mean()),
                    "mean_delta_error": float(delta.mean()),
                    "delta_error_95pct_ci": delta_ci,
                    "recovery_percent": float(
                        100.0 * (degraded.mean() - values.mean()) / max(degraded.mean(), 1e-15)
                    ),
                    "recovery_percent_95pct_ci": recovery_ci,
                    "improved_cells": int(np.sum(delta < -tolerance)),
                    "tied_cells": int(np.sum(np.abs(delta) <= tolerance)),
                    "worsened_cells": int(np.sum(delta > tolerance)),
                    "paired_sign_flip_p_two_sided": raw_p[method],
                    "holm_adjusted_p": adjusted[method],
                }
            )
            if method != PRIMARY_MERIT:
                advantage = values - merit_error
                raw_dom_p[method] = sign_flip_pvalue(
                    advantage, RNG_SEED + 50_000 + 1000 * metric_index + method_index
                )
        dom_adjusted = holm_adjust(raw_dom_p)
        for method_index, method in enumerate(METHODS[1:]):
            if method == PRIMARY_MERIT:
                continue
            advantage = method_cache[method] - merit_error
            tolerance = 1e-15
            dominance.append(
                {
                    "metric": metric,
                    "baseline": method,
                    "merit": PRIMARY_MERIT,
                    "mean_baseline_minus_merit_error": float(advantage.mean()),
                    "condition_bootstrap_95pct_ci": bootstrap_mean(
                        advantage, RNG_SEED + 60_000 + 1000 * metric_index + method_index
                    ),
                    "merit_better_cells": int(np.sum(advantage > tolerance)),
                    "tied_cells": int(np.sum(np.abs(advantage) <= tolerance)),
                    "merit_worse_cells": int(np.sum(advantage < -tolerance)),
                    "paired_sign_flip_p_two_sided": raw_dom_p[method],
                    "holm_adjusted_p": dom_adjusted[method],
                }
            )
    image_sensitivity: list[dict[str, Any]] = []
    primary_frames = frame[frame["primary_condition"] == True].copy()  # noqa: E712
    for method in METHODS[1:]:
        degraded = primary_frames["mean_probability_degraded_abs_source_error"].to_numpy(float)
        values = primary_frames[f"mean_probability_{method}_abs_source_error"].to_numpy(float)
        image_sensitivity.append(
            {
                "method": method,
                "frames": len(values),
                "recovery_percent": float(
                    100.0 * (degraded.mean() - values.mean()) / max(degraded.mean(), 1e-15)
                ),
                "status": "secondary_correlated_frame_sensitivity_not_independent_inference",
            }
        )
    checks = {
        "all_737_frames": len(frame) == EXPECTED_FRAMES,
        "all_50_conditions": len(condition) == EXPECTED_CELLS,
        "all_45_primary_conditions": len(primary) == PRIMARY_CELLS,
        "all_methods_present": len(summaries) == len(METRICS) * (len(METHODS) - 1),
        "all_values_finite": bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all()),
    }
    summary = {
        "schema_version": SCHEMA,
        "state": "COMPLETE_CLAIM_BOUNDED" if all(checks.values()) else "FAIL",
        "created_utc": utc_now(),
        "execution_freeze_sha256": freeze_hash,
        "counts": {
            "frames": len(frame),
            "condition_cells": len(condition),
            "primary_condition_cells": len(primary),
            "primary_frames": len(primary_frames),
        },
        "checks": checks,
        "claim_boundary": frozen["claim_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "frame_results.csv", index=False, encoding="utf-8")
    condition.to_csv(OUTPUT / "condition_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(summaries).to_csv(OUTPUT / "method_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(dominance).to_csv(OUTPUT / "merit_dominance.csv", index=False, encoding="utf-8")
    pd.DataFrame(image_sensitivity).to_csv(
        OUTPUT / "image_level_sensitivity.csv", index=False, encoding="utf-8"
    )
    write_json_atomic(OUTPUT / "summary.json", summary)
    write_json_atomic(STATUS, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "freeze", "run", "aggregate", "status"))
    arguments = parser.parse_args()
    if arguments.command == "smoke":
        result = smoke()
    elif arguments.command == "freeze":
        result = freeze()
    elif arguments.command == "run":
        result = run()
    elif arguments.command == "aggregate":
        result = aggregate()
    else:
        result = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {
            "state": "NOT_STARTED"
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
