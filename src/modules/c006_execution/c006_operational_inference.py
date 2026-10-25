#!/usr/bin/env python3
"""Frozen full-frame inference adapter for the MERIT C006/G09 gate.

This module never selects a model from outcomes.  It verifies every configured
asset by SHA-256, applies the predeclared tiling rule, and emits either a
restored PNG or a full-resolution D2 probability map.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.torch_version import TorchVersion

from c006_model import NAFNetUNet


SCHEMA = "MERIT-C006-G09-E2E-v1"


class ProtocolError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ProtocolError("Unexpected operational protocol schema")
    return payload


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ProtocolError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ProtocolError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError(f"Could not decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ProtocolError(f"Could not encode PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    if length < tile_size:
        raise ValueError("Input axis must be at least one tile")
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def positive_hann_window(tile_size: int, floor: float) -> np.ndarray:
    if tile_size < 2 or not 0.0 < floor < 1.0:
        raise ValueError("Invalid blend-window parameters")
    one_dimensional = np.maximum(np.hanning(tile_size).astype(np.float32), floor)
    return np.outer(one_dimensional, one_dimensional).astype(np.float32)


def _chunks(values: list[tuple[int, int]], size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


@torch.inference_mode()
def tiled_restore_gray(
    gray: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
    blend_floor: float,
    amp_dtype: str,
) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Restoration input must be a uint8 grayscale image")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    height, width = gray.shape
    positions = [
        (y, x)
        for y in axis_starts(height, tile_size, overlap)
        for x in axis_starts(width, tile_size, overlap)
    ]
    image = gray.astype(np.float32) / np.float32(255.0)
    accumulator = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    window = positive_hann_window(tile_size, blend_floor)
    model.eval()
    for chunk in _chunks(positions, batch_size):
        batch = np.zeros((len(chunk), 3, tile_size, tile_size), dtype=np.float32)
        for index, (y, x) in enumerate(chunk):
            batch[index, 0] = image[y : y + tile_size, x : x + tile_size]
        tensor = torch.from_numpy(batch).to(device)
        context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and amp_dtype == "bfloat16"
            else torch.amp.autocast(device_type=device.type, enabled=False)
        )
        with context:
            output = model(tensor)
        restored = output[:, 0].float().cpu().numpy()
        for tile, (y, x) in zip(restored, chunk):
            accumulator[y : y + tile_size, x : x + tile_size] += tile * window
            weights[y : y + tile_size, x : x + tile_size] += window
    if not np.all(weights > 0.0):
        raise ProtocolError("Restoration tiling left uncovered pixels")
    result = accumulator / weights
    if not np.isfinite(result).all():
        raise ProtocolError("Restoration produced non-finite pixels")
    return np.clip(result, 0.0, 1.0)


def _load_d2_class(code_path: Path):
    specification = importlib.util.spec_from_file_location(
        "merit_frozen_d2_detector", code_path
    )
    if specification is None or specification.loader is None:
        raise ProtocolError(f"Could not import frozen D2 code: {code_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.ResNet34UNet


def reflect_pad_to_multiple(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    pad_bottom = (-height) % tile_size
    pad_right = (-width) % tile_size
    padded = cv2.copyMakeBorder(
        image, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_REFLECT_101
    )
    return padded, pad_bottom, pad_right


@torch.inference_mode()
def tiled_detect_rgb(
    bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    tile_size: int,
    batch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Detector input must be a uint8 BGR image")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    original_height, original_width = bgr.shape[:2]
    padded, _, _ = reflect_pad_to_multiple(bgr, tile_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.0)
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    height, width = rgb.shape[:2]
    positions = [
        (y, x)
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]
    probability = np.empty((height, width), dtype=np.float32)
    model.eval()
    for chunk in _chunks(positions, batch_size):
        batch = np.stack(
            [
                np.moveaxis(
                    rgb[y : y + tile_size, x : x + tile_size], -1, 0
                )
                for y, x in chunk
            ]
        ).astype(np.float32, copy=False)
        logits = model(torch.from_numpy(batch).to(device))
        predicted = torch.sigmoid(logits[:, 0]).float().cpu().numpy()
        for tile, (y, x) in zip(predicted, chunk):
            probability[y : y + tile_size, x : x + tile_size] = tile
    probability = probability[:original_height, :original_width]
    if probability.shape != (original_height, original_width):
        raise ProtocolError("Detector output shape mismatch")
    if not np.isfinite(probability).all() or np.min(probability) < 0.0 or np.max(probability) > 1.0:
        raise ProtocolError("Detector output violates finite [0,1] contract")
    return probability


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ProtocolError("Operational E2E requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda")


def command_restore(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol)
    key = f"{args.arm}_seed{args.seed}"
    record = protocol["restoration"]["checkpoints"].get(key)
    if record is None:
        raise ProtocolError(f"Unfrozen restoration run: {key}")
    checkpoint_path = Path(record["path"])
    checkpoint_hash = require_hash(checkpoint_path, record["sha256"], key)
    require_hash(
        Path(protocol["restoration"]["model_code_path"]),
        protocol["restoration"]["model_code_sha256"],
        "C006 model code",
    )
    device = require_cuda()
    model = NAFNetUNet(width=int(protocol["restoration"]["model_width"]))
    # Locally generated, hash-bound C006 files contain only PyTorch's version
    # wrapper beyond the default weights-only allowlist.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state"], strict=True)
    model = model.to(device)
    image = read_bgr(args.input)
    expected = protocol["input_shape"]
    if image.shape != tuple(expected):
        raise ProtocolError(f"Input shape mismatch: {image.shape} != {tuple(expected)}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = protocol["restoration"]["tiling"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    restored = tiled_restore_gray(
        gray,
        model,
        device,
        tile_size=int(config["tile_size"]),
        overlap=int(config["overlap"]),
        batch_size=int(config["batch_size"]),
        blend_floor=float(config["positive_hann_floor"]),
        amp_dtype=protocol["restoration"]["amp_dtype"],
    )
    encoded = np.rint(restored * np.float32(255.0)).astype(np.uint8)
    write_png(args.output, cv2.cvtColor(encoded, cv2.COLOR_GRAY2BGR))
    return {
        "stage": "restore",
        "run": key,
        "checkpoint_sha256": checkpoint_hash,
        "output_shape": list(image.shape),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def command_detect(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol)
    config = protocol["detector"]
    code_path = Path(config["code_path"])
    require_hash(code_path, config["code_sha256"], "D2 detector code")
    require_hash(Path(config["protocol_path"]), config["protocol_sha256"], "D2 protocol")
    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_hash = require_hash(
        checkpoint_path, config["checkpoint_sha256"], "D2 checkpoint"
    )
    device = require_cuda()
    model_class = _load_d2_class(code_path)
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    image = read_bgr(args.input)
    expected = protocol["input_shape"]
    if image.shape != tuple(expected):
        raise ProtocolError(f"Input shape mismatch: {image.shape} != {tuple(expected)}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    probability = tiled_detect_rgb(
        image,
        model,
        device,
        tile_size=int(config["tiling"]["tile_size"]),
        batch_size=int(config["tiling"]["batch_size"]),
        mean=tuple(config["normalization"]["mean"]),
        std=tuple(config["normalization"]["std"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, probability, allow_pickle=False)
    return {
        "stage": "detect",
        "checkpoint_sha256": checkpoint_hash,
        "output_shape": list(probability.shape),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--arm", choices=("task_off", "task_on"), required=True)
    restore.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    restore.add_argument("--input", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    detect = commands.add_parser("detect")
    detect.add_argument("--input", type=Path, required=True)
    detect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = command_restore(args) if args.command == "restore" else command_detect(args)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
