"""Apply the development-selected MERIT v2.2 model to one image."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c010_pilot as pilot
from merit_v2_model import CalibratedPromptBlend


PROTOCOL = pilot.C010 / "protocol" / "pilot_v3.json"


def read_gray(path: Path) -> tuple[np.ndarray, np.dtype[Any]]:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode input image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Expected 8/16-bit grayscale-compatible image, got {image.shape} {image.dtype}")
    source_dtype = image.dtype
    scale = np.float32(255.0 if source_dtype == np.uint8 else 65535.0)
    return image.astype(np.float32) / scale, source_dtype


def write_gray(path: Path, image: np.ndarray, dtype: np.dtype[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = 255.0 if dtype == np.uint8 else 65535.0
    encoded_image = np.rint(np.clip(image, 0.0, 1.0) * maximum).astype(dtype)
    extension = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(extension, encoded_image)
    if not ok:
        raise ValueError(f"Cannot encode output format: {extension}")
    encoded.tofile(path)


def axis_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def blend_window(tile_size: int) -> np.ndarray:
    one = np.hanning(tile_size).astype(np.float32)
    return np.maximum(np.outer(one, one), np.float32(0.05))


def pad_to_tile(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape
    pad_h = max(0, tile_size - height)
    pad_w = max(0, tile_size - width)
    if not pad_h and not pad_w:
        return image, (height, width)
    mode = "reflect" if min(height, width) > 1 else "edge"
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode=mode)
    return padded.astype(np.float32, copy=False), (height, width)


def tiled_restore(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    prompt_blur: float,
    prompt_noise: float,
    tile_size: int = 448,
    overlap: int = 112,
    batch_size: int = 8,
) -> np.ndarray:
    if image.ndim != 2 or image.dtype != np.float32:
        raise ValueError("tiled_restore expects a float32 [H,W] image")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    padded, original_shape = pad_to_tile(image, tile_size)
    coordinates = [
        (y, x)
        for y in axis_starts(padded.shape[0], tile_size, overlap)
        for x in axis_starts(padded.shape[1], tile_size, overlap)
    ]
    window = blend_window(tile_size)
    accumulator = np.zeros_like(padded, dtype=np.float32)
    weights = np.zeros_like(padded, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(coordinates), batch_size):
            chunk = coordinates[start : start + batch_size]
            tiles = np.stack(
                [padded[y : y + tile_size, x : x + tile_size] for y, x in chunk]
            ).astype(np.float32)
            inputs = pilot.common.prompt_tensor(tiles, prompt_blur, prompt_noise)
            with pilot.autocast(device):
                restored = model(torch.from_numpy(inputs).to(device))
            restored_np = restored[:, 0].float().cpu().numpy()
            if not np.all(np.isfinite(restored_np)):
                raise FloatingPointError("Non-finite MERIT v2 output")
            for tile, (y, x) in zip(restored_np, chunk):
                accumulator[y : y + tile_size, x : x + tile_size] += tile * window
                weights[y : y + tile_size, x : x + tile_size] += window
    output = accumulator / np.maximum(weights, np.float32(1e-8))
    return np.clip(output[: original_shape[0], : original_shape[1]], 0.0, 1.0)


def load_model(seed: int, device: torch.device):
    protocol = pilot.load_json(PROTOCOL)
    base, metadata = pilot.factory.build_official_adapter("nafnet")
    checkpoint_path = (
        pilot.C009 / "runs" / "wp1" / "nafnet" / f"base_seed{seed}" / "final.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["model"], strict=True)
    calibration = protocol["calibration"]
    model = CalibratedPromptBlend(
        base,
        [float(value) for value in calibration["severity_anchors"]],
        [float(value) for value in calibration["blend_anchors"]],
    ).to(device).eval()
    return model, {
        "base_seed": seed,
        "base_checkpoint_path": str(checkpoint_path.resolve()),
        "base_checkpoint_sha256": pilot.sha256_file(checkpoint_path),
        "base_model": metadata,
        "calibration": calibration,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigma-h", type=float, required=True)
    parser.add_argument("--sigma-v", type=float, required=True)
    parser.add_argument("--noise-std-8bit", type=float, required=True)
    parser.add_argument("--base-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--tile-size", type=int, default=448)
    parser.add_argument("--overlap", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-ood", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = (args.sigma_h, args.sigma_v, args.noise_std_8bit)
    if any(value < 0 for value in values):
        raise ValueError("Degradation values must be non-negative")
    in_training_envelope = (
        0.2 <= args.sigma_h <= 3.0
        and 0.2 <= args.sigma_v <= 3.0
        and 0.5 <= args.noise_std_8bit <= 3.0
    ) or values == (0.0, 0.0, 0.0)
    if not in_training_envelope and not args.allow_ood:
        raise ValueError(
            "Prompt is outside the development envelope; pass --allow-ood to run with an explicit OOD marker"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    image, dtype = read_gray(input_path)
    device = torch.device("cuda")
    model, provenance = load_model(args.base_seed, device)
    prompt_blur = (args.sigma_h + args.sigma_v) / 6.0
    prompt_noise = args.noise_std_8bit / 3.0
    severity = 0.5 * (prompt_blur + prompt_noise)
    with torch.inference_mode():
        blend = float(
            model._interpolate(torch.tensor([severity], device=device)).item()
        )
    started = time.perf_counter()
    restored = tiled_restore(
        image,
        model,
        device,
        prompt_blur,
        prompt_noise,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    write_gray(output_path, restored, dtype)
    metadata = {
        "schema_version": "MERIT-C010-V2-INFERENCE-v1",
        "status": "PASS",
        "scientific_status": "DEVELOPMENT_MODEL_NOT_PROSPECTIVELY_CONFIRMED",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_shape": list(image.shape),
        "input_dtype": str(dtype),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "prompt": {
            "sigma_h": args.sigma_h,
            "sigma_v": args.sigma_v,
            "noise_std_8bit": args.noise_std_8bit,
            "normalized_blur": prompt_blur,
            "normalized_noise": prompt_noise,
            "severity": severity,
            "selected_blend": blend,
            "in_development_envelope": in_training_envelope,
        },
        "tiling": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "batch_size": args.batch_size,
        },
        "provenance": provenance,
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": pilot.sha256_file(PROTOCOL),
        "runner_sha256": pilot.sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    pilot.write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
