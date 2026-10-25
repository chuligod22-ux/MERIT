"""Prepare the immutable C051 patch bank, task targets, and paired streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import c051_common as common
import c051_dataset as dataset

STEPS = 4000
BATCH = 4


def build_bank() -> tuple[np.ndarray, list[Path]]:
    crack_path = common.LEGACY_DATA / "crack_bank.npy"
    crack = np.load(crack_path).astype(np.float32)
    frames = [Path(value) for value in dataset.clean_reference_frames()][:60]
    wall = dataset.build_patch_bank(
        [str(value) for value in frames], psz=448, per_frame=4, seed=0
    ).astype(np.float32)
    bank = np.concatenate([crack, wall], axis=0)
    if bank.ndim != 3 or bank.shape[1:] != (448, 448):
        raise common.C051Error(f"Unexpected bank shape: {bank.shape}")
    return bank, frames


def build_stream(seed: int, bank_size: int) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(202610510 + seed)
    return {
        "indices": rng.randint(0, bank_size, size=(STEPS, BATCH), dtype=np.int32),
        "sigma_h": rng.uniform(0.2, dataset.BLUR_MAX, size=(STEPS, BATCH)).astype(
            np.float32
        ),
        "sigma_v": rng.uniform(0.2, dataset.BLUR_MAX * 0.8, size=(STEPS, BATCH)).astype(
            np.float32
        ),
        "noise_std": rng.uniform(0.0, dataset.NOISE_MAX, size=(STEPS, BATCH)).astype(
            np.float32
        ),
        "noise_seed": rng.randint(0, 2**31 - 1, size=(STEPS, BATCH), dtype=np.int64),
    }


def task_targets(bank: np.ndarray, batch_size: int) -> tuple[np.ndarray, Path]:
    segmenter_dir = common.NRIQA / "03_src" / "crack_seg"
    sys.path.insert(0, str(segmenter_dir))
    from utils import load_unet_vgg16

    checkpoint = segmenter_dir / "models" / "model_unet_vgg16_best.pt"
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise common.C051Error("CUDA is required to prepare task targets")
    model = load_unet_vgg16(str(checkpoint)).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    result = np.empty(bank.shape, dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(bank), batch_size):
            gray = torch.from_numpy(bank[start : start + batch_size] / 255.0).to(
                device=device, dtype=torch.float32
            )[:, None]
            probability = torch.sigmoid(model((gray.repeat(1, 3, 1, 1) - mean) / std))[
                :, 0
            ]
            result[start : start + len(gray)] = (
                probability.cpu().numpy().astype(np.float16)
            )
    del model
    torch.cuda.empty_cache()
    return result, checkpoint


def execute(batch_size: int) -> dict[str, object]:
    common.ASSETS.mkdir(parents=True, exist_ok=True)
    bank, frames = build_bank()
    bank_path = common.ASSETS / "training_bank.npy"
    np.save(bank_path, bank)
    targets, segmenter_checkpoint = task_targets(bank, batch_size)
    target_path = common.ASSETS / "task_targets.npy"
    np.save(target_path, targets)
    streams: dict[str, dict[str, object]] = {}
    stream_dir = common.ASSETS / "streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    for seed in common.SEEDS:
        path = stream_dir / f"seed_{seed}.npz"
        np.savez_compressed(path, **build_stream(seed, len(bank)))
        streams[str(seed)] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }
    frame_manifest = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }
        for path in frames
    ]
    payload: dict[str, object] = {
        "schema_version": "MERIT-C051-ASSETS-v1",
        "status": "PASS",
        "bank": {
            "path": str(bank_path.resolve()),
            "shape": list(bank.shape),
            "bytes": bank_path.stat().st_size,
            "sha256": common.sha256_file(bank_path),
            "crack_source": {
                "path": str((common.LEGACY_DATA / "crack_bank.npy").resolve()),
                "sha256": common.sha256_file(common.LEGACY_DATA / "crack_bank.npy"),
            },
            "wall_source_frames": frame_manifest,
        },
        "task_targets": {
            "path": str(target_path.resolve()),
            "shape": list(targets.shape),
            "bytes": target_path.stat().st_size,
            "sha256": common.sha256_file(target_path),
            "segmenter_checkpoint": str(segmenter_checkpoint.resolve()),
            "segmenter_checkpoint_sha256": common.sha256_file(segmenter_checkpoint),
        },
        "streams": streams,
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_atomic(common.ASSETS / "assets_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(execute(args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
