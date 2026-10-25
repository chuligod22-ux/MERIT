"""Apply the internally heldout C012 MERIT v2.3 field profiles to one image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C012 = HERE.parent
MODIFIED = C012.parent
C010_SCRIPTS = MODIFIED / "c010_merit_v2_development" / "scripts"
C011_SCRIPTS = MODIFIED / "c011_field_validation" / "scripts"
for directory in (C010_SCRIPTS, C011_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import c010_infer_v3 as c010
import c011_common as common
import c012_profiles as profiles


def tiled_route(
    image: np.ndarray,
    base_model: torch.nn.Module,
    device: torch.device,
    *,
    task: str,
    detector_name: str | None,
    tile_size: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    padded, original_shape = c010.pad_to_tile(image, tile_size)
    coordinates = [(y, x) for y in c010.axis_starts(padded.shape[0], tile_size, overlap) for x in c010.axis_starts(padded.shape[1], tile_size, overlap)]
    window = c010.blend_window(tile_size)
    accumulator = np.zeros_like(padded, dtype=np.float32)
    weights = np.zeros_like(padded, dtype=np.float32)
    for start in range(0, len(coordinates), batch_size):
        chunk = coordinates[start : start + batch_size]
        tiles = np.stack([padded[y : y + tile_size, x : x + tile_size] for y, x in chunk]).astype(np.float32)
        zeros = np.zeros(len(tiles), dtype=np.float32)
        inputs = c010.pilot.common.prompt_tensor(tiles, zeros, zeros)
        tensor = torch.from_numpy(inputs).to(device)
        with c010.pilot.autocast(device):
            output = profiles.route(tensor, base_model, active=True, task=task, detector_name=detector_name)
        values = output[:, 0].float().cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("C012 inference produced non-finite output")
        for value, (y, x) in zip(values, chunk):
            accumulator[y : y + tile_size, x : x + tile_size] += value * window
            weights[y : y + tile_size, x : x + tile_size] += window
    height, width = original_shape
    return np.clip(accumulator[:height, :width] / weights[:height, :width], 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("identity", "crack_detection", "printed_width"), required=True)
    parser.add_argument("--detector", choices=tuple(profiles.DETECTOR_PROFILE))
    parser.add_argument("--base-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--tile-size", type=int, default=448)
    parser.add_argument("--overlap", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "crack_detection" and args.detector is None:
        raise ValueError("--detector is required for crack_detection")
    if args.task != "crack_detection" and args.detector is not None:
        raise ValueError("--detector is only valid for crack_detection")
    if not 0 <= args.overlap < args.tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    image, dtype = c010.read_gray(input_path)
    started = time.perf_counter()
    provenance: dict[str, Any] | None = None
    if args.task == "identity":
        output = image.copy()
        profile = "exact_identity"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for active C012 profiles")
        device = torch.device("cuda")
        model, provenance = c010.load_model(args.base_seed, device)
        output = tiled_route(
            image,
            model.base,
            device,
            task=args.task,
            detector_name=args.detector,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        profile = profiles.DETECTOR_PROFILE[args.detector] if args.task == "crack_detection" else profiles.WIDTH_PROFILE
    c010.write_gray(output_path, output, dtype)
    protocol = C012 / "protocol" / "holdout_freeze_v1.json"
    metadata = {
        "schema_version": "MERIT-C012-V23-INFERENCE-v1",
        "status": "PASS",
        "scientific_status": "INTERNAL_HOLDOUT_PASS_NOT_PROSPECTIVELY_EXTERNAL_CONFIRMED",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "input_sha256": common.sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": common.sha256_file(output_path),
        "input_shape": list(image.shape),
        "input_dtype": str(dtype),
        "task": args.task,
        "detector": args.detector,
        "profile": profile,
        "base_seed": args.base_seed,
        "tiling": {"tile_size": args.tile_size, "overlap": args.overlap, "batch_size": args.batch_size},
        "provenance": provenance,
        "holdout_protocol_path": str(protocol.resolve()),
        "holdout_protocol_sha256": common.sha256_file(protocol),
        "profile_code_sha256": common.sha256_file(HERE / "c012_profiles.py"),
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "Detector-conditioned crack-detection support and printed nominal apparent-width support within the sealed internal holdout. Natural-crack physical width remains undetermined.",
    }
    common.write_json_atomic(output_path.with_suffix(output_path.suffix + ".json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
