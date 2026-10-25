"""Forward-only official-backbone timing for the G13 full-frame budget."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
MANIFEST = C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json"
PREFLIGHT = C009_ROOT / "protocol" / "g11_source_preflight_v3.json"
CONTRACT = C009_ROOT / "protocol" / "g11_training_contract_v2.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("nafnet", "restormer", "mprnet"), required=True
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, choices=(4, 8, 16, 32), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    smoke = load_module(
        f"g13_smoke_helper_{args.family}", HERE / "g11_official_smoke.py"
    )
    factory = load_module(
        f"g13_factory_{args.family}", HERE / "g11_official_model_factory.py"
    )
    smoke.set_determinism(0)
    degraded, _, metadata, stream = smoke.frozen_batch(CONTRACT, 0)
    model, factory_metadata = factory.build_official_adapter(
        args.family, MANIFEST, PREFLIGHT
    )
    model = model.cuda().eval()
    device = torch.device("cuda")
    records = []
    for arm in ("base", "merit_full"):
        arm_batch = stream.arm_inputs(degraded, metadata, arm)
        repeats = math.ceil(args.batch_size / int(arm_batch.shape[0]))
        arm_batch = np.concatenate([arm_batch] * repeats, axis=0)[: args.batch_size]
        inputs = torch.from_numpy(arm_batch).cuda()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        with torch.no_grad():
            for step in range(args.warmup_steps + args.timed_steps):
                started = time.perf_counter()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(inputs)
                torch.cuda.synchronize(device)
                if step >= args.warmup_steps:
                    timings.append(time.perf_counter() - started)
        if output.shape != (args.batch_size, 1, 448, 448) or not torch.all(
            torch.isfinite(output)
        ):
            raise RuntimeError(f"Unexpected inference output: {tuple(output.shape)}")
        records.append(
            {
                "arm": arm,
                "batch_size": args.batch_size,
                "warmup_steps": args.warmup_steps,
                "timed_steps": args.timed_steps,
                "mean_batch_seconds": float(np.mean(timings)),
                "min_batch_seconds": min(timings),
                "max_batch_seconds": max(timings),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
        )

    height, width = 5460, 8192
    tile_size, overlap, batch_size = 448, 112, args.batch_size
    tile_count = len(axis_starts(height, tile_size, overlap)) * len(
        axis_starts(width, tile_size, overlap)
    )
    batches_per_frame = math.ceil(tile_count / batch_size)
    conservative_batch_seconds = max(row["mean_batch_seconds"] for row in records)
    model_only_seconds_per_frame = batches_per_frame * conservative_batch_seconds
    six_checkpoints_974_frames_h = (
        model_only_seconds_per_frame * 6 * 974 / 3600.0
    )
    result = {
        "schema_version": "MERIT-C009-G13-OFFICIAL-INFERENCE-SMOKE-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_MODEL_ONLY_PROJECTION",
        "family": args.family,
        "factory": factory_metadata,
        "records": records,
        "full_frame_projection": {
            "height": height,
            "width": width,
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": tile_size - overlap,
            "tile_count": tile_count,
            "batch_size": batch_size,
            "batches_per_frame": batches_per_frame,
            "conservative_mean_batch_seconds": conservative_batch_seconds,
            "model_only_seconds_per_frame_per_checkpoint": model_only_seconds_per_frame,
            "six_checkpoints_times_974_frames_gpu_hours": six_checkpoints_974_frames_h,
        },
        "coverage_limitation": "Model forward only. Image decode, tile construction/blending, detector inference, metrics, artifact I/O, and process startup are excluded.",
        "manifest_sha256": sha256_file(MANIFEST),
        "preflight_sha256": sha256_file(PREFLIGHT),
        "contract_sha256": sha256_file(CONTRACT),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "full_training_allowed_by_this_artifact": False,
    }
    output_path = (
        args.output.resolve()
        if args.output
        else C009_ROOT
        / "smoke"
        / f"g13_{args.family}_inference_smoke_b{args.batch_size}_v2.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
