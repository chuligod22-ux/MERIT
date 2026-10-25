"""Local-surrogate GPU timing smoke for MERIT C009 preflight.

This is not a confirmatory training runner.  It benchmarks the three locally
available architecture candidates with the frozen C006 fidelity/task-loss
calculation so G13 has a measured lower-bound before official implementations
are vendored and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
REVISION_WORK_ROOT = HERE.parents[1]
C006_ROOT = REVISION_WORK_ROOT / "c006_execution"
TUNNEL_ROOT = Path(r"<LOCAL_PATH_REDACTED>")
LEGACY_SOURCE = TUNNEL_ROOT / "03_restoration" / "03_src"

sys.path.insert(0, str(C006_ROOT))
from c006_model import NAFNetUNet  # noqa: E402
from c006_train import (  # noqa: E402
    deepcrack_probability,
    fft_loss,
    grad_loss,
    load_deepcrack,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_backbones() -> Any:
    module_path = LEGACY_SOURCE / "model_backbones.py"
    spec = importlib.util.spec_from_file_location("c009_legacy_backbones", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(name: str, legacy: Any) -> torch.nn.Module:
    if name == "nafnet_local":
        return NAFNetUNet(in_channels=3, out_channels=1, width=32)
    if name == "restormer_lite_local":
        return legacy.RestormerLite(in_ch=3, out_ch=1, width=48)
    if name == "mamba_lite_local":
        return legacy.MambaLite(
            in_ch=3, out_ch=1, width=128, n_blocks=5, n_state=4, n_down=4
        )
    raise ValueError(name)


def amp_context() -> Any:
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def benchmark(
    name: str,
    arm: str,
    steps: int,
    warmup: int,
    batch_size: int,
    patch_size: int,
    segmenter: torch.nn.Module | None,
    legacy: Any,
) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(20260924)
    torch.cuda.manual_seed_all(20260924)
    model = make_model(name, legacy).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    degraded = torch.rand(batch_size, 1, patch_size, patch_size, device=device)
    prompt_blur = torch.full_like(degraded, 0.5)
    prompt_noise = torch.full_like(degraded, 0.5)
    inputs = torch.cat([degraded, prompt_blur, prompt_noise], dim=1)
    target = torch.rand_like(degraded)
    l1 = torch.nn.L1Loss()
    target_probability: torch.Tensor | None = None
    if arm == "full_package":
        if segmenter is None:
            raise RuntimeError("full_package requires the frozen DeepCrack model")
        with torch.no_grad():
            target_probability = deepcrack_probability(segmenter, target)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    losses: list[float] = []
    total_iterations = warmup + steps
    for iteration in range(total_iterations):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with amp_context():
            output = model(inputs)
        output_float = output.float()
        target_float = target.float()
        fidelity = (
            l1(output_float, target_float)
            + grad_loss(output_float, target_float)
            + 0.1 * fft_loss(output_float, target_float)
        )
        if arm == "base":
            loss = fidelity
        else:
            assert segmenter is not None and target_probability is not None
            task = l1(
                deepcrack_probability(segmenter, output_float), target_probability
            )
            loss = fidelity + 5.0 * task
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if iteration >= warmup:
            timings.append(elapsed)
            losses.append(float(loss.detach().cpu()))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    mean_step_seconds = sum(timings) / len(timings)
    result = {
        "model": name,
        "arm": arm,
        "status": "PASS",
        "batch_size": batch_size,
        "patch_size": patch_size,
        "warmup_steps": warmup,
        "timed_steps": steps,
        "parameters": parameter_count,
        "mean_step_seconds": mean_step_seconds,
        "min_step_seconds": min(timings),
        "max_step_seconds": max(timings),
        "projected_8000_step_hours": mean_step_seconds * 8000 / 3600,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "last_loss": losses[-1],
    }
    del model, optimizer, degraded, prompt_blur, prompt_noise, inputs, target
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=448)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "local_surrogate_smoke_v1.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with (C006_ROOT / "c006_protocol.json").open(encoding="utf-8") as handle:
        c006_protocol = json.load(handle)
    device = torch.device("cuda")
    segmenter = load_deepcrack(c006_protocol, device)
    legacy = load_legacy_backbones()

    results: list[dict[str, Any]] = []
    for model_name in (
        "nafnet_local",
        "restormer_lite_local",
        "mamba_lite_local",
    ):
        for arm in ("base", "full_package"):
            try:
                results.append(
                    benchmark(
                        model_name,
                        arm,
                        args.steps,
                        args.warmup,
                        args.batch_size,
                        args.patch_size,
                        segmenter,
                        legacy,
                    )
                )
            except torch.cuda.OutOfMemoryError as error:
                results.append(
                    {
                        "model": model_name,
                        "arm": arm,
                        "status": "OOM",
                        "error": str(error),
                    }
                )
                torch.cuda.empty_cache()

    source_files = {
        "smoke": Path(__file__).resolve(),
        "c006_model": C006_ROOT / "c006_model.py",
        "c006_train": C006_ROOT / "c006_train.py",
        "local_backbones": LEGACY_SOURCE / "model_backbones.py",
    }
    output = {
        "schema_version": "MERIT-C009-LOCAL-SURROGATE-SMOKE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_use": False,
        "warning": (
            "RestormerLite and MambaLite are local reimplementations, not the "
            "reviewer-requested official upstream implementations. Timing is a "
            "lower-bound preflight only and cannot close G11 or final G13."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
        },
        "source_sha256": {
            label: sha256_file(path) for label, path in source_files.items()
        },
        "settings": {
            "steps": args.steps,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "amp": "bfloat16",
            "base_loss": "L1 + gradient + 0.1*FFT",
            "full_package_loss": "L1 + gradient + 0.1*FFT + 5.0*DeepCrack-L1",
            "input_channels": 3,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
