"""Official-backbone G11 base/full forward-backward smoke.

The runner is intentionally one-family-per-process.  It requires a PASS source
preflight, reuses the frozen paired D0 stream, checks base/full initial-state
and parameter equality, and measures bfloat16 training-step time and peak CUDA
memory.  It is not a full-training entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
REVISION_ROOT = C009_ROOT.parent
C006_ROOT = REVISION_ROOT / "c006_execution"
DEFAULT_MANIFEST = C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json"
DEFAULT_PREFLIGHT = C009_ROOT / "protocol" / "g11_source_preflight_v3.json"
DEFAULT_CONTRACT = C009_ROOT / "protocol" / "g11_training_contract_v2.json"


class SmokeError(RuntimeError):
    """Raised when an official smoke invariant fails."""


def load_sibling_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SmokeError(f"Cannot load local helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def amp_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def frozen_batch(
    contract_path: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any]:
    stream_module = load_sibling_module(
        "g11_training_stream_smoke", HERE / "g11_training_stream.py"
    )
    contract = stream_module.load_json(contract_path)
    bank, rows, _ = stream_module.verify_source_bank(contract)
    train_indices = np.array(
        [int(row["bank_index"]) for row in rows if row["split"] == "train"],
        dtype=np.int64,
    )
    generator = stream_module.latent_stream(
        bank,
        train_indices,
        seed,
        int(contract["training"]["batch_size"]),
        contract["stream"]["degradation"],
    )
    degraded, target, metadata = next(generator)
    return degraded, target, metadata, stream_module


def load_c006_loss_helpers():
    sys.path.insert(0, str(C006_ROOT))
    try:
        import c006_train
    finally:
        sys.path.pop(0)
    return c006_train


class _PreclipProfileWrapper(torch.nn.Module):
    """Expose the optimization path without the evaluation-only clamp."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.forward_preclip(inputs)


def profile_flops(model: torch.nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return {
            "status": "BLOCKED_DEPENDENCY",
            "dependency": "fvcore",
            "total_flops": None,
            "unsupported_ops": None,
        }
    was_training = model.training
    model.eval()
    try:
        analysis = FlopCountAnalysis(_PreclipProfileWrapper(model), sample)
        total = int(analysis.total())
        unsupported = {
            str(key): int(value)
            for key, value in analysis.unsupported_ops().items()
        }
        return {
            "status": "PASS" if not unsupported else "PARTIAL_UNSUPPORTED_OPS",
            "dependency": "fvcore",
            "total_flops": total,
            "unsupported_ops": unsupported,
        }
    except Exception as error:  # upstream/custom CUDA op coverage is audited here
        return {
            "status": "FAILED",
            "dependency": "fvcore",
            "total_flops": None,
            "unsupported_ops": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        model.train(was_training)


def benchmark_arm(
    family: str,
    arm: str,
    seed: int,
    warmup_steps: int,
    timed_steps: int,
    degraded: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, Any],
    stream_module: Any,
    factory_module: Any,
    loss_module: Any,
    manifest_path: Path,
    preflight_path: Path,
    contract_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_determinism(seed)
    model, factory_metadata = factory_module.build_official_adapter(
        family, manifest_path, preflight_path
    )
    initial_state_hash = state_dict_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    inputs = stream_module.arm_inputs(degraded, metadata, arm)
    input_tensor = torch.from_numpy(inputs).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    model = model.to(device).train()

    segmenter = None
    target_probability = None
    contract = stream_module.load_json(contract_path)
    if arm == "merit_full":
        c006_protocol = loss_module.load_json(C006_ROOT / "c006_protocol.json")
        expected = contract["training"]["deepcrack_checkpoint_sha256"]
        if c006_protocol["deepcrack_checkpoint"]["sha256"] != expected:
            raise SmokeError("DeepCrack hash differs between C006 and G11 contract")
        segmenter = loss_module.load_deepcrack(c006_protocol, device)
        with torch.no_grad():
            target_probability = loss_module.deepcrack_probability(
                segmenter, target_tensor
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["training"]["learning_rate"]),
        weight_decay=float(contract["training"]["weight_decay"]),
    )
    l1 = torch.nn.L1Loss()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    losses: list[float] = []
    for step in range(warmup_steps + timed_steps):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with amp_context(device):
            preclip_output = model.forward_preclip(input_tensor)
        output_float = preclip_output.float()
        target_float = target_tensor.float()
        l1_value = l1(output_float, target_float)
        gradient_value = loss_module.grad_loss(output_float, target_float)
        fft_value = loss_module.fft_loss(output_float, target_float)
        loss = l1_value + gradient_value + 0.1 * fft_value
        task_value = torch.zeros((), device=device, dtype=torch.float32)
        if arm == "merit_full":
            assert segmenter is not None and target_probability is not None
            restored_probability = loss_module.deepcrack_probability(
                segmenter, torch.clamp(output_float, 0.0, 1.0)
            )
            task_value = l1(restored_probability, target_probability)
            loss = loss + float(contract["training"]["task_lambda"]) * task_value
        if not torch.isfinite(loss):
            raise SmokeError(f"Non-finite {arm} loss at smoke step {step}")
        loss.backward()
        if not all(
            parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
            for parameter in model.parameters()
        ):
            raise SmokeError(f"Non-finite {arm} gradient at smoke step {step}")
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if step >= warmup_steps:
            timings.append(elapsed)
            losses.append(float(loss.detach().cpu()))

    with torch.no_grad(), amp_context(device):
        preclip = model.forward_preclip(input_tensor).float()
        clipped = torch.clamp(preclip, 0.0, 1.0)
    out_of_range = torch.logical_or(preclip < 0.0, preclip > 1.0)
    flops = profile_flops(model, input_tensor[:1])
    result = {
        "family": family,
        "arm": arm,
        "status": "PASS",
        "seed": seed,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "batch_size": int(input_tensor.shape[0]),
        "patch_size": int(input_tensor.shape[-1]),
        "initial_state_sha256": initial_state_hash,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "mean_step_seconds": float(np.mean(timings)),
        "min_step_seconds": min(timings),
        "max_step_seconds": max(timings),
        "projected_8000_step_hours_model_and_loss": float(np.mean(timings))
        * 8000
        / 3600,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "last_loss": losses[-1],
        "output_shape": list(clipped.shape),
        "output_all_finite": bool(torch.all(torch.isfinite(clipped)).item()),
        "preclip_out_of_range_fraction": float(out_of_range.float().mean().cpu()),
        "optimization_output_domain": "unclipped selected official output",
        "task_and_evaluation_output_domain": "clamped to [0,1]",
        "flops_448_single_image": flops,
        "factory": factory_metadata,
    }
    del optimizer, input_tensor, target_tensor, segmenter, target_probability, model
    torch.cuda.empty_cache()
    return result


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SmokeError("CUDA is required for official G11 smoke")
    manifest_path = args.manifest.resolve()
    preflight_path = args.preflight.resolve()
    contract_path = args.contract.resolve()
    factory_module = load_sibling_module(
        "g11_official_model_factory_smoke", HERE / "g11_official_model_factory.py"
    )
    loss_module = load_c006_loss_helpers()
    degraded, target, metadata, stream_module = frozen_batch(contract_path, args.seed)
    device = torch.device("cuda")
    results: list[dict[str, Any]] = []
    for arm in ("base", "merit_full"):
        result = benchmark_arm(
            args.family,
            arm,
            args.seed,
            args.warmup_steps,
            args.timed_steps,
            degraded,
            target,
            metadata,
            stream_module,
            factory_module,
            loss_module,
            manifest_path,
            preflight_path,
            contract_path,
            device,
        )
        results.append(result)
    if results[0]["initial_state_sha256"] != results[1]["initial_state_sha256"]:
        raise SmokeError("Base/full initial-state hashes differ")
    if results[0]["trainable_parameters"] != results[1]["trainable_parameters"]:
        raise SmokeError("Base/full trainable parameter counts differ")
    all_flops_complete = all(
        result["flops_448_single_image"]["status"] == "PASS" for result in results
    )
    output = {
        "schema_version": "MERIT-C009-G11-OFFICIAL-SMOKE-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_flops_complete else "PARTIAL_FLOPS_BLOCKED",
        "family": args.family,
        "confirmatory_training": False,
        "full_training_allowed_by_this_artifact": False,
        "paired_initial_state_equal": True,
        "paired_trainable_parameters_equal": True,
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_sha256": sha256_file(preflight_path),
        "contract_sha256": sha256_file(contract_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
        },
        "results": results,
    }
    return output


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("nafnet", "restormer", "mprnet"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        args.output.resolve()
        if args.output
        else C009_ROOT / "smoke" / f"g11_official_{args.family}_smoke_v3.json"
    )
    try:
        result = run_smoke(args)
    except Exception as error:
        failure = {
            "schema_version": "MERIT-C009-G11-OFFICIAL-SMOKE-v3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAILED_OR_BLOCKED",
            "family": args.family,
            "confirmatory_training": False,
            "full_training_allowed_by_this_artifact": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        }
        write_json(output_path, failure)
        raise
    write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
