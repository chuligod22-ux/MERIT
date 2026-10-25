"""Restart-capable official-backbone WP1 trainer for MERIT C009.

The non-pilot path is intentionally rigid: 8,000 steps, batch four, 448-pixel
patches, bfloat16, frozen D0-only stream, and fixed-final checkpoint selection.
The paired arms differ only in prompt channels and the predeclared task loss.
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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
REVISION_ROOT = C009_ROOT.parent
C006_ROOT = REVISION_ROOT / "c006_execution"
DEFAULT_CONTRACT = C009_ROOT / "protocol" / "g11_training_contract_v2.json"
DEFAULT_MANIFEST = C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json"
DEFAULT_PREFLIGHT = C009_ROOT / "protocol" / "g11_source_preflight_v3.json"
DEFAULT_G11 = C009_ROOT / "protocol" / "g11_closure_v1.json"
DEFAULT_G13 = C009_ROOT / "protocol" / "g13_feasibility_v2.json"
DEFAULT_FREEZE = C009_ROOT / "protocol" / "wp1_execution_freeze_v1.json"
DEFAULT_OUTPUT = C009_ROOT / "runs"


class TrainingError(RuntimeError):
    """Raised when a frozen invariant is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_sibling_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TrainingError(f"Cannot load local helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_c006_helpers():
    sys.path.insert(0, str(C006_ROOT))
    try:
        import c006_train
    finally:
        sys.path.pop(0)
    return c006_train


def verify_payload_hash(document: dict[str, Any]) -> None:
    payload = dict(document)
    recorded = payload.pop("payload_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(payload))
    if recorded != actual:
        raise TrainingError(
            f"Internal payload hash mismatch: recorded={recorded} actual={actual}"
        )


def verify_execution_gate(
    contract_path: Path,
    manifest_path: Path,
    preflight_path: Path,
    g11_path: Path,
    g13_path: Path,
    freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(contract_path)
    g11 = load_json(g11_path)
    g13 = load_json(g13_path)
    freeze = load_json(freeze_path)
    verify_payload_hash(g13)
    verify_payload_hash(freeze)
    if g11.get("status") != "CLOSED_WITH_DISCLOSED_PARTIAL_FLOP_COVERAGE":
        raise TrainingError("G11 is not closed")
    if g13.get("status") != "CLOSED_GO_FULL_SCOPE_AT_AUTHOR_RISK":
        raise TrainingError("G13 v2 does not authorize full-scope execution")
    if not g13.get("full_training_allowed"):
        raise TrainingError("G13 v2 full_training_allowed is false")
    if freeze.get("status") != "PASS":
        raise TrainingError("WP1 execution freeze is not PASS")

    expected = {
        contract_path: freeze["frozen_files"]["g11_training_contract_v2.json"],
        manifest_path: freeze["frozen_files"]["g11_implementation_manifest_v3.json"],
        preflight_path: freeze["frozen_files"]["g11_source_preflight_v3.json"],
        g11_path: freeze["frozen_files"]["g11_closure_v1.json"],
        g13_path: freeze["frozen_files"]["g13_feasibility_v2.json"],
        Path(__file__).resolve(): freeze["frozen_files"]["c009_wp1_train.py"],
        HERE / "g11_training_stream.py": freeze["frozen_files"]["g11_training_stream.py"],
        HERE / "g11_official_model_factory.py": freeze["frozen_files"]["g11_official_model_factory.py"],
        HERE / "g11_official_backbone_adapter.py": freeze["frozen_files"]["g11_official_backbone_adapter.py"],
        C006_ROOT / "c006_train.py": freeze["frozen_files"]["c006_train.py"],
    }
    for path, expected_hash in expected.items():
        actual = sha256_file(path)
        if actual.lower() != str(expected_hash).lower():
            raise TrainingError(
                f"Frozen file changed: {path} expected={expected_hash} actual={actual}"
            )
    if g13["source_evidence"]["g11_closure_sha256"] != sha256_file(g11_path):
        raise TrainingError("G13 does not bind the active G11 closure")
    if g13["source_evidence"]["training_contract_sha256"] != sha256_file(
        contract_path
    ):
        raise TrainingError("G13 does not bind the active training contract")
    if freeze["g13_sha256"] != sha256_file(g13_path):
        raise TrainingError("Execution freeze does not bind the active G13 v2")
    return contract, freeze


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


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class PairedTrainingStream:
    """Stateful implementation of the frozen G11 NumPy draw order."""

    def __init__(
        self,
        bank: np.ndarray,
        train_indices: np.ndarray,
        seed: int,
        batch_size: int,
        degradation: dict[str, list[float]],
    ) -> None:
        self.bank = bank
        self.train_indices = train_indices
        self.batch_size = batch_size
        self.degradation = degradation
        self.rng = np.random.RandomState(seed)

    def get_state(self) -> tuple[Any, ...]:
        return self.rng.get_state()

    def set_state(self, state: tuple[Any, ...]) -> None:
        self.rng.set_state(state)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        patch_size = int(self.bank.shape[-1])
        selected = self.train_indices[
            self.rng.randint(0, len(self.train_indices), size=self.batch_size)
        ]
        degraded_batch: list[np.ndarray] = []
        target_batch: list[np.ndarray] = []
        samples: list[dict[str, float | int]] = []
        for bank_index in selected:
            clean = self.bank[int(bank_index)].astype(np.float32) / np.float32(255.0)
            sigma_h = float(self.rng.uniform(*self.degradation["sigma_h_px"]))
            sigma_v = float(self.rng.uniform(*self.degradation["sigma_v_px"]))
            noise_std = float(
                self.rng.uniform(*self.degradation["noise_std_8bit"])
            )
            degraded = gaussian_filter(
                clean, sigma=(sigma_v, sigma_h), mode="reflect"
            ).astype(np.float32)
            noise = self.rng.randn(patch_size, patch_size).astype(np.float32)
            degraded = degraded + noise * np.float32(noise_std / 255.0)
            degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)
            degraded_batch.append(degraded)
            target_batch.append(clean[None])
            samples.append(
                {
                    "bank_index": int(bank_index),
                    "sigma_h_px": sigma_h,
                    "sigma_v_px": sigma_v,
                    "noise_std_8bit": noise_std,
                }
            )
        return (
            np.stack(degraded_batch).astype(np.float32, copy=False),
            np.stack(target_batch).astype(np.float32, copy=False),
            {"samples": samples},
        )


def update_stream_digest(
    digest: Any,
    step: int,
    degraded: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    digest.update(step.to_bytes(8, byteorder="big"))
    digest.update(canonical_json_bytes(metadata))
    digest.update(np.ascontiguousarray(degraded).tobytes(order="C"))
    digest.update(np.ascontiguousarray(target).tobytes(order="C"))


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_payload(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        ),
    }


def restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy_global"])
    # A checkpoint loaded with map_location=cuda also moves the saved CPU RNG
    # byte tensor.  Both CPU and CUDA generator setters require CPU ByteTensor
    # state objects and copy them internally to the appropriate generator.
    torch.set_rng_state(payload["torch_cpu"].cpu())
    if device.type == "cuda" and payload["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["torch_cuda"]])


def truncate_log_to_step(path: Path, completed_step: int) -> None:
    if not path.exists():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["step_completed"]) <= completed_step:
            retained.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    text = "\n".join(retained)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def verify_complete(run_dir: Path) -> dict[str, Any] | None:
    marker_path = run_dir / "RUN_COMPLETE.json"
    final_path = run_dir / "final.pt"
    manifest_path = run_dir / "run_manifest.json"
    if not (marker_path.exists() and final_path.exists() and manifest_path.exists()):
        return None
    marker = load_json(marker_path)
    verify_payload_hash(marker)
    if marker.get("status") != "PASS":
        raise TrainingError(f"Invalid completion marker: {marker_path}")
    if marker["final_checkpoint_sha256"] != sha256_file(final_path):
        raise TrainingError(f"Completed checkpoint hash mismatch: {final_path}")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise TrainingError("Completion marker exists but manifest is not complete")
    return manifest


def paired_manifest_check(
    run_dir: Path,
    arm: str,
    seed: int,
    initial_state_sha256: str,
    parameter_count: int,
) -> None:
    other = "merit_full" if arm == "base" else "base"
    sibling = run_dir.parent / f"{other}_seed{seed}" / "run_manifest.json"
    if not sibling.exists():
        return
    record = load_json(sibling)
    if record.get("initial_state_sha256") != initial_state_sha256:
        raise TrainingError(f"Paired initial state differs from {sibling}")
    if int(record.get("trainable_parameters", -1)) != parameter_count:
        raise TrainingError(f"Paired trainable parameter count differs from {sibling}")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "contract": args.contract.resolve(),
        "manifest": args.manifest.resolve(),
        "preflight": args.preflight.resolve(),
        "g11": args.g11.resolve(),
        "g13": args.g13.resolve(),
        "freeze": args.execution_freeze.resolve(),
    }
    contract, execution_freeze = verify_execution_gate(
        paths["contract"],
        paths["manifest"],
        paths["preflight"],
        paths["g11"],
        paths["g13"],
        paths["freeze"],
    )
    comparison = contract["paired_comparison"]
    training = contract["training"]
    if args.family not in comparison["backbones"]:
        raise TrainingError(f"Family is not frozen: {args.family}")
    if args.arm not in comparison["arms"]:
        raise TrainingError(f"Arm is not frozen: {args.arm}")
    if args.seed not in comparison["seeds"]:
        raise TrainingError(f"Seed is not frozen: {args.seed}")

    frozen_steps = int(training["steps"])
    total_steps = int(args.steps if args.pilot else frozen_steps)
    if args.pilot:
        if not 1 <= total_steps <= 50:
            raise TrainingError("Pilot runs require 1..50 total steps")
    elif args.steps is not None and int(args.steps) != frozen_steps:
        raise TrainingError(f"Full runs require exactly {frozen_steps} steps")
    stop_after = total_steps if args.stop_after_step is None else int(args.stop_after_step)
    if not 1 <= stop_after <= total_steps:
        raise TrainingError("stop-after-step must be within the run")
    if not args.pilot and stop_after != total_steps:
        raise TrainingError("A bounded stop is allowed only for pilot resume testing")
    checkpoint_every = 5 if args.pilot else int(
        training["recovery_checkpoint_interval_steps"]
    )
    if args.pilot and stop_after % checkpoint_every != 0 and stop_after != total_steps:
        raise TrainingError("Pilot stop must fall on a recovery boundary")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise TrainingError("The frozen WP1 execution requires an available CUDA device")
    run_kind = "pilot" if args.pilot else "wp1"
    run_dir = (
        args.output_dir.resolve()
        / run_kind
        / args.family
        / f"{args.arm}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = verify_complete(run_dir)
    if complete is not None:
        print(json.dumps({"status": "ALREADY_COMPLETE", "run_dir": str(run_dir)}))
        return complete

    stream_module = load_sibling_module(
        "g11_training_stream_wp1", HERE / "g11_training_stream.py"
    )
    factory_module = load_sibling_module(
        "g11_official_model_factory_wp1", HERE / "g11_official_model_factory.py"
    )
    loss_module = load_c006_helpers()
    bank, rows, bank_audit = stream_module.verify_source_bank(contract)
    train_indices = np.array(
        [int(row["bank_index"]) for row in rows if row["split"] == "train"],
        dtype=np.int64,
    )
    if not len(train_indices):
        raise TrainingError("Frozen bank has no training rows")

    set_determinism(args.seed)
    model, factory_metadata = factory_module.build_official_adapter(
        args.family, paths["manifest"], paths["preflight"]
    )
    initial_state_hash = state_dict_sha256(model)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    paired_manifest_check(
        run_dir, args.arm, args.seed, initial_state_hash, parameter_count
    )
    model = model.to(device).train()

    c006_protocol = loss_module.load_json(C006_ROOT / "c006_protocol.json")
    if (
        c006_protocol["deepcrack_checkpoint"]["sha256"]
        != training["deepcrack_checkpoint_sha256"]
    ):
        raise TrainingError("DeepCrack hash differs between C006 and G11 contract")
    segmenter: torch.nn.Module | None = None
    task_targets: dict[int, np.ndarray] | None = None
    if args.arm == "merit_full":
        segmenter = loss_module.load_deepcrack(c006_protocol, device)
        task_targets = loss_module.precompute_task_targets(
            segmenter, bank, train_indices, device
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_steps if args.pilot else int(training["cosine_t_max"])),
    )
    stream = PairedTrainingStream(
        bank,
        train_indices,
        args.seed,
        int(training["batch_size"]),
        contract["stream"]["degradation"],
    )

    run_manifest_path = run_dir / "run_manifest.json"
    log_path = run_dir / "train.jsonl"
    recovery_path = run_dir / "recovery.pt"
    final_path = run_dir / "final.pt"
    start_step = 0
    elapsed_before = 0.0
    stream_block_hashes: list[dict[str, Any]] = []
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location=device, weights_only=False)
        metadata = recovery["metadata"]
        invariants = {
            "family": args.family,
            "arm": args.arm,
            "seed": args.seed,
            "total_steps": total_steps,
            "initial_state_sha256": initial_state_hash,
            "execution_freeze_sha256": sha256_file(paths["freeze"]),
        }
        for key, expected in invariants.items():
            if metadata.get(key) != expected:
                raise TrainingError(
                    f"Recovery metadata mismatch for {key}: "
                    f"{metadata.get(key)} != {expected}"
                )
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        stream.set_state(recovery["stream_rng_state"])
        restore_rng(recovery["rng"], device)
        start_step = int(metadata["global_step"])
        elapsed_before = float(metadata["elapsed_seconds"])
        stream_block_hashes = list(metadata["stream_block_hashes"])
        truncate_log_to_step(log_path, start_step)
        resume_count = int(metadata.get("resume_count", 0)) + 1
    else:
        log_path.write_text("", encoding="utf-8", newline="\n")
        resume_count = 0

    run_manifest: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP1-RUN-v1",
        "status": "running",
        "run_kind": run_kind,
        "family": args.family,
        "arm": args.arm,
        "seed": args.seed,
        "total_steps": total_steps,
        "start_step_this_invocation": start_step,
        "batch_size": int(training["batch_size"]),
        "patch_size": int(training["patch_size"]),
        "precision": training["precision"],
        "fidelity_output_domain": training["fidelity_output_domain"],
        "task_output_domain": training["task_output_domain"],
        "initial_state_sha256": initial_state_hash,
        "parameters": int(factory_metadata["parameters"]),
        "trainable_parameters": parameter_count,
        "factory": factory_metadata,
        "bank_audit": bank_audit,
        "train_rows": int(len(train_indices)),
        "contract_sha256": sha256_file(paths["contract"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "preflight_sha256": sha256_file(paths["preflight"]),
        "g11_sha256": sha256_file(paths["g11"]),
        "g13_sha256": sha256_file(paths["g13"]),
        "execution_freeze_sha256": sha256_file(paths["freeze"]),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "resume_count": resume_count,
        "started_or_resumed_at_kst": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "authorization": {
            "g13_status": "CLOSED_GO_FULL_SCOPE_AT_AUTHOR_RISK",
            "journal_extension_decision": "PENDING",
        },
        "freeze_schema": execution_freeze["schema_version"],
    }
    write_json_atomic(run_manifest_path, run_manifest)

    l1_loss = torch.nn.L1Loss()
    interval_acc = {"l1": 0.0, "grad": 0.0, "fft": 0.0, "task": 0.0}
    interval_count = 0
    invocation_started = time.perf_counter()
    interval_started = invocation_started
    stream_digest = hashlib.sha256()
    block_start = start_step + 1
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        for zero_step in range(start_step, stop_after):
            degraded, target, metadata = stream.next_batch()
            update_stream_digest(stream_digest, zero_step, degraded, target, metadata)
            inputs = stream_module.arm_inputs(degraded, metadata, args.arm)
            input_tensor = torch.from_numpy(inputs).to(device)
            target_tensor = torch.from_numpy(target).to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                preclip_output = model.forward_preclip(input_tensor)
            output_float = preclip_output.float()
            target_float = target_tensor.float()
            l1_value = l1_loss(output_float, target_float)
            grad_value = loss_module.grad_loss(output_float, target_float)
            fft_value = loss_module.fft_loss(output_float, target_float)
            total_loss = l1_value + grad_value + 0.1 * fft_value
            task_value = torch.zeros((), device=device, dtype=torch.float32)
            if args.arm == "merit_full":
                assert segmenter is not None and task_targets is not None
                target_probability = torch.from_numpy(
                    loss_module.task_target_batch(task_targets, metadata)
                ).to(device)
                restored_probability = loss_module.deepcrack_probability(
                    segmenter, torch.clamp(output_float, 0.0, 1.0)
                )
                task_value = l1_loss(restored_probability, target_probability)
                total_loss = total_loss + float(training["task_lambda"]) * task_value
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss at step {zero_step + 1}")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()

            values = {
                "l1": float(l1_value.detach().cpu()),
                "grad": float(grad_value.detach().cpu()),
                "fft": float(fft_value.detach().cpu()),
                "task": float(task_value.detach().cpu()),
            }
            for key, value in values.items():
                interval_acc[key] += value
            interval_count += 1
            completed_step = zero_step + 1
            should_report = (
                completed_step == start_step + 1
                or completed_step % int(args.log_every) == 0
                or completed_step == stop_after
            )
            if should_report:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                elapsed_total = elapsed_before + (now - invocation_started)
                record = {
                    "step_completed": completed_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "mean_since_last": {
                        key: value / interval_count for key, value in interval_acc.items()
                    },
                    "elapsed_seconds": elapsed_total,
                    "interval_seconds": now - interval_started,
                    "peak_cuda_memory_bytes_this_invocation": torch.cuda.max_memory_allocated(
                        device
                    ),
                }
                log_handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                log_handle.flush()
                print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
                interval_acc = {"l1": 0.0, "grad": 0.0, "fft": 0.0, "task": 0.0}
                interval_count = 0
                interval_started = now

            checkpoint_boundary = completed_step % checkpoint_every == 0
            is_stop = completed_step == stop_after
            if checkpoint_boundary or is_stop:
                torch.cuda.synchronize(device)
                elapsed_total = elapsed_before + (time.perf_counter() - invocation_started)
                stream_block_hashes.append(
                    {
                        "start_step": block_start,
                        "end_step": completed_step,
                        "sha256": stream_digest.hexdigest(),
                    }
                )
                block_start = completed_step + 1
                stream_digest = hashlib.sha256()
                recovery_metadata = {
                    **run_manifest,
                    "status": "recovery",
                    "global_step": completed_step,
                    "elapsed_seconds": elapsed_total,
                    "stream_block_hashes": stream_block_hashes,
                    "resume_count": resume_count,
                }
                save_checkpoint_atomic(
                    recovery_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "stream_rng_state": stream.get_state(),
                        "rng": rng_payload(device),
                        "metadata": recovery_metadata,
                    },
                )

    torch.cuda.synchronize(device)
    elapsed_total = elapsed_before + (time.perf_counter() - invocation_started)
    if stop_after < total_steps:
        run_manifest.update(
            {
                "status": "paused_after_pilot_boundary",
                "global_step": stop_after,
                "elapsed_seconds": elapsed_total,
                "recovery_checkpoint_sha256": sha256_file(recovery_path),
                "stream_block_hashes": stream_block_hashes,
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)
        return run_manifest

    final_state_hash = state_dict_sha256(model)
    final_metadata = {
        **run_manifest,
        "status": "complete",
        "global_step": total_steps,
        "elapsed_seconds": elapsed_total,
        "stream_block_hashes": stream_block_hashes,
        "final_state_sha256": final_state_hash,
        "checkpoint_type": "fixed_final",
    }
    save_checkpoint_atomic(
        final_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metadata": final_metadata,
        },
    )
    final_hash = sha256_file(final_path)
    run_manifest.update(
        {
            "status": "complete",
            "global_step": total_steps,
            "completed_at_kst": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "elapsed_seconds": elapsed_total,
            "final_checkpoint_sha256": final_hash,
            "final_state_sha256": final_state_hash,
            "recovery_checkpoint_sha256": sha256_file(recovery_path),
            "stream_block_hashes": stream_block_hashes,
            "stream_sha256": sha256_bytes(canonical_json_bytes(stream_block_hashes)),
            "peak_cuda_memory_bytes_this_invocation": torch.cuda.max_memory_allocated(
                device
            ),
        }
    )
    write_json_atomic(run_manifest_path, run_manifest)
    marker: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP1-RUN-COMPLETE-v1",
        "status": "PASS",
        "family": args.family,
        "arm": args.arm,
        "seed": args.seed,
        "global_step": total_steps,
        "final_checkpoint_sha256": final_hash,
        "final_state_sha256": final_state_hash,
        "stream_sha256": run_manifest["stream_sha256"],
        "run_manifest_sha256": sha256_file(run_manifest_path),
    }
    marker["payload_sha256"] = sha256_bytes(canonical_json_bytes(marker))
    write_json_atomic(run_dir / "RUN_COMPLETE.json", marker)
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("nafnet", "restormer", "mprnet"), required=True)
    parser.add_argument("--arm", choices=("base", "merit_full"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--g11", type=Path, default=DEFAULT_G11)
    parser.add_argument("--g13", type=Path, default=DEFAULT_G13)
    parser.add_argument("--execution-freeze", type=Path, default=DEFAULT_FREEZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_kind = "pilot" if args.pilot else "wp1"
    failure_path = (
        args.output_dir.resolve()
        / run_kind
        / args.family
        / f"{args.arm}_seed{args.seed}"
        / "failure.json"
    )
    try:
        result = run_training(args)
        if failure_path.exists():
            failure_path.unlink()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as error:
        failure = {
            "status": "FAILED",
            "family": args.family,
            "arm": args.arm,
            "seed": args.seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
