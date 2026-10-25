"""Restart-capable additional-run trainer for C009 WP2 and WP3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
C006_ROOT = C009_ROOT.parent / "c006_execution"
DEFAULT_PROTOCOL = C009_ROOT / "protocol" / "wp23_training_contract_v1.json"
DEFAULT_FREEZE = C009_ROOT / "protocol" / "wp23_execution_freeze_v1.json"
DEFAULT_PILOT_CLOSURE = C009_ROOT / "protocol" / "wp23_pilot_closure_v1.json"
DEFAULT_OUTPUT = C009_ROOT / "runs"
DEFAULT_WP1_ROOT = C009_ROOT / "runs"


class FollowonTrainingError(RuntimeError):
    """Raised when a WP2/WP3 frozen invariant is violated."""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FollowonTrainingError(f"Cannot load local helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WP1 = load_module("c009_wp1_train_followon", HERE / "c009_wp1_train.py")


def verify_payload(document: dict[str, Any]) -> None:
    payload = dict(document)
    recorded = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(WP1.canonical_json_bytes(payload)).hexdigest()
    if recorded != actual:
        raise FollowonTrainingError("Internal JSON payload hash mismatch")


def verify_gate(
    protocol_path: Path,
    freeze_path: Path,
    pilot: bool,
    pilot_closure_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = WP1.load_json(protocol_path)
    freeze = WP1.load_json(freeze_path)
    verify_payload(freeze)
    if protocol.get("status") != "FROZEN_BEFORE_WP1_OR_WP23_PERFORMANCE_RESULTS":
        raise FollowonTrainingError("WP23 training contract is not frozen")
    if freeze.get("status") != "PASS":
        raise FollowonTrainingError("WP23 execution freeze is not PASS")
    paths = {
        protocol_path: freeze["frozen_files"]["wp23_training_contract_v1.json"],
        Path(__file__).resolve(): freeze["frozen_files"]["c009_wp23_train.py"],
        HERE / "c009_wp23_models.py": freeze["frozen_files"]["c009_wp23_models.py"],
        HERE / "c009_wp1_train.py": freeze["frozen_files"]["c009_wp1_train.py"],
        HERE / "g11_training_stream.py": freeze["frozen_files"]["g11_training_stream.py"],
        HERE / "g11_official_model_factory.py": freeze["frozen_files"]["g11_official_model_factory.py"],
        C009_ROOT / "protocol" / "g11_training_contract_v2.json": freeze["frozen_files"]["g11_training_contract_v2.json"],
        C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json": freeze["frozen_files"]["g11_implementation_manifest_v3.json"],
        C009_ROOT / "protocol" / "g11_source_preflight_v3.json": freeze["frozen_files"]["g11_source_preflight_v3.json"],
        C006_ROOT / "c006_train.py": freeze["frozen_files"]["c006_train.py"],
    }
    for path, expected in paths.items():
        actual = WP1.sha256_file(path)
        if actual != expected:
            raise FollowonTrainingError(
                f"Frozen follow-on file changed: {path} expected={expected} actual={actual}"
            )
    if not pilot:
        closure = WP1.load_json(pilot_closure_path)
        verify_payload(closure)
        if closure.get("status") != "PASS":
            raise FollowonTrainingError("WP23 pilot closure is not PASS")
        if closure.get("trainer_sha256") != WP1.sha256_file(Path(__file__).resolve()):
            raise FollowonTrainingError("WP23 pilot used a different trainer")
    return protocol, freeze


def verify_wp1_complete(wp1_root: Path, seed: int) -> dict[str, Any]:
    queue_path = wp1_root / "queue" / "full_queue_status.json"
    queue = WP1.load_json(queue_path)
    if queue.get("status") != "complete" or int(queue.get("completed_runs", -1)) != 18:
        raise FollowonTrainingError("WP1 18-run queue is not complete")
    run_dir = wp1_root / "wp1" / "nafnet" / f"merit_full_seed{seed}"
    manifest = WP1.verify_complete(run_dir)
    if manifest is None:
        raise FollowonTrainingError(f"WP1 NAFNet full run is incomplete: seed {seed}")
    return manifest


def build_model(
    wp: str,
    structure: str,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    WP1.set_determinism(seed)
    if wp == "wp2" and structure == "zero_prompt_full":
        factory = load_module(
            "g11_official_model_factory_wp23", HERE / "g11_official_model_factory.py"
        )
        model, metadata = factory.build_official_adapter(
            "nafnet",
            C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json",
            C009_ROOT / "protocol" / "g11_source_preflight_v3.json",
        )
        return model, {"wp": wp, "structure": structure, **metadata}, "base"
    if wp == "wp3" and structure in {"cascade", "parallel"}:
        models = load_module("c009_wp23_models_runtime", HERE / "c009_wp23_models.py")
        model, metadata = models.build_structure(structure)
        return model, {"wp": wp, **metadata}, "merit_full"
    raise FollowonTrainingError(f"Invalid WP/structure pair: {wp}/{structure}")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    freeze_path = args.execution_freeze.resolve()
    protocol, freeze = verify_gate(
        protocol_path,
        freeze_path,
        args.pilot,
        args.pilot_closure.resolve(),
    )
    expected_pairs = {
        (row["wp"], row["structure"], int(row["seed"]))
        for row in protocol["run_matrix"]
    }
    if (args.wp, args.structure, args.seed) not in expected_pairs:
        raise FollowonTrainingError("Requested run is outside the frozen WP23 matrix")
    wp1_manifest = verify_wp1_complete(args.wp1_run_root.resolve(), args.seed)

    shared = protocol["shared_training"]
    total_steps = int(args.steps if args.pilot else shared["steps"])
    if args.pilot and not 1 <= total_steps <= 50:
        raise FollowonTrainingError("Pilot requires 1..50 steps")
    if not args.pilot and args.steps is not None and int(args.steps) != int(shared["steps"]):
        raise FollowonTrainingError("Full follow-on runs require exactly 8,000 steps")
    stop_after = total_steps if args.stop_after_step is None else int(args.stop_after_step)
    if not 1 <= stop_after <= total_steps:
        raise FollowonTrainingError("Invalid stop-after-step")
    if not args.pilot and stop_after != total_steps:
        raise FollowonTrainingError("Bounded stop is pilot-only")
    checkpoint_every = 5 if args.pilot else int(shared["recovery_checkpoint_interval_steps"])
    if args.pilot and stop_after < total_steps and stop_after % checkpoint_every:
        raise FollowonTrainingError("Pilot stop must be a recovery boundary")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise FollowonTrainingError("WP23 execution requires CUDA")
    run_kind = "pilot23" if args.pilot else args.wp
    run_dir = (
        args.output_dir.resolve()
        / run_kind
        / f"{args.structure}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = WP1.verify_complete(run_dir)
    if complete is not None:
        print(json.dumps({"status": "ALREADY_COMPLETE", "run_dir": str(run_dir)}))
        return complete

    g11_contract_path = C009_ROOT / "protocol" / "g11_training_contract_v2.json"
    g11_contract = WP1.load_json(g11_contract_path)
    stream_module = load_module(
        "g11_training_stream_wp23", HERE / "g11_training_stream.py"
    )
    loss_module = WP1.load_c006_helpers()
    bank, rows, bank_audit = stream_module.verify_source_bank(g11_contract)
    train_indices = np.array(
        [int(row["bank_index"]) for row in rows if row["split"] == "train"],
        dtype=np.int64,
    )
    model, model_metadata, input_arm = build_model(
        args.wp, args.structure, args.seed
    )
    initial_state_hash = WP1.state_dict_sha256(model)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.wp == "wp2":
        if initial_state_hash != wp1_manifest["initial_state_sha256"]:
            raise FollowonTrainingError(
                "WP2 zero-prompt-full initial state differs from WP1 informative NAFNet"
            )
        if parameter_count != int(wp1_manifest["trainable_parameters"]):
            raise FollowonTrainingError("WP2 parameter count differs from WP1 NAFNet")
    else:
        expected = int(protocol["wp3"][args.structure]["parameters"])
        if parameter_count != expected:
            raise FollowonTrainingError(
                f"WP3 parameter mismatch: {parameter_count} != {expected}"
            )
    model = model.to(device).train()

    c006_protocol = loss_module.load_json(C006_ROOT / "c006_protocol.json")
    if (
        c006_protocol["deepcrack_checkpoint"]["sha256"]
        != g11_contract["training"]["deepcrack_checkpoint_sha256"]
    ):
        raise FollowonTrainingError("DeepCrack hash mismatch")
    segmenter = loss_module.load_deepcrack(c006_protocol, device)
    task_targets = loss_module.precompute_task_targets(
        segmenter, bank, train_indices, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(shared["learning_rate"]),
        weight_decay=float(shared["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_steps if args.pilot else int(shared["cosine_t_max"])),
    )
    stream = WP1.PairedTrainingStream(
        bank,
        train_indices,
        args.seed,
        int(shared["batch_size"]),
        g11_contract["stream"]["degradation"],
    )

    manifest_path = run_dir / "run_manifest.json"
    log_path = run_dir / "train.jsonl"
    recovery_path = run_dir / "recovery.pt"
    final_path = run_dir / "final.pt"
    start_step = 0
    elapsed_before = 0.0
    block_hashes: list[dict[str, Any]] = []
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location=device, weights_only=False)
        metadata = recovery["metadata"]
        invariants = {
            "wp": args.wp,
            "structure": args.structure,
            "seed": args.seed,
            "total_steps": total_steps,
            "initial_state_sha256": initial_state_hash,
            "execution_freeze_sha256": WP1.sha256_file(freeze_path),
        }
        for key, expected in invariants.items():
            if metadata.get(key) != expected:
                raise FollowonTrainingError(f"Recovery mismatch for {key}")
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        stream.set_state(recovery["stream_rng_state"])
        WP1.restore_rng(recovery["rng"], device)
        start_step = int(metadata["global_step"])
        elapsed_before = float(metadata["elapsed_seconds"])
        block_hashes = list(metadata["stream_block_hashes"])
        WP1.truncate_log_to_step(log_path, start_step)
        resume_count = int(metadata.get("resume_count", 0)) + 1
    else:
        log_path.write_text("", encoding="utf-8", newline="\n")
        resume_count = 0

    run_manifest: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP23-RUN-v1",
        "status": "running",
        "run_kind": run_kind,
        "wp": args.wp,
        "structure": args.structure,
        "seed": args.seed,
        "total_steps": total_steps,
        "start_step_this_invocation": start_step,
        "input_arm": input_arm,
        "task_loss_enabled": True,
        "batch_size": int(shared["batch_size"]),
        "patch_size": int(shared["patch_size"]),
        "initial_state_sha256": initial_state_hash,
        "parameters": parameter_count,
        "trainable_parameters": parameter_count,
        "model": model_metadata,
        "bank_audit": bank_audit,
        "train_rows": int(len(train_indices)),
        "wp1_reference_manifest_sha256": WP1.sha256_file(
            args.wp1_run_root.resolve()
            / "wp1"
            / "nafnet"
            / f"merit_full_seed{args.seed}"
            / "run_manifest.json"
        ),
        "wp1_reference_stream_sha256": wp1_manifest["stream_sha256"],
        "protocol_sha256": WP1.sha256_file(protocol_path),
        "g11_contract_sha256": WP1.sha256_file(g11_contract_path),
        "execution_freeze_sha256": WP1.sha256_file(freeze_path),
        "runner_sha256": WP1.sha256_file(Path(__file__).resolve()),
        "resume_count": resume_count,
        "started_or_resumed_at_kst": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "freeze_schema": freeze["schema_version"],
    }
    WP1.write_json_atomic(manifest_path, run_manifest)

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
            WP1.update_stream_digest(stream_digest, zero_step, degraded, target, metadata)
            inputs = stream_module.arm_inputs(degraded, metadata, input_arm)
            input_tensor = torch.from_numpy(inputs).to(device)
            target_tensor = torch.from_numpy(target).to(device)
            optimizer.zero_grad(set_to_none=True)
            with WP1.amp_context(device):
                preclip_output = model.forward_preclip(input_tensor)
            output_float = preclip_output.float()
            target_float = target_tensor.float()
            l1_value = l1_loss(output_float, target_float)
            grad_value = loss_module.grad_loss(output_float, target_float)
            fft_value = loss_module.fft_loss(output_float, target_float)
            target_probability = torch.from_numpy(
                loss_module.task_target_batch(task_targets, metadata)
            ).to(device)
            restored_probability = loss_module.deepcrack_probability(
                segmenter, torch.clamp(output_float, 0.0, 1.0)
            )
            task_value = l1_loss(restored_probability, target_probability)
            total_loss = (
                l1_value
                + grad_value
                + 0.1 * fft_value
                + float(shared["task_lambda"]) * task_value
            )
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
            completed = zero_step + 1
            if (
                completed == start_step + 1
                or completed % int(args.log_every) == 0
                or completed == stop_after
            ):
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                record = {
                    "step_completed": completed,
                    "lr": optimizer.param_groups[0]["lr"],
                    "mean_since_last": {
                        key: value / interval_count for key, value in interval_acc.items()
                    },
                    "elapsed_seconds": elapsed_before + now - invocation_started,
                    "interval_seconds": now - interval_started,
                    "peak_cuda_memory_bytes_this_invocation": torch.cuda.max_memory_allocated(device),
                }
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                log_handle.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                interval_acc = {"l1": 0.0, "grad": 0.0, "fft": 0.0, "task": 0.0}
                interval_count = 0
                interval_started = now
            if completed % checkpoint_every == 0 or completed == stop_after:
                torch.cuda.synchronize(device)
                elapsed = elapsed_before + time.perf_counter() - invocation_started
                block_hashes.append(
                    {
                        "start_step": block_start,
                        "end_step": completed,
                        "sha256": stream_digest.hexdigest(),
                    }
                )
                block_start = completed + 1
                stream_digest = hashlib.sha256()
                metadata_checkpoint = {
                    **run_manifest,
                    "status": "recovery",
                    "global_step": completed,
                    "elapsed_seconds": elapsed,
                    "stream_block_hashes": block_hashes,
                    "resume_count": resume_count,
                }
                WP1.save_checkpoint_atomic(
                    recovery_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "stream_rng_state": stream.get_state(),
                        "rng": WP1.rng_payload(device),
                        "metadata": metadata_checkpoint,
                    },
                )

    torch.cuda.synchronize(device)
    elapsed = elapsed_before + time.perf_counter() - invocation_started
    if stop_after < total_steps:
        run_manifest.update(
            {
                "status": "paused_after_pilot_boundary",
                "global_step": stop_after,
                "elapsed_seconds": elapsed,
                "stream_block_hashes": block_hashes,
                "recovery_checkpoint_sha256": WP1.sha256_file(recovery_path),
            }
        )
        WP1.write_json_atomic(manifest_path, run_manifest)
        return run_manifest

    stream_hash = hashlib.sha256(WP1.canonical_json_bytes(block_hashes)).hexdigest()
    if not args.pilot and stream_hash != wp1_manifest["stream_sha256"]:
        raise FollowonTrainingError(
            "Additional-run stream hash differs from the corresponding WP1 NAFNet stream"
        )
    final_state_hash = WP1.state_dict_sha256(model)
    final_metadata = {
        **run_manifest,
        "status": "complete",
        "global_step": total_steps,
        "elapsed_seconds": elapsed,
        "stream_block_hashes": block_hashes,
        "stream_sha256": stream_hash,
        "final_state_sha256": final_state_hash,
        "checkpoint_type": "fixed_final",
    }
    WP1.save_checkpoint_atomic(
        final_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metadata": final_metadata,
        },
    )
    final_hash = WP1.sha256_file(final_path)
    run_manifest.update(
        {
            "status": "complete",
            "global_step": total_steps,
            "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "stream_block_hashes": block_hashes,
            "stream_sha256": stream_hash,
            "stream_matches_wp1_nafnet": (stream_hash == wp1_manifest["stream_sha256"]),
            "final_state_sha256": final_state_hash,
            "final_checkpoint_sha256": final_hash,
            "recovery_checkpoint_sha256": WP1.sha256_file(recovery_path),
            "peak_cuda_memory_bytes_this_invocation": torch.cuda.max_memory_allocated(device),
        }
    )
    WP1.write_json_atomic(manifest_path, run_manifest)
    marker: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP23-RUN-COMPLETE-v1",
        "status": "PASS",
        "wp": args.wp,
        "structure": args.structure,
        "seed": args.seed,
        "global_step": total_steps,
        "final_checkpoint_sha256": final_hash,
        "final_state_sha256": final_state_hash,
        "stream_sha256": stream_hash,
        "run_manifest_sha256": WP1.sha256_file(manifest_path),
    }
    marker["payload_sha256"] = hashlib.sha256(
        WP1.canonical_json_bytes(marker)
    ).hexdigest()
    WP1.write_json_atomic(run_dir / "RUN_COMPLETE.json", marker)
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wp", choices=("wp2", "wp3"), required=True)
    parser.add_argument(
        "--structure",
        choices=("zero_prompt_full", "cascade", "parallel"),
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wp1-run-root", type=Path, default=DEFAULT_WP1_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execution-freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--pilot-closure", type=Path, default=DEFAULT_PILOT_CLOSURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_kind = "pilot23" if args.pilot else args.wp
    failure_path = (
        args.output_dir.resolve()
        / run_kind
        / f"{args.structure}_seed{args.seed}"
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
            "wp": args.wp,
            "structure": args.structure,
            "seed": args.seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        WP1.write_json_atomic(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
