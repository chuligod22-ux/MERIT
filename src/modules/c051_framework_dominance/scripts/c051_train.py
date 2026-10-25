"""Train one deterministic C051 backbone/arm/seed cell with atomic resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import c051_common as common
import c051_dataset as dataset

STEPS = 4000
BATCH = 4


def load_segmenter(device: torch.device) -> torch.nn.Module:
    segmenter_dir = common.NRIQA / "03_src" / "crack_seg"
    sys.path.insert(0, str(segmenter_dir))
    from utils import load_unet_vgg16

    checkpoint = segmenter_dir / "models" / "model_unet_vgg16_best.pt"
    model = load_unet_vgg16(str(checkpoint)).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def grad_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    horizontal = (
        output[..., 1:] - output[..., :-1] - (target[..., 1:] - target[..., :-1])
    )
    vertical = (
        output[..., 1:, :]
        - output[..., :-1, :]
        - (target[..., 1:, :] - target[..., :-1, :])
    )
    return horizontal.abs().mean() + vertical.abs().mean()


def frequency_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.fft.rfft2(output.float(), norm="ortho") - torch.fft.rfft2(
        target.float(), norm="ortho"
    )
    return (
        difference.abs().detach()
        * (difference.real.square() + difference.imag.square())
    ).mean()


def make_batch(
    bank: np.ndarray, stream: dict[str, np.ndarray], step: int, arm: str
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    indices = stream["indices"][step]
    clean = bank[indices] / np.float32(255.0)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for position, source in enumerate(clean):
        sigma_h = float(stream["sigma_h"][step, position])
        sigma_v = float(stream["sigma_v"][step, position])
        noise_std = float(stream["noise_std"][step, position])
        rng = np.random.RandomState(int(stream["noise_seed"][step, position]))
        degraded = gaussian_filter(source, (sigma_v, sigma_h), mode="reflect")
        degraded = degraded + rng.standard_normal(source.shape).astype(np.float32) * (
            noise_std / 255.0
        )
        degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)
        if arm == "on":
            inputs.append(
                np.stack(
                    [
                        degraded,
                        np.full_like(
                            degraded, 0.5 * (sigma_h + sigma_v) / dataset.BLUR_MAX
                        ),
                        np.full_like(degraded, noise_std / dataset.NOISE_MAX),
                    ]
                )
            )
        else:
            inputs.append(degraded[None])
        targets.append(source[None])
    return (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(targets)),
        indices,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    specification: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": "MERIT-C051-TRAINING-CHECKPOINT-v1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "next_step": step,
            "specification": specification,
        },
        temporary,
    )
    os.replace(temporary, path)


def execute(backbone: str, arm: str, seed: int) -> dict[str, Any]:
    if (
        backbone not in common.BACKBONES
        or arm not in common.ARMS
        or seed not in common.SEEDS
    ):
        raise common.C051Error("Invalid C051 cell")
    if not torch.cuda.is_available():
        raise common.C051Error("CUDA is required")
    protocol = common.validate_protocol()
    manifest = common.load_json(common.ASSETS / "assets_manifest.json")
    if common.payload_sha256(manifest) != manifest.get("payload_sha256"):
        raise common.C051Error("Asset manifest payload mismatch")
    bank_path = Path(str(manifest["bank"]["path"]))
    target_path = Path(str(manifest["task_targets"]["path"]))
    stream_path = Path(str(manifest["streams"][str(seed)]["path"]))
    for path, expected in (
        (bank_path, manifest["bank"]["sha256"]),
        (target_path, manifest["task_targets"]["sha256"]),
        (stream_path, manifest["streams"][str(seed)]["sha256"]),
    ):
        if common.sha256_file(path) != expected:
            raise common.C051Error(f"Asset hash mismatch: {path}")
    common.seed_everything(202610510 + seed)
    device = torch.device("cuda")
    off_model, on_model = common.paired_initial_models(backbone, seed)
    model = (on_model if arm == "on" else off_model).to(device)
    del off_model, on_model
    segmenter = load_segmenter(device) if arm == "on" else None
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    bank = np.load(bank_path, mmap_mode="r")
    targets = np.load(target_path, mmap_mode="r")
    loaded = np.load(stream_path)
    stream = {name: loaded[name] for name in loaded.files}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS)
    output = common.run_dir(backbone, arm, seed)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "resume.pt"
    final_checkpoint = output / "final.pt"
    log_path = output / "training.jsonl"
    specification = {
        "backbone": backbone,
        "arm": arm,
        "seed": seed,
        "width": common.WIDTHS[backbone],
        "steps": STEPS,
        "batch_size": BATCH,
        "asset_manifest_payload_sha256": manifest["payload_sha256"],
        "protocol_payload_sha256": protocol["payload_sha256"],
    }
    start_step = 0
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["specification"] != specification:
            raise common.C051Error("Resume specification mismatch")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_step = int(saved["next_step"])
    model.train()
    started = time.perf_counter()
    for step in range(start_step, STEPS):
        input_cpu, target_cpu, indices = make_batch(bank, stream, step, arm)
        image = input_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restored = model(image)
        if arm == "on":
            fidelity = (
                torch.nn.functional.l1_loss(restored.float(), target)
                + grad_loss(restored.float(), target)
                + 0.1 * frequency_loss(restored, target)
            )
            clean_probability = torch.from_numpy(targets[indices].astype(np.float32))[
                :, None
            ].to(device)
            restored_rgb = (restored.float().repeat(1, 3, 1, 1) - mean) / std
            task = torch.nn.functional.l1_loss(
                torch.sigmoid(segmenter(restored_rgb))[:, :1], clean_probability
            )
            loss = fidelity + 5.0 * task
        else:
            fidelity = torch.nn.functional.l1_loss(restored.float(), target)
            task = torch.zeros((), device=device)
            loss = fidelity
        if not torch.isfinite(loss):
            raise common.C051Error(f"Non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        record = {
            "step": step + 1,
            "loss": float(loss.detach()),
            "fidelity": float(fidelity.detach()),
            "task": float(task.detach()),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        if (step + 1) % 250 == 0 or step + 1 == STEPS:
            save_checkpoint(
                checkpoint, model, optimizer, scheduler, step + 1, specification
            )
    if scheduler.last_epoch != STEPS:
        raise common.C051Error(f"Unexpected scheduler step: {scheduler.last_epoch}")
    save_checkpoint(final_checkpoint, model, optimizer, scheduler, STEPS, specification)
    result: dict[str, Any] = {
        "schema_version": "MERIT-C051-TRAINING-RUN-v1",
        "status": "PASS_COMPLETE",
        "model_id": common.model_id(backbone, arm, seed),
        "specification": specification,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": common.sha256_file(final_checkpoint),
        "log": str(log_path.resolve()),
        "log_sha256": common.sha256_file(log_path),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    result["payload_sha256"] = common.payload_sha256(result)
    common.write_json_atomic(output / "RUN_COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=common.BACKBONES, required=True)
    parser.add_argument("--arm", choices=common.ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=common.SEEDS, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            execute(args.backbone, args.arm, args.seed), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
