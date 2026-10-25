"""Two-stage MERIT v2.1 development pilot."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c010_pilot as pilot


PROTOCOL = pilot.C010 / "protocol" / "pilot_v2.json"
RUN_DIR = pilot.C010 / "runs" / "pilot_v2_nafnet_seed0"


def stream_for(protocol: dict[str, Any]):
    g11 = pilot.load_json(pilot.G11_CONTRACT)
    bank, rows, _ = pilot.training_stream.verify_source_bank(g11)
    indices = pilot.np.asarray(
        [int(row["bank_index"]) for row in rows if row["split"] == "train"],
        dtype=pilot.np.int64,
    )
    config = protocol["training"]
    return pilot.training_stream.latent_stream(
        bank,
        indices,
        int(config["seed"]),
        int(config["batch_size"]),
        g11["stream"]["degradation"],
    )


def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def calibrate_blend(
    model: pilot.MeritV2,
    protocol: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    config = protocol["training"]["blend_calibration"]
    set_trainable(model.corrector, False)
    set_trainable(model.corrector.prompt_mlp, True)
    optimizer = torch.optim.AdamW(
        model.corrector.prompt_mlp.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["steps"])
    )
    stream = stream_for(protocol)
    log: list[dict[str, Any]] = []
    accum = defaultdict(float)
    interval = 0
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        degraded, target, metadata = next(stream)
        inputs = torch.from_numpy(pilot.prompt_inputs(degraded, metadata)).to(device)
        target_tensor = torch.from_numpy(target).to(device).float()
        source = inputs[:, :1]
        zeros = torch.zeros_like(source)
        with torch.no_grad(), pilot.autocast(device):
            base = model.base.forward_preclip(torch.cat([source, zeros, zeros], dim=1))
        source_float = source.float()
        base_float = base.float()
        direction = base_float - source_float
        numerator = torch.sum(
            (target_tensor - source_float) * direction, dim=(1, 2, 3)
        )
        denominator = torch.sum(direction.square(), dim=(1, 2, 3)).clamp_min(1e-12)
        target_alpha = torch.clamp(numerator / denominator, 0.5, 1.5).detach()
        prompt_values = torch.cat(
            [
                inputs[:, 1:2].float().mean(dim=(2, 3)),
                inputs[:, 2:3].float().mean(dim=(2, 3)),
            ],
            dim=1,
        )
        predicted = 1.0 + model.corrector.blend_span * torch.tanh(
            model.corrector.prompt_mlp(prompt_values)
        ).flatten()
        loss = torch.mean((predicted - target_alpha) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        accum["mse"] += float(loss.detach().cpu())
        accum["target_alpha"] += float(target_alpha.mean().cpu())
        accum["predicted_alpha"] += float(predicted.detach().mean().cpu())
        interval += 1
        if step == 1 or step % 100 == 0 or step == int(config["steps"]):
            record = {
                "stage": "blend_calibration",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "mean_since_last": {key: value / interval for key, value in accum.items()},
            }
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
            log.append(record)
            accum = defaultdict(float)
            interval = 0
    return log


def fit_residual(
    model: pilot.MeritV2,
    protocol: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    config = protocol["training"]["residual_fit"]
    set_trainable(model.corrector, True)
    set_trainable(model.corrector.prompt_mlp, False)
    parameters = [p for p in model.corrector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["steps"])
    )
    stream = stream_for(protocol)
    l1 = torch.nn.L1Loss()
    log: list[dict[str, Any]] = []
    accum = defaultdict(float)
    interval = 0
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        degraded, target, metadata = next(stream)
        inputs = torch.from_numpy(pilot.prompt_inputs(degraded, metadata)).to(device)
        target_tensor = torch.from_numpy(target).to(device)
        optimizer.zero_grad(set_to_none=True)
        with pilot.autocast(device):
            output, _, residual, blend = model.forward_components(inputs)
        output_float = output.float()
        target_float = target_tensor.float()
        l1_value = l1(output_float, target_float)
        grad_value = pilot.grad_loss(output_float, target_float)
        fft_value = pilot.fft_loss(output_float, target_float)
        loss = l1_value + grad_value + 0.1 * fft_value
        if not torch.isfinite(loss):
            raise pilot.PilotError(f"Non-finite residual loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        values = {
            "loss": float(loss.detach().cpu()),
            "l1": float(l1_value.detach().cpu()),
            "grad": float(grad_value.detach().cpu()),
            "fft": float(fft_value.detach().cpu()),
            "residual_abs": float(residual.detach().float().abs().mean().cpu()),
            "blend_mean": float(blend.detach().float().mean().cpu()),
        }
        for key, value in values.items():
            accum[key] += value
        interval += 1
        if step == 1 or step % 100 == 0 or step == int(config["steps"]):
            record = {
                "stage": "residual_fit",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "mean_since_last": {key: value / interval for key, value in accum.items()},
            }
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
            log.append(record)
            accum = defaultdict(float)
            interval = 0
    set_trainable(model.corrector.prompt_mlp, True)
    return log


def main() -> None:
    if not torch.cuda.is_available():
        raise pilot.PilotError("CUDA is required")
    if (RUN_DIR / "RUN_COMPLETE.json").exists():
        print((RUN_DIR / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        return
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    protocol = pilot.load_json(PROTOCOL)
    seed = int(protocol["training"]["seed"])
    pilot.set_determinism(seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model, provenance = pilot.build_model(protocol, device)
    model.train()
    started = time.perf_counter()
    blend_log = calibrate_blend(model, protocol, device)
    residual_log = fit_residual(model, protocol, device)
    log_path = RUN_DIR / "train.jsonl"
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in blend_log + residual_log:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    checkpoint_path = RUN_DIR / "final.pt"
    torch.save(
        {
            "corrector": model.corrector.state_dict(),
            "protocol_sha256": pilot.sha256_file(PROTOCOL),
            "base_checkpoint_sha256": provenance["base_checkpoint_sha256"],
        },
        checkpoint_path,
    )
    development = pilot.evaluate(model, protocol, device, RUN_DIR)
    result = {
        "schema_version": "MERIT-C010-V2-PILOT-COMPLETE-v2",
        "status": "PASS" if development["status"] == "PASS_DEVELOPMENT_TARGETS" else "ITERATE",
        "scientific_status": "EXPLORATORY_UNBLINDED_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": pilot.sha256_file(PROTOCOL),
        "runner_sha256": pilot.sha256_file(Path(__file__).resolve()),
        "model_sha256": pilot.sha256_file(HERE / "merit_v2_model.py"),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": pilot.sha256_file(checkpoint_path),
        "training_log_path": str(log_path.resolve()),
        "training_log_sha256": pilot.sha256_file(log_path),
        "provenance": provenance,
        "development": development,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    pilot.write_json_atomic(RUN_DIR / "RUN_COMPLETE.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
