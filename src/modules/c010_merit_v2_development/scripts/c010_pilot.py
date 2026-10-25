"""Train and evaluate the first exploratory MERIT v2 adapter pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
C010 = HERE.parent
C009 = C010.parent / "c009_integrated_experiment"
C009_SCRIPTS = C009 / "scripts"
if str(C009_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C009_SCRIPTS))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c009_evaluation_common as common
import g11_official_model_factory as factory
import g11_training_stream as training_stream
from merit_v2_model import MeritV2, PromptResidualCorrector


DEFAULT_PROTOCOL = C010 / "protocol" / "pilot_v1.json"
DEFAULT_RUN = C010 / "runs" / "pilot_v1_nafnet_seed0"
G11_CONTRACT = C009 / "protocol" / "g11_training_contract_v2.json"


class PilotError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PilotError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".c010.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def prompt_inputs(degraded: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    blur = np.empty_like(degraded, dtype=np.float32)
    noise = np.empty_like(degraded, dtype=np.float32)
    for index, sample in enumerate(metadata["samples"]):
        blur[index].fill(
            (float(sample["sigma_h_px"]) + float(sample["sigma_v_px"])) / 6.0
        )
        noise[index].fill(float(sample["noise_std_8bit"]) / 3.0)
    return np.stack([degraded, blur, noise], axis=1).astype(np.float32, copy=False)


def load_holdout(contract: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, str]]]:
    bank_path = common.verify_asset(contract["assets"]["d0_bank"], "D0 bank")
    manifest_path = common.verify_asset(
        contract["assets"]["d0_manifest"], "D0 manifest"
    )
    bank = np.load(bank_path, allow_pickle=False)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = [
        int(row["bank_index"])
        for row in rows
        if row["split"] == "synthetic_holdout"
    ]
    if len(indices) != 50:
        raise PilotError(f"Expected 50 development rows, found {len(indices)}")
    selected_rows = [rows[index] for index in indices]
    return bank[np.asarray(indices)].astype(np.float32) / np.float32(255.0), selected_rows


def deterministic_rng(
    seed: int, bank_index: int, condition_index: int
) -> np.random.RandomState:
    mixed = (seed * 1_000_003 + bank_index * 1009 + condition_index * 9176) % (
        2**32 - 1
    )
    return np.random.RandomState(mixed)


def degrade(
    clean: np.ndarray,
    condition: dict[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    output = gaussian_filter(
        clean,
        sigma=(float(condition["sigma_v_px"]), float(condition["sigma_h_px"])),
        mode="reflect",
    ).astype(np.float32)
    output += rng.randn(*clean.shape).astype(np.float32) * np.float32(
        float(condition["noise_std_8bit"]) / 255.0
    )
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def development_samples(
    contract: dict[str, Any], bank: np.ndarray, rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    conditions = contract["controlled_d0"]["in_distribution_conditions"]
    seeds = contract["controlled_d0"]["noise_seeds"]
    for clean, row in zip(bank, rows):
        bank_index = int(row["bank_index"])
        for condition_index, condition in enumerate(conditions):
            for seed in seeds:
                samples.append(
                    {
                        "sample_id": f"id__b{bank_index}__{condition['id']}__n{seed}",
                        "domain": "id",
                        "parent_sha256": row["parent_sha256"],
                        "condition": condition["id"],
                        "clean": clean,
                        "degraded": degrade(
                            clean,
                            condition,
                            deterministic_rng(int(seed), bank_index, condition_index),
                        ),
                        "p_b": (
                            float(condition["sigma_h_px"])
                            + float(condition["sigma_v_px"])
                        )
                        / 6.0,
                        "p_n": float(condition["noise_std_8bit"]) / 3.0,
                    }
                )
        samples.append(
            {
                "sample_id": f"identity__b{bank_index}",
                "domain": "identity",
                "parent_sha256": row["parent_sha256"],
                "condition": "identity",
                "clean": clean,
                "degraded": clean.copy(),
                "p_b": 0.0,
                "p_n": 0.0,
            }
        )
    return samples


def psnr(clean: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((clean.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))


def gradient_mae(clean: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        0.5
        * (
            np.mean(np.abs(np.diff(clean, axis=1) - np.diff(candidate, axis=1)))
            + np.mean(np.abs(np.diff(clean, axis=0) - np.diff(candidate, axis=0)))
        )
    )


def edge_normal_overshoot_p95(clean: np.ndarray, candidate: np.ndarray) -> float:
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 1e-6]
    if positive.size == 0:
        return 0.0
    mask = magnitude >= float(np.percentile(positive, 75.0))
    yy, xx = np.indices(clean.shape, dtype=np.float32)
    nx = gx / np.maximum(magnitude, 1e-6)
    ny = gy / np.maximum(magnitude, 1e-6)
    plus_x, plus_y = xx + nx * 1.5, yy + ny * 1.5
    minus_x, minus_y = xx - nx * 1.5, yy - ny * 1.5
    clean_plus = cv2.remap(
        clean, plus_x, plus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    clean_minus = cv2.remap(
        clean, minus_x, minus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    cand_plus = cv2.remap(
        candidate, plus_x, plus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    cand_minus = cv2.remap(
        candidate, minus_x, minus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    lower = np.minimum(clean_plus, clean_minus)
    upper = np.maximum(clean_plus, clean_minus)
    overshoot = np.maximum.reduce(
        [
            cand_plus - upper,
            lower - cand_plus,
            cand_minus - upper,
            lower - cand_minus,
            np.zeros_like(clean),
        ]
    )
    return float(np.percentile(overshoot[mask], 95.0))


def build_model(
    protocol: dict[str, Any], device: torch.device
) -> tuple[MeritV2, dict[str, Any]]:
    base_spec = protocol["base"]
    if base_spec["family"] != "nafnet" or int(base_spec["seed"]) != 0:
        raise PilotError("pilot_v1 is bound to NAFNet base seed 0")
    base, metadata = factory.build_official_adapter("nafnet")
    checkpoint_path = (C010 / base_spec["checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["model"], strict=True)
    adapter = protocol["adapter"]
    corrector = PromptResidualCorrector(
        hidden_channels=int(adapter["hidden_channels"]),
        blocks=int(adapter["residual_blocks"]),
        max_delta=float(adapter["max_delta"]),
        blend_span=float(adapter["blend_span"]),
    )
    model = MeritV2(base, corrector).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    provenance = {
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "base_metadata": metadata,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }
    return model, provenance


def grad_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output_dx = output[:, :, :, 1:] - output[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    output_dy = output[:, :, 1:, :] - output[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (
        torch.mean(torch.abs(output_dx - target_dx))
        + torch.mean(torch.abs(output_dy - target_dy))
    )


def fft_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.fft.rfft2(output.float(), norm="ortho") - torch.fft.rfft2(
        target.float(), norm="ortho"
    )
    return torch.mean(torch.abs(difference))


def train(
    model: MeritV2,
    protocol: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> list[dict[str, Any]]:
    config = protocol["training"]
    g11 = load_json(G11_CONTRACT)
    bank, rows, _ = training_stream.verify_source_bank(g11)
    train_indices = np.asarray(
        [int(row["bank_index"]) for row in rows if row["split"] == "train"],
        dtype=np.int64,
    )
    stream = training_stream.latent_stream(
        bank,
        train_indices,
        int(config["seed"]),
        int(config["batch_size"]),
        g11["stream"]["degradation"],
    )
    optimizer = torch.optim.AdamW(
        model.corrector.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["steps"])
    )
    l1 = torch.nn.L1Loss()
    model.train()
    log: list[dict[str, Any]] = []
    accum = defaultdict(float)
    interval = 0
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        degraded, target, metadata = next(stream)
        inputs = torch.from_numpy(prompt_inputs(degraded, metadata)).to(device)
        target_tensor = torch.from_numpy(target).to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            output, _, residual, blend = model.forward_components(inputs)
        output_float = output.float()
        target_float = target_tensor.float()
        l1_value = l1(output_float, target_float)
        grad_value = grad_loss(output_float, target_float)
        fft_value = fft_loss(output_float, target_float)
        loss = l1_value + grad_value + 0.1 * fft_value
        if not torch.isfinite(loss):
            raise PilotError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.corrector.parameters(), 1.0)
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
        if step == 1 or step % 50 == 0 or step == int(config["steps"]):
            record = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
                "mean_since_last": {key: value / interval for key, value in accum.items()},
            }
            log.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
            accum = defaultdict(float)
            interval = 0
    log_path = run_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in log:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return log


def batches(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parent_bootstrap(
    records: list[dict[str, Any]], key: str, replicates: int
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["parent_sha256"])].append(float(record[key]))
    values = np.asarray([np.mean(group) for group in grouped.values()], dtype=np.float64)
    rng = np.random.RandomState(20261011)
    draws = rng.randint(0, len(values), size=(replicates, len(values)))
    boot = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "parents": int(len(values)),
    }


def evaluate(
    model: MeritV2,
    protocol: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    contract = common.load_contract()
    bank, rows = load_holdout(contract)
    samples = development_samples(contract, bank, rows)
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in batches(samples, 16):
            degraded = np.stack([sample["degraded"] for sample in batch]).astype(np.float32)
            blur = [float(sample["p_b"]) for sample in batch]
            noise = [float(sample["p_n"]) for sample in batch]
            inputs = torch.from_numpy(common.prompt_tensor(degraded, blur, noise)).to(device)
            with autocast(device):
                v2_preclip, base_preclip, residual, blend = model.forward_components(inputs)
            v2 = np.clip(v2_preclip.float().cpu().numpy()[:, 0], 0.0, 1.0)
            base = np.clip(base_preclip.float().cpu().numpy()[:, 0], 0.0, 1.0)
            residual_abs = residual.float().abs().mean(dim=(1, 2, 3)).cpu().numpy()
            blend_values = blend.float().flatten().cpu().numpy()
            for index, sample in enumerate(batch):
                clean = sample["clean"]
                source = sample["degraded"]
                base_psnr = psnr(clean, base[index])
                v2_psnr = psnr(clean, v2[index])
                base_grad = gradient_mae(clean, base[index])
                v2_grad = gradient_mae(clean, v2[index])
                base_over = edge_normal_overshoot_p95(clean, base[index])
                v2_over = edge_normal_overshoot_p95(clean, v2[index])
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "domain": sample["domain"],
                        "parent_sha256": sample["parent_sha256"],
                        "condition": sample["condition"],
                        "base_psnr_db": base_psnr,
                        "v2_psnr_db": v2_psnr,
                        "delta_psnr_db": v2_psnr - base_psnr,
                        "base_gradient_mae": base_grad,
                        "v2_gradient_mae": v2_grad,
                        "delta_gradient_mae": v2_grad - base_grad,
                        "base_overshoot_p95": base_over,
                        "v2_overshoot_p95": v2_over,
                        "delta_overshoot_p95": v2_over - base_over,
                        "v2_source_mean_abs": float(np.mean(np.abs(v2[index] - source))),
                        "v2_source_p99_abs": float(np.percentile(np.abs(v2[index] - source), 99.0)),
                        "adapter_residual_abs": float(residual_abs[index]),
                        "adapter_blend": float(blend_values[index]),
                    }
                )
    metric_path = run_dir / "development_metrics.jsonl"
    with metric_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    restoration = [record for record in records if record["domain"] == "id"]
    identity = [record for record in records if record["domain"] == "identity"]
    replicates = int(protocol["development_evaluation"]["bootstrap_replicates"])
    psnr_result = parent_bootstrap(restoration, "delta_psnr_db", replicates)
    gradient_result = parent_bootstrap(restoration, "delta_gradient_mae", replicates)
    overshoot_result = parent_bootstrap(restoration, "delta_overshoot_p95", replicates)
    identity_changes = np.asarray(
        [record["v2_source_mean_abs"] for record in identity], dtype=np.float64
    )
    identity_p99 = np.asarray(
        [record["v2_source_p99_abs"] for record in identity], dtype=np.float64
    )
    gates = {
        "psnr": psnr_result["mean"] >= 0.10 and psnr_result["ci_low"] > 0.0,
        "gradient": gradient_result["mean"] < 0.0,
        "identity": float(identity_changes.max()) <= 1e-7
        and float(identity_p99.max()) <= 1e-7,
        "overshoot": overshoot_result["mean"] <= 0.0,
    }
    result = {
        "schema_version": "MERIT-C010-V2-PILOT-RESULT-v1",
        "status": "PASS_DEVELOPMENT_TARGETS" if all(gates.values()) else "ITERATE",
        "scientific_status": "EXPLORATORY_UNBLINDED_DEVELOPMENT_ONLY",
        "records": len(records),
        "restoration_records": len(restoration),
        "identity_records": len(identity),
        "delta_psnr_db": psnr_result,
        "delta_gradient_mae": gradient_result,
        "delta_overshoot_p95": overshoot_result,
        "identity": {
            "max_mean_absolute_change": float(identity_changes.max()),
            "max_p99_absolute_change": float(identity_p99.max()),
        },
        "gates": gates,
        "metrics_path": str(metric_path.resolve()),
        "metrics_sha256": sha256_file(metric_path),
    }
    write_json_atomic(run_dir / "development_summary.json", result)
    return result


def run(protocol_path: Path, run_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise PilotError("CUDA is required for the MERIT v2 pilot")
    complete_path = run_dir / "RUN_COMPLETE.json"
    if complete_path.exists():
        result = load_json(complete_path)
        result["status"] = "ALREADY_COMPLETE"
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_json(protocol_path)
    if protocol.get("status") != "EXPLORATORY_DEVELOPMENT_ONLY":
        raise PilotError("The pilot protocol is not marked exploratory")
    seed = int(protocol["training"]["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, provenance = build_model(protocol, device)
    started = time.perf_counter()
    train_log = train(model, protocol, device, run_dir)
    checkpoint_path = run_dir / "final.pt"
    torch.save(
        {
            "corrector": model.corrector.state_dict(),
            "protocol_sha256": sha256_file(protocol_path),
            "base_checkpoint_sha256": provenance["base_checkpoint_sha256"],
            "steps": int(protocol["training"]["steps"]),
        },
        checkpoint_path,
    )
    development = evaluate(model, protocol, device, run_dir)
    result = {
        "schema_version": "MERIT-C010-V2-PILOT-COMPLETE-v1",
        "status": "PASS" if development["status"] == "PASS_DEVELOPMENT_TARGETS" else "ITERATE",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scientific_status": "EXPLORATORY_UNBLINDED_DEVELOPMENT_ONLY",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_sha256": sha256_file(HERE / "merit_v2_model.py"),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "provenance": provenance,
        "training_log_records": len(train_log),
        "development": development,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    write_json_atomic(complete_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.protocol.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
