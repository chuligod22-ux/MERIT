"""Fine-tune the NAFNet branch for signal-dependent Poisson-Gaussian noise."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402


RUN_DIR = common.C018 / "runs" / "development_v1" / "poisson_nafnet_seed0"


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def gradient_l1(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidate_dx = candidate[:, :, :, 1:] - candidate[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    candidate_dy = candidate[:, :, 1:, :] - candidate[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (
        functional.l1_loss(candidate_dx, target_dx)
        + functional.l1_loss(candidate_dy, target_dy)
    )


def save_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_training_bank() -> np.ndarray:
    bank = np.load(
        common.C006 / "artifacts" / "d0_bank_448.npy", allow_pickle=False
    )
    with (common.C006 / "artifacts" / "d0_bank_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    indices = [
        int(row["bank_index"]) for row in rows if row["split"] == "train"
    ]
    if len(indices) != 161:
        raise RuntimeError(f"Expected 161 training crops, received {len(indices)}")
    return bank[np.asarray(indices)].astype(np.float32) / np.float32(255.0)


def augment(clean: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    output = clean
    rotations = int(rng.randint(0, 4))
    if rotations:
        output = np.rot90(output, rotations)
    if bool(rng.randint(0, 2)):
        output = np.flip(output, axis=0)
    if bool(rng.randint(0, 2)):
        output = np.flip(output, axis=1)
    return np.ascontiguousarray(output, dtype=np.float32)


def build_model(device: torch.device):
    evaluation_common = common.import_file(
        "c009_evaluation_common_c018_train",
        common.C009 / "scripts" / "c009_evaluation_common.py",
    )
    model, provenance = evaluation_common.build_and_load_model(
        {"kind": "wp1", "family": "nafnet", "arm": "base", "seed": 0},
        device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    return model, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = common.load_json(common.DEVELOPMENT_PROTOCOL)
    training = protocol["poisson_training"]
    steps = int(args.steps or training["steps"])
    batch_size = int(args.batch_size or training["batch_size"])
    seed = int(training["seed"])
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    completion = RUN_DIR / "RUN_COMPLETE.json"
    if completion.exists() and steps == int(training["steps"]):
        print(completion.read_text(encoding="utf-8"))
        return

    device = torch.device("cuda")
    set_determinism(seed)
    rng = np.random.RandomState(seed)
    bank = load_training_bank()
    model, provenance = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    log_path = RUN_DIR / ("train.jsonl" if steps == int(training["steps"]) else f"pilot_{steps}.jsonl")
    log_path.write_text("", encoding="utf-8")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    peak_range = [float(value) for value in training["peak_photons_range"]]
    additive_range = [float(value) for value in training["additive_std_8bit_range"]]

    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        for step in range(1, steps + 1):
            indices = rng.randint(0, len(bank), size=batch_size)
            clean_np = np.stack([augment(bank[index], rng) for index in indices])
            clean = torch.from_numpy(clean_np[:, None]).to(device)
            peaks = torch.empty((batch_size, 1, 1, 1), device=device).uniform_(
                peak_range[0], peak_range[1]
            )
            additive = torch.empty((batch_size, 1, 1, 1), device=device).uniform_(
                additive_range[0] / 255.0, additive_range[1] / 255.0
            )
            degraded = torch.poisson(torch.clamp(clean, 0.0, 1.0) * peaks) / peaks
            degraded = torch.clamp(degraded + torch.randn_like(clean) * additive, 0.0, 1.0)
            zeros = torch.zeros_like(degraded)
            inputs = torch.cat([degraded, zeros, zeros], dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.forward_preclip(inputs)
            output_float = output.float()
            mse = functional.mse_loss(output_float, clean)
            l1 = functional.l1_loss(output_float, clean)
            gradient = gradient_l1(output_float, clean)
            loss = mse + 0.02 * l1 + 0.05 * gradient
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step == 1 or step % args.log_every == 0 or step == steps:
                torch.cuda.synchronize(device)
                record = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "l1": float(l1.detach().cpu()),
                    "gradient": float(gradient.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                log_handle.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % 1000 == 0 or step == steps:
                checkpoint_name = (
                    "final.pt" if step == steps else f"step_{step:05d}.pt"
                )
                save_atomic(
                    RUN_DIR / checkpoint_name,
                    {
                        "schema_version": "MERIT-C018-POISSON-BRANCH-v1",
                        "model": model.state_dict(),
                        "step": step,
                        "protocol_sha256": common.sha256_file(common.DEVELOPMENT_PROTOCOL),
                        "source_checkpoint_sha256": provenance["checkpoint_sha256"],
                    },
                )

    final_path = RUN_DIR / "final.pt"
    result = {
        "schema_version": "MERIT-C018-POISSON-TRAINING-COMPLETE-v1",
        "status": "PASS",
        "scientific_status": "UNBLINDED_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "steps": steps,
        "batch_size": batch_size,
        "training_crops": len(bank),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint_path": str(final_path.resolve()),
        "checkpoint_sha256": common.sha256_file(final_path),
        "protocol_sha256": common.sha256_file(common.DEVELOPMENT_PROTOCOL),
        "source": provenance,
        "log_path": str(log_path.resolve()),
        "log_sha256": common.sha256_file(log_path),
    }
    if steps == int(training["steps"]):
        common.write_json_atomic(completion, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
