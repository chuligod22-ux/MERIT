#!/usr/bin/env python
"""Train the frozen C017 SegFormer and Mamba2 detector panel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
MODIFIED = C017.parent
CV_DIR = MODIFIED / "c005_sensitivity" / "d2_tunnel_cv"
FROZEN_D2_DIR = MODIFIED / "c005_freeze" / "d2_detector"
PROTOCOL_PATH = C017 / "protocol" / "c017_protocol_v1.json"
DEFAULT_OUTPUT = C017 / "runs" / "training_v1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CV_DIR))
sys.path.insert(0, str(FROZEN_D2_DIR))

import c017_models as models
import d2_detector as frozen


def import_path(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


cv = import_path("c017_d2_cv", CV_DIR / "d2_tunnel_cv.py")


class C017TrainingError(RuntimeError):
    """Raised when the frozen C017 training contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def save_torch_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(list(values)))


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("schema_version") != "MERIT-C017-PROTOCOL-v1":
        raise C017TrainingError("Unexpected protocol schema")
    if not protocol.get("frozen"):
        raise C017TrainingError("C017 protocol is not frozen")
    payload = dict(protocol)
    claimed = payload.pop("payload_sha256", None)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if claimed != actual:
        raise C017TrainingError("Protocol payload hash mismatch")
    if tuple(protocol["models"]) != models.MODEL_NAMES:
        raise C017TrainingError("Model list changed")
    return protocol


def completed_run(run_dir: Path) -> bool:
    final_path = run_dir / "run_final.json"
    if not final_path.is_file():
        return False
    final = load_json(final_path)
    checkpoint = run_dir / str(final.get("checkpoint", ""))
    return checkpoint.is_file() and sha256_file(checkpoint) == final.get("checkpoint_sha256")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    frozen_protocol: Mapping[str, Any],
    accumulation: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_items = 0
    total_batches = len(loader)
    for batch_index, (images, masks, _, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
        loss = cv.loss_for_cv(logits, masks, frozen_protocol)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise C017TrainingError("Non-finite training output")
        (loss / accumulation).backward()
        if batch_index % accumulation == 0 or batch_index == total_batches:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        count = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_items += count
    if not total_items:
        raise C017TrainingError("Training loader produced no samples")
    return total_loss / total_items


def run_one(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    model_name: str,
    fold: str,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    if model_name not in models.MODEL_NAMES:
        raise C017TrainingError(f"Unknown frozen model: {model_name}")
    if fold not in protocol["data"]["tunnels"]:
        raise C017TrainingError(f"Unknown fold: {fold}")
    if seed not in protocol["training"]["seeds"]:
        raise C017TrainingError(f"Unknown seed: {seed}")
    run_dir = output_root / model_name / f"fold_{fold}" / f"seed_{seed}"
    if completed_run(run_dir):
        result = load_json(run_dir / "run_final.json")
        print(json.dumps({"status": "SKIP_COMPLETE", "run": str(run_dir)}), flush=True)
        return result
    run_dir.mkdir(parents=True, exist_ok=True)

    cv_protocol = cv.load_protocol(Path(protocol["assets"]["cv_protocol"]["path"]))
    frozen_protocol = frozen.load_protocol(Path(protocol["assets"]["d2_protocol"]["path"]))
    d2_root = Path(protocol["data"]["d2_root"])
    parent_pairs = cv.collect_parent_pairs(d2_root, cv_protocol, frozen_protocol)
    held_parents = list(protocol["data"]["tunnels"][fold])
    held_set = set(held_parents)
    train_pairs = [
        pair
        for parent, pairs in parent_pairs.items()
        if parent not in held_set
        for pair in pairs
    ]
    if held_set & {pair.parent for pair in train_pairs}:
        raise C017TrainingError("Held-out tunnel leaked into training")

    frozen.seed_everything(seed)
    device = frozen.require_cuda()
    model = models.build_model(model_name, Path(protocol["assets"]["segformer_pretrained_dir"]))
    model = model.to(device)
    model_contract = models.validate_output(model, device, int(protocol["data"]["tile_size"]))
    configuration = protocol["training"]["models"][model_name]
    dataset = frozen.D2Tiles(
        train_pairs,
        tile_size=int(protocol["data"]["tile_size"]),
        rotation_quadrants=(0, 1, 2, 3),
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    protocol_sha = sha256_file(protocol_path)
    start = {
        "schema_version": "MERIT-C017-TRAIN-RUN-v1",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_file_sha256": protocol_sha,
        "training_code_sha256": sha256_file(Path(__file__)),
        "models_code_sha256": sha256_file(HERE / "c017_models.py"),
        "model": model_name,
        "model_contract": model_contract,
        "fold": fold,
        "seed": seed,
        "held_out_parents": held_parents,
        "training_parents": sorted({pair.parent for pair in train_pairs}),
        "training_tiles": len(train_pairs),
        "device_name": torch.cuda.get_device_name(device),
        "started_unix_time": time.time(),
        "claim_boundary": protocol["claim_boundary"],
    }
    start_path = run_dir / "run_start.json"
    if not start_path.exists():
        write_json_atomic(start_path, start)
    else:
        previous = load_json(start_path)
        for key in ("protocol_payload_sha256", "training_code_sha256", "models_code_sha256"):
            if previous.get(key) != start[key]:
                raise C017TrainingError(f"Resume provenance changed: {key}")

    latest_path = run_dir / "latest.pt"
    log_path = run_dir / "train.jsonl"
    epoch_rows: list[dict[str, Any]] = []
    start_epoch = 1
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        if latest.get("protocol_payload_sha256") != protocol["payload_sha256"]:
            raise C017TrainingError("Resume checkpoint protocol mismatch")
        model.load_state_dict(latest["model_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        epoch_rows = list(latest["epochs"])
        start_epoch = int(latest["epoch"]) + 1
    epochs = int(protocol["training"]["epochs"])
    started = time.perf_counter()
    log_mode = "a" if log_path.exists() else "x"
    with log_path.open(log_mode, encoding="utf-8", newline="\n") as log:
        for epoch in range(start_epoch, epochs + 1):
            dataset.set_epoch(epoch)
            # Recreate the shuffled loader from an epoch-specific seed.  A resumed
            # run then sees the exact same batches as an uninterrupted run.
            loader = frozen.make_loader(
                dataset,
                int(configuration["physical_batch_size"]),
                int(protocol["training"]["workers"]),
                True,
                seed + epoch * 10_000,
            )
            epoch_started = time.perf_counter()
            loss = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                frozen_protocol,
                int(configuration["gradient_accumulation"]),
            )
            row = {
                "epoch": epoch,
                "train_loss": loss,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            epoch_rows.append(row)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            save_torch_atomic(
                latest_path,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "epochs": epoch_rows,
                    "protocol_payload_sha256": protocol["payload_sha256"],
                },
            )
            print(json.dumps({"model": model_name, "fold": fold, "seed": seed, **row}), flush=True)

    checkpoint = run_dir / "final_epoch.pt"
    save_torch_atomic(
        checkpoint,
        {
            "model_state": model.state_dict(),
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "epoch": epochs,
            "protocol_payload_sha256": protocol["payload_sha256"],
            "models_code_sha256": sha256_file(HERE / "c017_models.py"),
        },
    )
    model.eval()
    parent_metrics = {
        parent: cv.evaluate_parent(
            model,
            parent_pairs[parent],
            parent,
            d2_root,
            cv_protocol,
            seed,
            device,
        )
        for parent in held_parents
    }
    macro = {
        metric: mean(row[metric] for row in parent_metrics.values())
        for metric in ("pixel_auprc_4096_bin", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5")
    }
    result = {
        **start,
        "epochs": epoch_rows,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "parent_metrics": parent_metrics,
        "tunnel_macro": macro,
        "completed_unix_time": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(run_dir / "run_final.json", result)
    latest_path.unlink(missing_ok=True)
    print(json.dumps({"status": "COMPLETE", "model": model_name, "fold": fold, "seed": seed, "tunnel_macro": macro}, indent=2), flush=True)
    return result


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    models_summary: dict[str, Any] = {}
    for model_name in models.MODEL_NAMES:
        rows = []
        for fold in protocol["data"]["tunnels"]:
            for seed in protocol["training"]["seeds"]:
                run_dir = output_root / model_name / f"fold_{fold}" / f"seed_{seed}"
                if not completed_run(run_dir):
                    raise C017TrainingError(f"Incomplete run: {run_dir}")
                run = load_json(run_dir / "run_final.json")
                rows.append({"fold": fold, "seed": seed, **run["tunnel_macro"]})
        per_fold = {
            fold: {
                metric: mean(row[metric] for row in rows if row["fold"] == fold)
                for metric in ("pixel_auprc_4096_bin", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5")
            }
            for fold in protocol["data"]["tunnels"]
        }
        models_summary[model_name] = {
            "runs": len(rows),
            "per_fold": per_fold,
            "four_tunnel_mean": {
                metric: mean(per_fold[fold][metric] for fold in per_fold)
                for metric in ("pixel_auprc_4096_bin", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5")
            },
        }
    summary = {
        "schema_version": "MERIT-C017-TRAIN-SUMMARY-v1",
        "status": "COMPLETE",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "models": models_summary,
    }
    payload = dict(summary)
    summary["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    write_json_atomic(output_root / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = value.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run-one")
    one.add_argument("--model", choices=models.MODEL_NAMES, required=True)
    one.add_argument("--fold", choices=tuple("ABCD"), required=True)
    one.add_argument("--seed", type=int, required=True)
    commands.add_parser("run-all")
    commands.add_parser("summarize")
    return value


def main() -> int:
    args = parser().parse_args()
    protocol = validate_protocol(args.protocol)
    output = args.output.resolve()
    if args.command == "run-one":
        run_one(protocol, args.protocol, args.model, args.fold, args.seed, output)
    elif args.command == "run-all":
        for model_name in models.MODEL_NAMES:
            for seed in protocol["training"]["seeds"]:
                for fold in protocol["data"]["tunnels"]:
                    run_one(protocol, args.protocol, model_name, fold, int(seed), output)
        summarize(protocol, output)
    else:
        print(json.dumps(summarize(protocol, output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
