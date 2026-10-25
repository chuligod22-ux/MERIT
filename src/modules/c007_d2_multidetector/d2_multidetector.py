#!/usr/bin/env python3
"""Train and evaluate a frozen multi-detector panel for MERIT D2 sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol.json"
SCHEMA = "MERIT-C007-D2-MULTIDETECTOR-v1"
CONDITIONS = ("reference", "degraded", "task_off", "task_on")
INDEPENDENT = (
    "d2_resnet34_unet",
    "d2_vgg16_unet",
    "d2_deeplabv3_resnet50",
)
LEGACY = "legacy_deepcrack_vgg16_unet"
ALL_DETECTORS = (*INDEPENDENT, LEGACY)
METRICS = ("pixel_auprc", "f1_at_0_5")


class MultiDetectorError(RuntimeError):
    """Frozen protocol, data, model, or result contract was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise MultiDetectorError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise MultiDetectorError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MultiDetectorError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise MultiDetectorError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "MERIT-D2-MULTIDETECTOR-SYNTHETIC-POSTHOC-20260923":
        raise MultiDetectorError("Unexpected protocol ID")
    if tuple(protocol["detectors"]["independent_panel"]) != INDEPENDENT:
        raise MultiDetectorError("Independent detector panel changed")
    if protocol["detectors"]["coupled_reference"] != LEGACY:
        raise MultiDetectorError("Legacy reference changed")
    if protocol["training"]["effective_batch_size"] != 64:
        raise MultiDetectorError("Effective batch size must remain 64")
    if protocol["training"]["epochs"] != 12:
        raise MultiDetectorError("Epoch count must remain 12")
    if [row["chain"] for row in protocol["detectors"]["seed_pairing"]] != [0, 1, 2]:
        raise MultiDetectorError("Seed pairing changed")
    return protocol


def verify_static_assets(protocol: Mapping[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for label, expected in protocol["sha256"].items():
        verified[label] = require_hash(Path(protocol["paths"][label]), expected, label)
    return verified


def load_modules(protocol: Mapping[str, Any]):
    # Windows DataLoader uses spawn and must be able to import Dataset classes by
    # their real module names in each worker. Keep these source directories on
    # sys.path and use import_module rather than ephemeral spec-only aliases.
    module_paths = {
        "d2_detector": Path(protocol["paths"]["d2_code"]),
        "d2_tunnel_cv": Path(protocol["paths"]["cv_code"]),
        "d2_external_synthetic": Path(protocol["paths"]["external_code"]),
    }
    for path in module_paths.values():
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    d2 = importlib.import_module("d2_detector")
    cv = importlib.import_module("d2_tunnel_cv")
    external = importlib.import_module("d2_external_synthetic")
    legacy = import_module(
        Path(protocol["paths"]["legacy_architecture_code"]), "merit_multi_legacy"
    )
    external_protocol = json.loads(
        Path(protocol["paths"]["external_protocol"]).read_text(encoding="utf-8")
    )
    _, _, restorer = external.load_source_modules(external_protocol)
    return d2, cv, external, legacy, restorer, external_protocol


class OutputAdapter(nn.Module):
    """Make torchvision segmentation models expose one tensor of logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.model(inputs)
        if isinstance(output, dict):
            output = output["out"]
        return output


def create_vgg16_unet(
    legacy_module: Any, imagenet_path: Path | None
) -> tuple[nn.Module, str | None]:
    model = legacy_module.UNet16(pretrained=False)
    digest = None
    if imagenet_path is not None:
        state = torch.load(imagenet_path, map_location="cpu", weights_only=True)
        feature_state = {
            key[len("features.") :]: value
            for key, value in state.items()
            if key.startswith("features.")
        }
        model.encoder.load_state_dict(feature_state, strict=True)
        digest = sha256_file(imagenet_path)
    return OutputAdapter(model), digest


def create_deeplabv3(resnet_path: Path | None) -> tuple[nn.Module, str | None]:
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=1,
        aux_loss=False,
    )
    digest = None
    if resnet_path is not None:
        state = torch.load(resnet_path, map_location="cpu", weights_only=True)
        backbone_state = {
            key: value for key, value in state.items() if not key.startswith("fc.")
        }
        model.backbone.load_state_dict(backbone_state, strict=True)
        digest = sha256_file(resnet_path)
    return OutputAdapter(model), digest


def create_train_model(
    architecture: str,
    protocol: Mapping[str, Any],
    legacy_module: Any,
    *,
    initialize_encoder: bool,
) -> tuple[nn.Module, str | None]:
    if architecture == "d2_vgg16_unet":
        path = Path(protocol["paths"]["vgg16_imagenet"]) if initialize_encoder else None
        return create_vgg16_unet(legacy_module, path)
    if architecture == "d2_deeplabv3_resnet50":
        path = Path(protocol["paths"]["resnet50_imagenet"]) if initialize_encoder else None
        return create_deeplabv3(path)
    raise MultiDetectorError(f"Unsupported train architecture: {architecture}")


def load_legacy_detector(
    protocol: Mapping[str, Any], legacy_module: Any, device: torch.device
) -> nn.Module:
    model, _ = create_vgg16_unet(legacy_module, None)
    checkpoint = torch.load(
        Path(protocol["paths"]["legacy_deepcrack_checkpoint"]),
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise MultiDetectorError("Legacy DeepCrack checkpoint lacks model state")
    model.model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_freeze(protocol_path: Path, output: Path) -> dict[str, Any]:
    freeze_path = output / "freeze.json"
    if not freeze_path.is_file():
        raise MultiDetectorError("Run audit before smoke/train/evaluate")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require_hash(protocol_path, freeze["protocol_sha256"], "frozen protocol")
    require_hash(Path(freeze["code_path"]), freeze["code_sha256"], "frozen code")
    require_hash(
        Path(freeze["cohort_path"]), freeze["cohort_sha256"], "frozen cohort"
    )
    return freeze


def collect_training_pairs(protocol: Mapping[str, Any], d2: Any, cv: Any):
    cv_protocol = json.loads(
        Path(protocol["paths"]["cv_protocol"]).read_text(encoding="utf-8")
    )
    frozen_protocol = json.loads(
        Path(protocol["paths"]["d2_protocol"]).read_text(encoding="utf-8")
    )
    pairs = cv.collect_parent_pairs(
        Path(protocol["paths"]["d2_root"]), cv_protocol, frozen_protocol
    )
    return cv_protocol, frozen_protocol, pairs


def command_audit(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    assets = verify_static_assets(protocol)
    d2, cv, external, _, _, _ = load_modules(protocol)
    cv_protocol, _, parent_pairs = collect_training_pairs(protocol, d2, cv)
    cohort_path = Path(protocol["paths"]["external_cohort"])
    cohort = read_csv(cohort_path)
    if len(cohort) != 1703:
        raise MultiDetectorError(f"Expected 1,703 cohort rows, observed {len(cohort)}")
    external.verify_cohort_files(cohort)
    counts = {
        tunnel: sum(len(parent_pairs[parent]) for parent in parents)
        for tunnel, parents in cv_protocol["tunnels"].items()
    }
    expected = cv_protocol["expected_base_tiles_after_quarantine"]
    if any(counts[tunnel] != int(expected[tunnel]) for tunnel in counts):
        raise MultiDetectorError(f"Training tile counts changed: {counts}")
    output = args.output_dir.resolve()
    freeze = {
        "schema_version": SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "created_utc": utc_now(),
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "static_assets": assets,
        "cohort_path": str(cohort_path),
        "cohort_sha256": sha256_file(cohort_path),
        "cohort_tiles": len(cohort),
        "cohort_parents": len({row["parent"] for row in cohort}),
        "cohort_tunnels": len({row["tunnel"] for row in cohort}),
        "training_tiles_by_heldout_tunnel": {
            tunnel: sum(counts.values()) - counts[tunnel] for tunnel in counts
        },
        "detector_panel": list(ALL_DETECTORS),
    }
    write_json_exclusive(output / "freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2), flush=True)
    return 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    d2: Any,
    frozen_protocol: Mapping[str, Any],
    accumulation: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_items = 0
    batches = len(loader)
    remainder = batches % accumulation
    for batch_index, (images, masks, _, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        group_size = (
            remainder
            if remainder and batch_index > batches - remainder
            else accumulation
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images).float()
            loss = d2.loss_from_protocol(logits, masks, frozen_protocol)
        (loss / group_size).backward()
        if batch_index % accumulation == 0 or batch_index == batches:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        count = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_items += count
    return total_loss / total_items


def train_one_run(
    architecture: str,
    fold: str,
    seed: int,
    output: Path,
    protocol: Mapping[str, Any],
    d2: Any,
    cv_protocol: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    parent_pairs: Mapping[str, Sequence[Any]],
    legacy_module: Any,
) -> dict[str, Any]:
    run_dir = output / "training" / architecture / f"fold_{fold}" / f"seed_{seed}"
    final_path = run_dir / "run_final.json"
    if final_path.is_file():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        checkpoint = run_dir / payload["checkpoint"]
        require_hash(checkpoint, payload["checkpoint_sha256"], "existing checkpoint")
        return payload
    if run_dir.exists() and any(run_dir.iterdir()):
        raise MultiDetectorError(f"Incomplete nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    held = set(cv_protocol["tunnels"][fold])
    train_pairs = [
        pair
        for parent, pairs in parent_pairs.items()
        if parent not in held
        for pair in pairs
    ]
    d2.seed_everything(seed)
    device = d2.require_cuda()
    model, encoder_hash = create_train_model(
        architecture, protocol, legacy_module, initialize_encoder=True
    )
    model = model.to(device)
    dataset = d2.D2Tiles(
        train_pairs,
        tile_size=512,
        rotation_quadrants=(0, 1, 2, 3),
        seed=seed,
    )
    loader = d2.make_loader(
        dataset,
        int(protocol["training"]["physical_batch_size"]),
        int(protocol["training"]["workers"]),
        True,
        seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    start = {
        "schema_version": SCHEMA,
        "architecture": architecture,
        "fold": fold,
        "seed": seed,
        "heldout_parents": sorted(held),
        "training_tiles": len(train_pairs),
        "encoder_sha256": encoder_hash,
        "started_utc": utc_now(),
        "gpu": torch.cuda.get_device_name(device),
    }
    write_json_exclusive(run_dir / "run_start.json", start)
    started = time.perf_counter()
    epochs: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    with (run_dir / "train.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for epoch in range(1, int(protocol["training"]["epochs"]) + 1):
            dataset.set_epoch(epoch)
            epoch_start = time.perf_counter()
            loss = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                d2,
                frozen_protocol,
                int(protocol["training"]["gradient_accumulation"]),
            )
            row = {
                "epoch": epoch,
                "train_loss": loss,
                "elapsed_seconds": time.perf_counter() - epoch_start,
            }
            epochs.append(row)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps({**start, **row}, sort_keys=True), flush=True)
    checkpoint_path = run_dir / "final_epoch.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "fold": fold,
            "seed": seed,
            "epoch": int(protocol["training"]["epochs"]),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "code_sha256": sha256_file(Path(__file__)),
            "encoder_sha256": encoder_hash,
        },
        checkpoint_path,
    )
    final = {
        **start,
        "epochs": epochs,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "completed_utc": utc_now(),
    }
    write_json_exclusive(final_path, final)
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return final


def command_smoke(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    load_freeze(args.protocol, args.output_dir.resolve())
    verify_static_assets(protocol)
    d2, cv, _, legacy, _, _ = load_modules(protocol)
    _, frozen_protocol, parent_pairs = collect_training_pairs(protocol, d2, cv)
    sample_pairs = [pair for pairs in parent_pairs.values() for pair in pairs][:4]
    dataset = d2.D2Tiles(sample_pairs, 512, (0,), 20260921)
    loader = d2.make_loader(dataset, 4, 0, False, 20260921)
    images, masks, _, _ = next(iter(loader))
    device = d2.require_cuda()
    results: dict[str, Any] = {}
    for architecture in protocol["training"]["architectures_to_train"]:
        model, encoder_hash = create_train_model(
            architecture, protocol, legacy, initialize_encoder=True
        )
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images.to(device)).float()
            loss = d2.loss_from_protocol(logits, masks.to(device), frozen_protocol)
        loss.backward()
        optimizer.step()
        results[architecture] = {
            "shape": list(logits.shape),
            "loss": float(loss.detach().cpu()),
            "encoder_sha256": encoder_hash,
        }
        del model, optimizer, logits, loss
        torch.cuda.empty_cache()
    legacy_model = load_legacy_detector(protocol, legacy, device)
    with torch.inference_mode():
        legacy_shape = list(legacy_model(images.to(device)) .shape)
    results[LEGACY] = {"shape": legacy_shape}
    payload = {"schema_version": SCHEMA, "status": "PASS", "models": results}
    write_json_exclusive(args.output_dir.resolve() / "smoke.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


def command_train(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    protocol = load_protocol(args.protocol)
    load_freeze(args.protocol, output)
    verify_static_assets(protocol)
    d2, cv, _, legacy, _, _ = load_modules(protocol)
    cv_protocol, frozen_protocol, parent_pairs = collect_training_pairs(protocol, d2, cv)
    total = len(protocol["training"]["architectures_to_train"]) * 4 * 3
    completed = 0
    started = time.perf_counter()
    for architecture in protocol["training"]["architectures_to_train"]:
        for fold in protocol["training"]["folds"]:
            for pairing in protocol["detectors"]["seed_pairing"]:
                train_one_run(
                    architecture,
                    fold,
                    int(pairing["detector_seed"]),
                    output,
                    protocol,
                    d2,
                    cv_protocol,
                    frozen_protocol,
                    parent_pairs,
                    legacy,
                )
                completed += 1
                elapsed = time.perf_counter() - started
                write_json_atomic(
                    output / "training_status.json",
                    {
                        "schema_version": SCHEMA,
                        "state": "TRAINING_COMPLETE" if completed == total else "RUNNING",
                        "completed_runs": completed,
                        "total_runs": total,
                        "elapsed_seconds_this_invocation": elapsed,
                        "projected_remaining_seconds": elapsed / completed * (total - completed),
                        "updated_utc": utc_now(),
                    },
                )
    print((output / "training_status.json").read_text(encoding="utf-8"), flush=True)
    return 0


def load_training_checkpoint(
    architecture: str,
    fold: str,
    seed: int,
    output: Path,
    protocol: Mapping[str, Any],
    legacy_module: Any,
    device: torch.device,
) -> nn.Module:
    run_dir = output / "training" / architecture / f"fold_{fold}" / f"seed_{seed}"
    final = json.loads((run_dir / "run_final.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / final["checkpoint"]
    require_hash(checkpoint_path, final["checkpoint_sha256"], f"{architecture} checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, _ = create_train_model(
        architecture, protocol, legacy_module, initialize_encoder=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def load_resnet34_checkpoint(
    fold: str,
    seed: int,
    external_protocol: Mapping[str, Any],
    d2: Any,
    device: torch.device,
) -> nn.Module:
    record = external_protocol["detectors"]["checkpoints"][fold][str(seed)]
    path = Path(record["path"])
    require_hash(path, record["sha256"], "ResNet34-UNet checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = d2.ResNet34UNet()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def load_restorers(
    restorer_seed: int,
    external_protocol: Mapping[str, Any],
    external: Any,
    restorer_module: Any,
    device: torch.device,
) -> dict[str, nn.Module]:
    source = external.load_restoration_source(external_protocol)
    result: dict[str, nn.Module] = {}
    with torch.serialization.safe_globals([TorchVersion]):
        for arm in ("task_off", "task_on"):
            record = external.restoration_record(source, arm, restorer_seed)
            path = Path(record["path"])
            require_hash(path, record["sha256"], f"{arm} restorer")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = restorer_module.NAFNetUNet(
                width=int(external_protocol["restoration"]["model_width"])
            )
            model.load_state_dict(checkpoint["state"], strict=True)
            result[arm] = model.to(device).eval()
    return result


def run_evaluation_chain(
    fold: str,
    pairing: Mapping[str, Any],
    output: Path,
    protocol: Mapping[str, Any],
    cohort: Sequence[Mapping[str, str]],
    d2: Any,
    cv: Any,
    external: Any,
    legacy_module: Any,
    restorer_module: Any,
    external_protocol: Mapping[str, Any],
    maximum_rows: int | None = None,
) -> dict[str, Any]:
    rows = [row for row in cohort if row["tunnel"] == fold]
    if maximum_rows is not None:
        rows = rows[:maximum_rows]
    device = external.require_cuda()
    seed = int(pairing["detector_seed"])
    detectors = {
        "d2_resnet34_unet": load_resnet34_checkpoint(
            fold, seed, external_protocol, d2, device
        ),
        "d2_vgg16_unet": load_training_checkpoint(
            "d2_vgg16_unet", fold, seed, output, protocol, legacy_module, device
        ),
        "d2_deeplabv3_resnet50": load_training_checkpoint(
            "d2_deeplabv3_resnet50",
            fold,
            seed,
            output,
            protocol,
            legacy_module,
            device,
        ),
        LEGACY: load_legacy_detector(protocol, legacy_module, device),
    }
    restorers = load_restorers(
        int(pairing["restorer_seed"]),
        external_protocol,
        external,
        restorer_module,
        device,
    )
    dataset = external.ExternalSyntheticDataset(rows, external_protocol)
    loader = DataLoader(
        dataset,
        batch_size=int(protocol["detectors"]["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(protocol["detectors"]["workers"]),
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=external.worker_seed,
        generator=torch.Generator().manual_seed(20260924 + int(pairing["chain"])),
    )
    bins = int(protocol["detectors"]["histogram_bins"])
    threshold = float(protocol["detectors"]["fixed_threshold"])
    accumulators: dict[str, dict[str, dict[str, Any]]] = {
        detector: defaultdict(
            lambda: {
                condition: external.Accumulator(bins) for condition in CONDITIONS
            }
        )
        for detector in ALL_DETECTORS
    }
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    total_batches = len(loader)
    processed = 0
    for batch_index, (clean_tensor, degraded_tensor, mask_tensor, parents, _) in enumerate(
        loader, start=1
    ):
        clean = clean_tensor.numpy()
        degraded = degraded_tensor.numpy()
        masks = mask_tensor.numpy()
        candidates = {
            "reference": clean,
            "degraded": degraded,
            "task_off": external.restore_batch(
                degraded_tensor, restorers["task_off"], device
            ),
            "task_on": external.restore_batch(
                degraded_tensor, restorers["task_on"], device
            ),
        }
        for detector_name, detector in detectors.items():
            for condition, candidate in candidates.items():
                probabilities = external.detect_batch(candidate, detector, device)
                for index, parent in enumerate(parents):
                    external.update_accumulator(
                        accumulators[detector_name][parent][condition],
                        probabilities[index : index + 1],
                        masks[index : index + 1],
                        clean[index : index + 1],
                        candidate[index : index + 1],
                        cv,
                        threshold,
                        False,
                    )
        processed += len(clean)
        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "chain": pairing["chain"],
                        "processed": processed,
                        "total": len(rows),
                        "batch": batch_index,
                        "batches": total_batches,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    parent_results = {
        detector: {
            parent: {
                condition: external.finalize_accumulator(accumulator, d2, cv)
                for condition, accumulator in conditions.items()
            }
            for parent, conditions in by_parent.items()
        }
        for detector, by_parent in accumulators.items()
    }
    result = {
        "schema_version": SCHEMA,
        "fold": fold,
        "pairing": dict(pairing),
        "tile_count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "detectors": parent_results,
    }
    del detectors, restorers, loader, dataset
    torch.cuda.empty_cache()
    return result


def command_evaluate(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    protocol = load_protocol(args.protocol)
    load_freeze(args.protocol, output)
    verify_static_assets(protocol)
    d2, cv, external, legacy, restorer, external_protocol = load_modules(protocol)
    cohort = read_csv(Path(protocol["paths"]["external_cohort"]))
    progress = output / "evaluation"
    progress.mkdir(parents=True, exist_ok=True)
    total = 12
    completed = 0
    started = time.perf_counter()
    for fold in ("A", "B", "C", "D"):
        for pairing in protocol["detectors"]["seed_pairing"]:
            path = progress / f"fold_{fold}_chain{pairing['chain']}.json"
            if not path.exists():
                result = run_evaluation_chain(
                    fold,
                    pairing,
                    output,
                    protocol,
                    cohort,
                    d2,
                    cv,
                    external,
                    legacy,
                    restorer,
                    external_protocol,
                )
                write_json_exclusive(path, result)
            completed += 1
            elapsed = time.perf_counter() - started
            write_json_atomic(
                output / "evaluation_status.json",
                {
                    "schema_version": SCHEMA,
                    "state": "EVALUATION_COMPLETE" if completed == total else "RUNNING",
                    "completed_chains": completed,
                    "total_chains": total,
                    "elapsed_seconds_this_invocation": elapsed,
                    "projected_remaining_seconds": elapsed / completed * (total - completed),
                    "updated_utc": utc_now(),
                },
            )
    print((output / "evaluation_status.json").read_text(encoding="utf-8"), flush=True)
    return 0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise MultiDetectorError("Cannot average empty values")
    return float(sum(values) / len(values))


def bootstrap_detector(
    parent_rows: Sequence[Mapping[str, Any]],
    detector: str,
    metric: str,
    left: str,
    right: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in parent_rows:
        if row["detector"] == detector:
            grouped[(row["tunnel"], row["parent"])][row["condition"]] = float(
                row[metric]
            )
    deltas: dict[str, list[float]] = defaultdict(list)
    for (tunnel, _), values in grouped.items():
        deltas[tunnel].append(values[left] - values[right])
    per_tunnel = {tunnel: mean(deltas[tunnel]) for tunnel in ("A", "B", "C", "D")}
    point = mean(per_tunnel.values())
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        values = []
        for tunnel in ("A", "B", "C", "D"):
            source = np.asarray(deltas[tunnel], dtype=np.float64)
            values.append(float(rng.choice(source, size=len(source), replace=True).mean()))
        samples[index] = np.mean(values)
    return {
        "point": point,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "positive_tunnels": sum(value > 0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def bootstrap_panel(
    parent_rows: Sequence[Mapping[str, Any]],
    metric: str,
    left: str,
    right: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in parent_rows:
        if row["detector"] in INDEPENDENT:
            values[(row["tunnel"], row["parent"])][row["detector"]][
                row["condition"]
            ] = float(row[metric])
    by_tunnel: dict[str, list[float]] = defaultdict(list)
    for (tunnel, _), detectors in values.items():
        if set(detectors) != set(INDEPENDENT):
            raise MultiDetectorError("Panel detector missing for a parent")
        by_tunnel[tunnel].append(
            mean(detectors[detector][left] - detectors[detector][right] for detector in INDEPENDENT)
        )
    per_tunnel = {tunnel: mean(by_tunnel[tunnel]) for tunnel in ("A", "B", "C", "D")}
    point = mean(per_tunnel.values())
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        tunnel_means = []
        for tunnel in ("A", "B", "C", "D"):
            source = np.asarray(by_tunnel[tunnel], dtype=np.float64)
            tunnel_means.append(float(rng.choice(source, size=len(source), replace=True).mean()))
        samples[index] = np.mean(tunnel_means)
    return {
        "point": point,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "positive_tunnels": sum(value > 0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def detector_gate(result: Mapping[str, Any]) -> bool:
    return bool(
        result["point"] > 0
        and result["ci_lower"] > 0
        and result["positive_tunnels"] >= 3
    )


def command_summarize(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    protocol = load_protocol(args.protocol)
    load_freeze(args.protocol, output)
    final_path = output / "multidetector_final.json"
    if final_path.exists():
        raise MultiDetectorError(f"Refusing to overwrite {final_path}")
    chain_rows: list[dict[str, Any]] = []
    for fold in ("A", "B", "C", "D"):
        for pairing in protocol["detectors"]["seed_pairing"]:
            path = output / "evaluation" / f"fold_{fold}_chain{pairing['chain']}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for detector, parents in payload["detectors"].items():
                for parent, conditions in parents.items():
                    for condition, metrics in conditions.items():
                        chain_rows.append(
                            {
                                "detector": detector,
                                "tunnel": fold,
                                "chain": int(pairing["chain"]),
                                "detector_seed": int(pairing["detector_seed"]),
                                "restorer_seed": int(pairing["restorer_seed"]),
                                "parent": parent,
                                "condition": condition,
                                "tile_count": metrics["tile_count"],
                                "pixel_auprc": metrics["pixel_auprc"],
                                "precision_at_0_5": metrics["precision_at_0_5"],
                                "recall_at_0_5": metrics["recall_at_0_5"],
                                "f1_at_0_5": metrics["f1_at_0_5"],
                            }
                        )
    write_csv_exclusive(output / "detector_parent_chain_metrics.csv", chain_rows)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in chain_rows:
        grouped[(row["detector"], row["tunnel"], row["parent"], row["condition"])].append(row)
    parent_rows: list[dict[str, Any]] = []
    for (detector, tunnel, parent, condition), rows in sorted(grouped.items()):
        parent_rows.append(
            {
                "detector": detector,
                "tunnel": tunnel,
                "parent": parent,
                "condition": condition,
                "chains": len(rows),
                "tile_count": rows[0]["tile_count"],
                "pixel_auprc": mean(float(row["pixel_auprc"]) for row in rows),
                "precision_at_0_5": mean(float(row["precision_at_0_5"]) for row in rows),
                "recall_at_0_5": mean(float(row["recall_at_0_5"]) for row in rows),
                "f1_at_0_5": mean(float(row["f1_at_0_5"]) for row in rows),
            }
        )
    write_csv_exclusive(output / "detector_parent_mean_metrics.csv", parent_rows)
    tunnel_rows: list[dict[str, Any]] = []
    for detector in ALL_DETECTORS:
        for tunnel in ("A", "B", "C", "D"):
            for condition in CONDITIONS:
                subset = [
                    row
                    for row in parent_rows
                    if row["detector"] == detector
                    and row["tunnel"] == tunnel
                    and row["condition"] == condition
                ]
                tunnel_rows.append(
                    {
                        "detector": detector,
                        "tunnel": tunnel,
                        "condition": condition,
                        "parents": len(subset),
                        **{
                            metric: mean(float(row[metric]) for row in subset)
                            for metric in (
                                "pixel_auprc",
                                "precision_at_0_5",
                                "recall_at_0_5",
                                "f1_at_0_5",
                            )
                        },
                    }
                )
    write_csv_exclusive(output / "detector_tunnel_macro_metrics.csv", tunnel_rows)
    equal_tunnel: dict[str, dict[str, dict[str, float]]] = {}
    for detector in ALL_DETECTORS:
        equal_tunnel[detector] = {}
        for condition in CONDITIONS:
            subset = [
                row
                for row in tunnel_rows
                if row["detector"] == detector and row["condition"] == condition
            ]
            equal_tunnel[detector][condition] = {
                metric: mean(float(row[metric]) for row in subset)
                for metric in (
                    "pixel_auprc",
                    "precision_at_0_5",
                    "recall_at_0_5",
                    "f1_at_0_5",
                )
            }
    resamples = int(protocol["statistics"]["bootstrap_resamples"])
    base_seed = int(protocol["statistics"]["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    definitions = (
        ("reference_minus_degraded", "reference", "degraded"),
        ("task_on_minus_degraded", "task_on", "degraded"),
        ("task_off_minus_degraded", "task_off", "degraded"),
        ("task_on_minus_task_off", "task_on", "task_off"),
    )
    for comparison_index, (name, left, right) in enumerate(definitions):
        comparisons[name] = {"detectors": {}, "panel": {}}
        for metric_index, metric in enumerate(METRICS):
            for detector_index, detector in enumerate(ALL_DETECTORS):
                comparisons[name]["detectors"].setdefault(detector, {})[metric] = (
                    bootstrap_detector(
                        parent_rows,
                        detector,
                        metric,
                        left,
                        right,
                        resamples=resamples,
                        seed=base_seed
                        + comparison_index * 100
                        + metric_index * 10
                        + detector_index,
                    )
                )
            comparisons[name]["panel"][metric] = bootstrap_panel(
                parent_rows,
                metric,
                left,
                right,
                resamples=resamples,
                seed=base_seed + comparison_index * 100 + metric_index * 10 + 9,
            )
    primary_detectors = comparisons["task_on_minus_degraded"]["detectors"]
    primary_panel = comparisons["task_on_minus_degraded"]["panel"]["pixel_auprc"]
    passing = {
        detector: detector_gate(primary_detectors[detector]["pixel_auprc"])
        for detector in INDEPENDENT
    }
    panel_gate = bool(
        primary_panel["point"] > 0
        and primary_panel["ci_lower"] > 0
        and sum(passing.values()) >= 2
        and primary_panel["positive_tunnels"] >= 3
    )
    task_panel = comparisons["task_on_minus_task_off"]["panel"]["pixel_auprc"]
    task_passing = {
        detector: detector_gate(
            comparisons["task_on_minus_task_off"]["detectors"][detector]["pixel_auprc"]
        )
        for detector in INDEPENDENT
    }
    task_gate = bool(
        task_panel["point"] > 0
        and task_panel["ci_lower"] > 0
        and sum(task_passing.values()) >= 2
        and task_panel["positive_tunnels"] >= 3
    )
    manipulation = {
        detector: detector_gate(
            comparisons["reference_minus_degraded"]["detectors"][detector]["pixel_auprc"]
        )
        for detector in INDEPENDENT
    }
    final = {
        "schema_version": SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "post_hoc": True,
        "equal_tunnel_macro": equal_tunnel,
        "comparisons": comparisons,
        "gates": {
            "degradation_manipulation_by_detector": manipulation,
            "primary_detector_gates": passing,
            "primary_panel_gate": panel_gate,
            "task_loss_detector_gates": task_passing,
            "task_loss_panel_gate": task_gate,
        },
        "verdict": "SUPPORTED_WITH_LIMITS" if panel_gate else "NOT_SUPPORTED",
        "task_loss_verdict": "SUPPORTED_WITH_LIMITS" if task_gate else "NOT_SUPPORTED",
        "claim_boundary": protocol["claim_boundary"],
        "artifacts": {
            "chain_csv_sha256": sha256_file(output / "detector_parent_chain_metrics.csv"),
            "parent_csv_sha256": sha256_file(output / "detector_parent_mean_metrics.csv"),
            "tunnel_csv_sha256": sha256_file(output / "detector_tunnel_macro_metrics.csv"),
            "freeze_sha256": sha256_file(output / "freeze.json"),
        },
        "completed_utc": utc_now(),
    }
    write_json_exclusive(final_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("audit", command_audit),
        ("smoke", command_smoke),
        ("train", command_train),
        ("evaluate", command_evaluate),
        ("summarize", command_summarize),
    ):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        command.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.protocol = args.protocol.resolve()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
