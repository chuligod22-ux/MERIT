"""Evaluate one C051 restorer on the frozen four-tunnel detector panel."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import c051_common as common

C009_SCRIPTS = common.MODIFIED / "c009_integrated_experiment" / "scripts"
if str(C009_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C009_SCRIPTS))

import c009_d2_external_eval as d2_evaluation  # noqa: E402


def load_restorer(
    backbone: str, arm: str, seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    marker_path = common.run_dir(backbone, arm, seed) / "RUN_COMPLETE.json"
    if not marker_path.is_file():
        raise common.C051Error(f"Incomplete training cell: {marker_path.parent}")
    marker = common.load_json(marker_path)
    if common.payload_sha256(marker) != marker.get("payload_sha256"):
        raise common.C051Error(f"Training marker payload mismatch: {marker_path}")
    checkpoint = Path(str(marker["checkpoint"]))
    if common.sha256_file(checkpoint) != marker["checkpoint_sha256"]:
        raise common.C051Error(f"Training checkpoint hash mismatch: {checkpoint}")
    model = common.build_model(backbone, 3 if arm == "on" else 1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device).eval(), {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": marker["checkpoint_sha256"],
        "training_payload_sha256": marker["payload_sha256"],
    }


def restore_batch(
    degraded_tensor: torch.Tensor,
    model: torch.nn.Module,
    arm: str,
    device: torch.device,
) -> np.ndarray:
    degraded = degraded_tensor.numpy().astype(np.float32, copy=False) / np.float32(
        255.0
    )
    if arm == "on":
        blur = np.full_like(degraded, np.float32(0.5 * (2.6 + 2.4) / 3.0))
        noise = np.full_like(degraded, np.float32(2.5 / 3.0))
        inputs = np.stack([degraded, blur, noise], axis=1)
    else:
        inputs = degraded[:, None]
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        restored = model(torch.from_numpy(inputs).to(device))[:, 0]
    result = restored.float().cpu().numpy()
    if not np.all(np.isfinite(result)):
        raise common.C051Error("Non-finite restoration output")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def evaluate_fold(
    backbone: str,
    arm: str,
    seed: int,
    fold: str,
    model: torch.nn.Module,
    cohort: list[dict[str, str]],
    modules: tuple[Any, ...],
    device: torch.device,
) -> dict[str, Any]:
    multi, protocol, d2, cv, external, legacy, external_protocol = modules
    rows = [row for row in cohort if row["tunnel"] == fold]
    detector_seed = {0: 20260921, 1: 20260922, 2: 20260923}[seed]
    detectors = d2_evaluation.load_detectors(
        fold, detector_seed, multi, protocol, d2, legacy, external_protocol, device
    )
    dataset = external.ExternalSyntheticDataset(rows, external_protocol)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    bins = int(protocol["detectors"]["histogram_bins"])
    threshold = float(protocol["detectors"]["fixed_threshold"])
    accumulators: dict[str, Any] = {
        detector_name: collections.defaultdict(
            lambda: {
                condition: external.Accumulator(bins)
                for condition in ("degraded", "restored")
            }
        )
        for detector_name in detectors
    }
    started = time.perf_counter()
    processed = 0
    for batch_index, (
        clean_tensor,
        degraded_tensor,
        mask_tensor,
        parents,
        _,
    ) in enumerate(loader, start=1):
        clean = clean_tensor.numpy()
        degraded = degraded_tensor.numpy()
        masks = mask_tensor.numpy()
        restored = restore_batch(degraded_tensor, model, arm, device)
        for detector_name, detector in detectors.items():
            for condition, candidate in (
                ("degraded", degraded),
                ("restored", restored),
            ):
                probabilities = external.detect_batch(candidate, detector, device)
                for index, parent in enumerate(parents):
                    external.update_accumulator(
                        accumulators[detector_name][str(parent)][condition],
                        probabilities[index : index + 1],
                        masks[index : index + 1],
                        clean[index : index + 1],
                        candidate[index : index + 1],
                        cv,
                        threshold,
                        False,
                    )
        processed += len(clean)
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(
                json.dumps(
                    {
                        "model_id": common.model_id(backbone, arm, seed),
                        "fold": fold,
                        "processed": processed,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    parent_results = {
        detector_name: {
            parent: {
                condition: external.finalize_accumulator(accumulator, d2, cv)
                for condition, accumulator in conditions.items()
            }
            for parent, conditions in by_parent.items()
        }
        for detector_name, by_parent in accumulators.items()
    }
    matched_burden = {
        detector_name: {
            parent: d2_evaluation.matched_sensitivity_burden(
                conditions["restored"],
                parent_results[detector_name][parent]["degraded"],
            )
            for parent, conditions in by_parent.items()
        }
        for detector_name, by_parent in accumulators.items()
    }
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C051-EVALUATION-FOLD-v1",
        "status": "PASS",
        "model_id": common.model_id(backbone, arm, seed),
        "backbone": backbone,
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "detector_seed": detector_seed,
        "tile_count": len(rows),
        "detectors": parent_results,
        "matched_sensitivity_burden": matched_burden,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    del detectors, dataset, loader, accumulators
    torch.cuda.empty_cache()
    return payload


def execute(backbone: str, arm: str, seed: int) -> dict[str, Any]:
    if (
        backbone not in common.BACKBONES
        or arm not in common.ARMS
        or seed not in common.SEEDS
    ):
        raise common.C051Error("Invalid evaluation cell")
    if not torch.cuda.is_available():
        raise common.C051Error("CUDA is required")
    protocol_c051 = common.validate_protocol()
    modules = d2_evaluation.load_modules()
    multi, protocol, _, _, _, _, _ = modules
    cohort = multi.read_csv(Path(protocol["paths"]["external_cohort"]))
    if len(cohort) != 1703:
        raise common.C051Error(f"Unexpected cohort rows: {len(cohort)}")
    device = torch.device("cuda")
    model, provenance = load_restorer(backbone, arm, seed, device)
    output = common.EVALUATION / common.model_id(backbone, arm, seed)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fold_hashes: dict[str, str] = {}
    for fold in ("A", "B", "C", "D"):
        path = output / f"fold_{fold}.json"
        if path.is_file():
            existing = common.load_json(path)
            if common.payload_sha256(existing) != existing.get("payload_sha256"):
                raise common.C051Error(f"Existing evaluation payload mismatch: {path}")
        else:
            torch.cuda.reset_peak_memory_stats(device)
            common.write_json_atomic(
                path,
                evaluate_fold(
                    backbone, arm, seed, fold, model, cohort, modules, device
                ),
            )
        fold_hashes[fold] = common.sha256_file(path)
    result: dict[str, Any] = {
        "schema_version": "MERIT-C051-EVALUATION-RUN-v1",
        "status": "PASS_COMPLETE",
        "model_id": common.model_id(backbone, arm, seed),
        "backbone": backbone,
        "arm": arm,
        "seed": seed,
        "provenance": provenance,
        "protocol_payload_sha256": protocol_c051["payload_sha256"],
        "fold_sha256": fold_hashes,
        "tile_count": len(cohort),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
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
