"""Evaluate one C009 WP1 checkpoint on the frozen four-tunnel D2 panel."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
REVISION = C009.parent.parent
D2_MULTI = REVISION / "40_수정본" / "c007_d2_multidetector"
D2_EXTERNAL = REVISION / "40_수정본" / "c007_d2_external_synthetic"
DEFAULT_OUTPUT = C009 / "evaluation" / "wp5_external_four_tunnel"


def load_modules():
    for path in (D2_MULTI, D2_EXTERNAL):
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    multi = common.import_module("c009_d2_multidetector", D2_MULTI / "d2_multidetector.py")
    protocol = multi.load_protocol(D2_MULTI / "protocol.json")
    d2, cv, external, legacy, _, external_protocol = multi.load_modules(protocol)
    return multi, protocol, d2, cv, external, legacy, external_protocol


def restore_batch(
    degraded_tensor: torch.Tensor,
    model: torch.nn.Module,
    specification: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    degraded = degraded_tensor.numpy().astype(np.float32, copy=False)
    count = degraded.shape[0]
    if specification["arm"] == "merit_full":
        blur = np.full(count, 0.8333333333333334, dtype=np.float32)
        noise = np.full(count, 0.8333333333333334, dtype=np.float32)
    else:
        blur = np.zeros(count, dtype=np.float32)
        noise = np.zeros(count, dtype=np.float32)
    inputs = common.prompt_tensor(degraded, blur, noise)
    with torch.inference_mode(), common.autocast(device):
        output = model(torch.from_numpy(inputs).to(device))
    result = output[:, 0].float().cpu().numpy()
    if not np.all(np.isfinite(result)):
        raise common.EvaluationError("Non-finite D2 restoration output")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def load_detectors(
    fold: str,
    detector_seed: int,
    multi: Any,
    protocol: dict[str, Any],
    d2: Any,
    legacy: Any,
    external_protocol: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    training_root = D2_MULTI / "artifacts" / "multidetector_v2"
    return {
        "d2_resnet34_unet": multi.load_resnet34_checkpoint(
            fold, detector_seed, external_protocol, d2, device
        ),
        "d2_vgg16_unet": multi.load_training_checkpoint(
            "d2_vgg16_unet", fold, detector_seed, training_root, protocol, legacy, device
        ),
        "d2_deeplabv3_resnet50": multi.load_training_checkpoint(
            "d2_deeplabv3_resnet50", fold, detector_seed, training_root, protocol, legacy, device
        ),
    }


def matched_sensitivity_burden(
    restored: Any,
    degraded_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Match restored sensitivity to the frozen degraded threshold-0.5 target.

    Histogram bins are searched from the highest threshold downward.  The first
    bin attaining at least the degraded-input sensitivity is therefore the
    most specific available operating point that does not undershoot the
    frozen target.
    """

    target = float(degraded_metrics["recall_at_0_5"])
    positive = np.asarray(restored.positive, dtype=np.int64)
    negative = np.asarray(restored.negative, dtype=np.int64)
    positive_total = int(positive.sum())
    negative_total = int(negative.sum())
    if positive_total <= 0 or negative_total <= 0:
        raise common.EvaluationError("Matched-sensitivity burden needs both classes")
    true_positive = np.cumsum(positive[::-1], dtype=np.int64)
    false_positive = np.cumsum(negative[::-1], dtype=np.int64)
    recalls = true_positive.astype(np.float64) / positive_total
    eligible = np.flatnonzero(recalls >= target)
    if not len(eligible):
        raise common.EvaluationError("Restored score histogram cannot attain target sensitivity")
    reverse_index = int(eligible[0])
    threshold_bin = len(positive) - 1 - reverse_index
    restored_fpr = float(false_positive[reverse_index] / negative_total)
    degraded_fpr = float(degraded_metrics["fp"] / degraded_metrics["negative_pixels"])
    return {
        "target_degraded_sensitivity_at_0_5": target,
        "restored_achieved_sensitivity": float(recalls[reverse_index]),
        "restored_threshold_bin": threshold_bin,
        "restored_threshold_approx": float(threshold_bin / (len(positive) - 1)),
        "degraded_false_positive_pixel_rate_at_0_5": degraded_fpr,
        "restored_false_positive_pixel_rate_at_matched_sensitivity": restored_fpr,
        "delta_false_positive_pixel_rate_restored_minus_degraded": restored_fpr
        - degraded_fpr,
        "matching_rule": "highest histogram threshold attaining at least degraded recall@0.5",
    }


def run_fold(
    fold: str,
    specification: dict[str, Any],
    model: torch.nn.Module,
    cohort: list[dict[str, str]],
    multi: Any,
    protocol: dict[str, Any],
    d2: Any,
    cv: Any,
    external: Any,
    legacy: Any,
    external_protocol: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rows = [row for row in cohort if row["tunnel"] == fold]
    detector_seed = {0: 20260921, 1: 20260922, 2: 20260923}[int(specification["seed"])]
    detectors = load_detectors(
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
    for batch_index, (clean_tensor, degraded_tensor, mask_tensor, parents, _) in enumerate(loader, start=1):
        clean = clean_tensor.numpy()
        degraded = degraded_tensor.numpy()
        masks = mask_tensor.numpy()
        restored = restore_batch(degraded_tensor, model, specification, device)
        candidates = {"degraded": degraded, "restored": restored}
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
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(
                json.dumps(
                    {
                        "model_id": common.model_id(specification),
                        "fold": fold,
                        "processed": processed,
                        "total": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
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
            parent: matched_sensitivity_burden(
                conditions["restored"],
                parent_results[detector_name][parent]["degraded"],
            )
            for parent, conditions in by_parent.items()
        }
        for detector_name, by_parent in accumulators.items()
    }
    result = {
        "schema_version": "MERIT-C009-D2-EXTERNAL-FOLD-v1",
        "status": "PASS",
        "model_id": common.model_id(specification),
        "fold": fold,
        "restorer_seed": specification["seed"],
        "detector_seed": detector_seed,
        "tile_count": len(rows),
        "detectors": parent_results,
        "matched_sensitivity_burden": matched_burden,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    result["payload_sha256"] = hashlib.sha256(common.canonical_bytes(result)).hexdigest()
    del detectors, dataset, loader
    torch.cuda.empty_cache()
    return result


def run(specification: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if specification["kind"] != "wp1":
        raise common.EvaluationError("D2 external matrix is restricted to 18 WP1 models")
    contract = common.load_contract()
    common.verify_asset(contract["assets"]["d2_cohort"], "D2 cohort")
    common.verify_asset(contract["assets"]["d2_protocol"], "D2 protocol")
    common.verify_asset(
        contract["assets"]["d2_multidetector_protocol"], "D2 multidetector protocol"
    )
    for index, record in enumerate(contract["assets"]["d2_detector_checkpoints"]):
        common.verify_asset(record, f"D2 detector checkpoint {index}")
    multi, protocol, d2, cv, external, legacy, external_protocol = load_modules()
    cohort = multi.read_csv(Path(protocol["paths"]["external_cohort"]))
    if len(cohort) != int(contract["wp5_external_four_tunnel"]["cohort_rows"]):
        raise common.EvaluationError("D2 cohort coverage mismatch")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise common.EvaluationError("CUDA is required")
    model, provenance = common.build_and_load_model(specification, device)
    model_name = common.model_id(specification)
    output_dir = output_root / model_name
    marker_path = output_dir / "RUN_COMPLETE.json"
    if marker_path.exists():
        marker = common.load_json(marker_path)
        marker["status"] = "ALREADY_COMPLETE"
        return marker
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed: list[str] = []
    for fold in ("A", "B", "C", "D"):
        path = output_dir / f"fold_{fold}.json"
        if path.exists():
            existing = common.load_json(path)
            if existing.get("status") != "PASS" or existing.get("model_id") != model_name:
                raise common.EvaluationError(f"Invalid existing D2 fold: {path}")
        else:
            torch.cuda.reset_peak_memory_stats(device)
            result = run_fold(
                fold,
                specification,
                model,
                cohort,
                multi,
                protocol,
                d2,
                cv,
                external,
                legacy,
                external_protocol,
                device,
            )
            common.write_json_exclusive(path, result)
        completed.append(fold)
        common.write_json_atomic(
            output_dir / "status.json",
            {
                "schema_version": "MERIT-C009-D2-EXTERNAL-STATUS-v1",
                "status": "running" if len(completed) < 4 else "evaluation_complete",
                "model_id": model_name,
                "completed_folds": completed,
                "total_folds": 4,
                "elapsed_seconds_this_invocation": time.perf_counter() - started,
            },
        )
    fold_hashes = {fold: common.sha256_file(output_dir / f"fold_{fold}.json") for fold in completed}
    marker: dict[str, Any] = {
        "schema_version": "MERIT-C009-D2-EXTERNAL-RUN-COMPLETE-v1",
        "status": "PASS",
        "model_id": model_name,
        "specification": specification,
        "provenance": provenance,
        "contract_sha256": common.EXPECTED_CONTRACT_SHA256,
        "completed_folds": completed,
        "fold_sha256": fold_hashes,
        "tile_count": len(cohort),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    marker["payload_sha256"] = hashlib.sha256(common.canonical_bytes(marker)).hexdigest()
    common.write_json_atomic(marker_path, marker)
    return marker


def find_specification(model_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    matches = [spec for spec in common.wp1_specifications(contract) if common.model_id(spec) == model_name]
    if len(matches) != 1:
        raise common.EvaluationError(f"Unknown WP1 model id: {model_name}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = common.load_contract()
    result = run(find_specification(args.model_id, contract), args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
