"""Run one frozen C055 seed/backbone/tunnel confirmation unit."""

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

HERE = Path(__file__).resolve().parent
C055 = HERE.parent
MODIFIED = C055.parent
C051_SCRIPTS = MODIFIED / "c051_framework_dominance" / "scripts"
if str(C051_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C051_SCRIPTS))

import c051_common as common  # noqa: E402
import c051_evaluate_v2 as evaluation  # noqa: E402
import c051_v2_common as c051_v2  # noqa: E402

d2_evaluation = evaluation.d2_evaluation
FREEZE = C055 / "protocol" / "confirmation_freeze_v1.json"
SEEDS = (1, 2)
TUNNELS = ("A", "B", "C", "D")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_freeze() -> dict[str, Any]:
    payload = common.load_json(FREEZE)
    if common.payload_sha256(payload) != payload.get("payload_sha256"):
        raise common.C051Error("C055 freeze payload mismatch")
    for relative, expected in payload["code_sha256"].items():
        if sha256_file(C055 / relative) != expected:
            raise common.C051Error(f"C055 frozen code mismatch: {relative}")
    return payload


def selected_fusion(
    raw: np.ndarray, off: np.ndarray, on: np.ndarray, condition: str
) -> np.ndarray:
    if condition == "max3":
        return np.maximum(np.maximum(raw, off), on).astype(np.float32)
    if condition == "max_raw_off":
        return np.maximum(raw, off).astype(np.float32)
    if condition == "mean3":
        return ((raw + off + on) / np.float32(3.0)).astype(np.float32)
    raise common.C051Error(f"unfrozen C055 condition: {condition}")


def output_path(seed: int, backbone: str, fold: str) -> Path:
    return C055 / "runs" / f"confirmation_seed{seed}" / backbone / f"fold_{fold}.json"


def evaluate(seed: int, backbone: str, fold: str, freeze: dict[str, Any]) -> dict[str, Any]:
    common.validate_protocol()
    c051_v2.validate_protocol_v2()
    modules = d2_evaluation.load_modules()
    multi, protocol, d2, cv, external, legacy, external_protocol = modules
    cohort = multi.read_csv(Path(protocol["paths"]["external_cohort"]))
    rows = [row for row in cohort if row["tunnel"] == fold]
    device = torch.device("cuda")
    off_model, off_provenance = evaluation.load_restorer(backbone, "off", seed, device)
    on_model, on_provenance = evaluation.load_restorer(backbone, "on", seed, device)
    detector_seed = freeze["detector_seeds"][str(seed)]
    detectors = d2_evaluation.load_detectors(
        fold, detector_seed, multi, protocol, d2, legacy, external_protocol, device
    )
    dataset = external.ExternalSyntheticDataset(rows, external_protocol)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    bins = int(protocol["detectors"]["histogram_bins"])
    threshold = float(protocol["detectors"]["fixed_threshold"])
    accumulators: dict[str, Any] = {
        detector: collections.defaultdict(
            lambda: {condition: external.Accumulator(bins) for condition in ("raw", "off", "framework")}
        )
        for detector in detectors
    }
    selected = freeze["selection"][backbone][fold]["conditions"]
    started = time.perf_counter()
    processed = 0
    for batch_index, (clean_tensor, degraded_tensor, mask_tensor, parents, _) in enumerate(
        loader, start=1
    ):
        clean = clean_tensor.numpy()
        raw_image = degraded_tensor.numpy()
        masks = mask_tensor.numpy()
        off_image = evaluation.restore_batch(degraded_tensor, off_model, "off", device)
        on_image = evaluation.restore_batch(degraded_tensor, on_model, "on", device)
        for detector_name, detector in detectors.items():
            raw_probability = external.detect_batch(raw_image, detector, device)
            off_probability = external.detect_batch(off_image, detector, device)
            on_probability = external.detect_batch(on_image, detector, device)
            probabilities = {
                "raw": raw_probability,
                "off": off_probability,
                "framework": selected_fusion(
                    raw_probability, off_probability, on_probability, selected[detector_name]
                ),
            }
            for condition, probability in probabilities.items():
                for index, parent in enumerate(parents):
                    external.update_accumulator(
                        accumulators[detector_name][str(parent)][condition],
                        probability[index : index + 1],
                        masks[index : index + 1],
                        clean[index : index + 1],
                        raw_image[index : index + 1],
                        cv,
                        threshold,
                        False,
                    )
        processed += len(clean)
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "backbone": backbone,
                        "fold": fold,
                        "processed": processed,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    parent_results = {
        detector: {
            parent: {
                condition: external.finalize_accumulator(accumulator, d2, cv)
                for condition, accumulator in by_condition.items()
            }
            for parent, by_condition in by_parent.items()
        }
        for detector, by_parent in accumulators.items()
    }
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C055-CONFIRMATION-FOLD-v1",
        "status": "PASS",
        "seed": seed,
        "backbone": backbone,
        "fold": fold,
        "tile_count": len(rows),
        "detector_seed": detector_seed,
        "selected_conditions": selected,
        "off_provenance": off_provenance,
        "on_provenance": on_provenance,
        "parent_results": parent_results,
        "freeze_payload_sha256": freeze["payload_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_atomic(output_path(seed, backbone, fold), payload)
    return payload


def execute(seed: int, backbone: str, fold: str) -> dict[str, Any]:
    if seed not in SEEDS or backbone not in common.BACKBONES or fold not in TUNNELS:
        raise common.C051Error("invalid C055 confirmation unit")
    if not torch.cuda.is_available():
        raise common.C051Error("CUDA is required")
    freeze = load_freeze()
    path = output_path(seed, backbone, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = common.load_json(path)
        if common.payload_sha256(payload) != payload.get("payload_sha256"):
            raise common.C051Error(f"C055 payload mismatch: {path}")
        return payload
    return evaluate(seed, backbone, fold, freeze)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--backbone", choices=common.BACKBONES, required=True)
    parser.add_argument("--fold", choices=TUNNELS, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.seed, args.backbone, args.fold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
