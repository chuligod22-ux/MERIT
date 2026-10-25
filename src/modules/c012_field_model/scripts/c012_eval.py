"""Run C012 development or sealed holdout evaluation."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
C012 = HERE.parent
MODIFIED = C012.parent
C011_SCRIPTS = MODIFIED / "c011_field_validation" / "scripts"
if str(C011_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C011_SCRIPTS))

import c011_common as common
import c012_profiles as profiles


SCHEMA = "MERIT-C012-FIELD-MODEL-EVAL-v1"
HOLDOUT_PROTOCOL = C012 / "protocol" / "holdout_freeze_v1.json"
SPLITS = {
    "development": {
        "tunnels": ("A", "B"),
        "width_sources": tuple(f"W{index:03d}" for index in range(1, 26)),
    },
    "holdout": {
        "tunnels": ("C", "D"),
        "width_sources": tuple(f"W{index:03d}" for index in range(26, 51)),
    },
}


def output_root(stage: str, endpoint: str) -> Path:
    if stage == "development" and endpoint == "d2":
        return C012 / "runs" / "development_v2" / endpoint
    return C012 / "runs" / f"{stage}_v1" / endpoint


def verify_holdout_protocol() -> dict[str, Any]:
    if not HOLDOUT_PROTOCOL.is_file():
        raise RuntimeError(f"Missing frozen holdout protocol: {HOLDOUT_PROTOCOL}")
    protocol = common.load_json(HOLDOUT_PROTOCOL)
    if protocol.get("schema_version") != "MERIT-C012-HOLDOUT-FREEZE-v1":
        raise RuntimeError("Unexpected C012 holdout protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_HOLDOUT_OUTPUTS":
        raise RuntimeError("C012 holdout protocol is not frozen")
    for label, record in protocol["code"].items():
        common.verify_asset(record, f"C012 code:{label}")
    common.verify_asset(protocol["development_summary"], "C012 development summary")
    return protocol


def encode(value: torch.Tensor) -> np.ndarray:
    return np.rint(value[:, 0].float().cpu().numpy() * np.float32(255.0)).astype(np.uint8)


def run_d2_unit(fold: str, seed: int, cohort: list[dict[str, str]], path: Path) -> dict[str, Any]:
    c009_common, d2_eval, _ = common.load_c009_modules()
    c010 = common.load_c010_inference()
    multi, detector_protocol, d2, cv, external, legacy, external_protocol = d2_eval.load_modules()
    device = torch.device("cuda")
    detector_seed = {0: 20260921, 1: 20260922, 2: 20260923}[seed]
    base_model, provenance = c010.load_model(seed, device)
    detectors = d2_eval.load_detectors(fold, detector_seed, multi, detector_protocol, d2, legacy, external_protocol, device)
    rows = [row for row in cohort if row["tunnel"] == fold]
    dataset = external.ExternalSyntheticDataset(rows, external_protocol)
    loader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0, pin_memory=True)
    bins = int(detector_protocol["detectors"]["histogram_bins"])
    threshold = float(detector_protocol["detectors"]["fixed_threshold"])
    accumulators = {
        detector: collections.defaultdict(
            lambda: {condition: external.Accumulator(bins) for condition in ("raw", "field_profile")}
        )
        for detector in common.DETECTORS
    }
    started = time.perf_counter()
    for batch_index, (image_tensor, degraded_tensor, mask_tensor, parents, metadata) in enumerate(loader, start=1):
        raw = image_tensor.numpy().astype(np.uint8, copy=False)
        masks = mask_tensor.numpy().astype(np.uint8, copy=False)
        normalized = raw.astype(np.float32) / np.float32(255.0)
        zeros = np.zeros(len(raw), dtype=np.float32)
        tensor = torch.from_numpy(c009_common.prompt_tensor(normalized, zeros, zeros)).to(device)
        for detector_name, detector in detectors.items():
            candidate = encode(profiles.detector_candidate(tensor, base_model.base, detector_name))
            images = {"raw": raw, "field_profile": candidate}
            probabilities = {name: external.detect_batch(value, detector, device) for name, value in images.items()}
            for sample_index, parent in enumerate(parents):
                for condition in images:
                    external.update_accumulator(
                        accumulators[detector_name][parent][condition],
                        probabilities[condition][sample_index : sample_index + 1],
                        masks[sample_index : sample_index + 1],
                        raw[sample_index : sample_index + 1],
                        images[condition][sample_index : sample_index + 1],
                        cv,
                        threshold,
                        False,
                    )
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(json.dumps({"endpoint": "d2", "fold": fold, "seed": seed, "batch": batch_index, "batches": len(loader)}), flush=True)
    results = {
        detector: {
            parent: {condition: external.finalize_accumulator(accumulator, d2, cv) for condition, accumulator in by_condition.items()}
            for parent, by_condition in by_parent.items()
        }
        for detector, by_parent in accumulators.items()
    }
    burdens = {
        detector: {
            parent: d2_eval.matched_sensitivity_burden(by_condition["field_profile"], results[detector][parent]["raw"])
            for parent, by_condition in by_parent.items()
        }
        for detector, by_parent in accumulators.items()
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "fold": fold,
        "seed": seed,
        "detector_seed": detector_seed,
        "tile_count": len(dataset),
        "detector_profiles": profiles.DETECTOR_PROFILE,
        "detectors": results,
        "matched_sensitivity_burden": burdens,
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_exclusive(path, payload)
    del base_model, detectors, dataset, loader
    torch.cuda.empty_cache()
    return payload


def run_d2(stage: str) -> dict[str, Any]:
    if stage == "holdout":
        verify_holdout_protocol()
    protocol = common.load_protocol()
    cohort = common.read_csv(common.verify_asset(protocol["assets"]["d2_cohort"], "D2 cohort"))
    root = output_root(stage, "d2")
    progress = root / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    units = [(fold, seed) for seed in (0, 1, 2) for fold in SPLITS[stage]["tunnels"]]
    started = time.perf_counter()
    for index, (fold, seed) in enumerate(units, start=1):
        path = progress / f"fold_{fold}_seed{seed}.json"
        if not path.exists():
            run_d2_unit(fold, seed, cohort, path)
        common.write_json_atomic(root / "status.json", {"schema_version": SCHEMA, "status": "RUNNING" if index < len(units) else "EVALUATION_COMPLETE", "completed": index, "total": len(units), "updated_utc": common.utc_now()})
    marker = {"schema_version": SCHEMA, "status": "PASS", "stage": stage, "units": len(units), "elapsed_seconds_this_invocation": time.perf_counter() - started}
    marker["payload_sha256"] = common.payload_sha256(marker)
    common.write_json_atomic(root / "RUN_COMPLETE.json", marker)
    return marker


@torch.inference_mode()
def tiled_width(gray: np.ndarray, base_model: torch.nn.Module, device: torch.device, c009_common: Any, c010: Any) -> np.ndarray:
    source = gray.astype(np.float32) / np.float32(255.0)
    padded, original_shape = c010.pad_to_tile(source, 448)
    coordinates = [(y, x) for y in c010.axis_starts(padded.shape[0], 448, 112) for x in c010.axis_starts(padded.shape[1], 448, 112)]
    window = c010.blend_window(448)
    accumulator = np.zeros_like(padded, dtype=np.float32)
    weights = np.zeros_like(padded, dtype=np.float32)
    for start in range(0, len(coordinates), 32):
        chunk = coordinates[start : start + 32]
        tiles = np.stack([padded[y : y + 448, x : x + 448] for y, x in chunk]).astype(np.float32)
        zeros = np.zeros(len(tiles), dtype=np.float32)
        tensor = torch.from_numpy(c009_common.prompt_tensor(tiles, zeros, zeros)).to(device)
        output = profiles.width_candidate(tensor, base_model)[:, 0].float().cpu().numpy()
        for value, (y, x) in zip(output, chunk):
            accumulator[y : y + 448, x : x + 448] += value * window
            weights[y : y + 448, x : x + 448] += window
    height, width = original_shape
    return np.rint(np.clip(accumulator[:height, :width] / weights[:height, :width], 0.0, 1.0) * 255.0).astype(np.uint8)


def run_width_unit(source_id: str, rows: list[dict[str, str]], traces: dict[tuple[str, float], dict[str, Any]], seed: int, path: Path) -> dict[str, Any]:
    c009_common, _, width_eval = common.load_c009_modules()
    c010 = common.load_c010_inference()
    v1, v2 = width_eval.load_measurement_modules()
    device = torch.device("cuda")
    model, provenance = c010.load_model(seed, device)
    source_path = width_eval.SOURCE_ROOT / rows[0]["filename"]
    image = v1.read_image(source_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    started = time.perf_counter()
    candidate = tiled_width(gray, model.base, device, c009_common, c010)
    images = {"raw": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "width_profile": cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)}
    measurements = []
    for row in sorted(rows, key=lambda value: float(value["nominal_width_mm"])):
        nominal = float(row["nominal_width_mm"])
        polygon = v1.polygon(row)
        for condition, candidate_image in images.items():
            measurement = width_eval.measure_variant(v1.rectify(candidate_image, polygon), traces[(source_id, nominal)], source_id, nominal, v1, v2)
            measurement["condition"] = condition
            measurements.append(measurement)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "source_id": source_id,
        "seed": seed,
        "width_profile": profiles.WIDTH_PROFILE,
        "measurements": measurements,
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_exclusive(path, payload)
    del model
    torch.cuda.empty_cache()
    return payload


def run_width(stage: str) -> dict[str, Any]:
    if stage == "holdout":
        verify_holdout_protocol()
    common.load_protocol()
    c009_common, _, width_eval = common.load_c009_modules()
    ledger, traces = width_eval.load_inputs(c009_common.load_contract())
    by_source: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in ledger:
        by_source[row["source_id"]].append(row)
    root = output_root(stage, "width")
    progress = root / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    units = [(source, seed) for seed in (0, 1, 2) for source in SPLITS[stage]["width_sources"]]
    started = time.perf_counter()
    for index, (source, seed) in enumerate(units, start=1):
        path = progress / f"{source}_seed{seed}.json"
        if not path.exists():
            run_width_unit(source, by_source[source], traces, seed, path)
        print(json.dumps({"endpoint": "width", "stage": stage, "completed": index, "total": len(units), "source": source, "seed": seed}), flush=True)
        common.write_json_atomic(root / "status.json", {"schema_version": SCHEMA, "status": "RUNNING" if index < len(units) else "EVALUATION_COMPLETE", "completed": index, "total": len(units), "updated_utc": common.utc_now()})
    marker = {"schema_version": SCHEMA, "status": "PASS", "stage": stage, "units": len(units), "elapsed_seconds_this_invocation": time.perf_counter() - started}
    marker["payload_sha256"] = common.payload_sha256(marker)
    common.write_json_atomic(root / "RUN_COMPLETE.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", choices=("d2", "width"))
    parser.add_argument("--stage", choices=tuple(SPLITS), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run_d2(args.stage) if args.endpoint == "d2" else run_width(args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
