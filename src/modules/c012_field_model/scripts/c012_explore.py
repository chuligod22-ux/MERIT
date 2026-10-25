"""C012 development screen for a field-safe MERIT v2.3 adapter.

This is an explicitly unblinded engineering screen.  It uses only the C011
development partition (tunnels A/B and printed-width frames W001-W025) and
must never be reported as prospective field confirmation.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
C012 = HERE.parent
MODIFIED = C012.parent
C011_SCRIPTS = MODIFIED / "c011_field_validation" / "scripts"
if str(C011_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C011_SCRIPTS))

import c011_common as common


SCHEMA = "MERIT-C012-FIELD-DEVELOPMENT-SCREEN-v1"
RUN_ROOT = C012 / "runs" / "screen_v1"
DISCOVERY_TUNNELS = ("A", "B")
DISCOVERY_WIDTH_SOURCES = tuple(f"W{index:03d}" for index in range(1, 26))

# All candidates preserve an exact identity route when the operational prompt
# is zero.  These values describe only the active field route.
CANDIDATES: dict[str, dict[str, float | str]] = {
    "inv0125": {"family": "inverse_base_residual", "alpha": -0.125, "gain": 0.0},
    "inv0250": {"family": "inverse_base_residual", "alpha": -0.250, "gain": 0.0},
    "inv0375": {"family": "inverse_base_residual", "alpha": -0.375, "gain": 0.0},
    "inv0500": {"family": "inverse_base_residual", "alpha": -0.500, "gain": 0.0},
    "inv0750": {"family": "inverse_base_residual", "alpha": -0.750, "gain": 0.0},
    "sharp010": {"family": "unsharp", "alpha": 0.0, "gain": 0.10},
    "sharp020": {"family": "unsharp", "alpha": 0.0, "gain": 0.20},
    "sharp035": {"family": "unsharp", "alpha": 0.0, "gain": 0.35},
    "combo_a025_g010": {"family": "combined", "alpha": -0.250, "gain": 0.10},
    "combo_a025_g020": {"family": "combined", "alpha": -0.250, "gain": 0.20},
}
CONDITIONS = ("raw", *CANDIDATES)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def gaussian_blur(source: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    radius = 3
    axis = torch.arange(-radius, radius + 1, device=source.device, dtype=torch.float32)
    kernel_1d = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).to(source.dtype)
    kernel = kernel[None, None]
    padded = F.pad(source, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel)


@torch.inference_mode()
def make_candidates(
    inputs: torch.Tensor,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    source = inputs[:, :1]
    base = model.base.forward_preclip(inputs)
    highpass = source - gaussian_blur(source)
    outputs = {"raw": source}
    for name, specification in CANDIDATES.items():
        alpha = float(specification["alpha"])
        gain = float(specification["gain"])
        outputs[name] = torch.clamp(
            source + alpha * (base - source) + gain * highpass,
            0.0,
            1.0,
        )
    return outputs


def encode_candidates(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: np.rint(value[:, 0].float().cpu().numpy() * np.float32(255.0)).astype(np.uint8)
        for name, value in outputs.items()
    }


def evenly_spaced_rows(rows: list[dict[str, str]], maximum: int) -> list[dict[str, str]]:
    if len(rows) <= maximum:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def screen_d2(maximum_rows_per_tunnel: int, seed: int) -> dict[str, Any]:
    c009_common, d2_eval, _ = common.load_c009_modules()
    c010 = common.load_c010_inference()
    multi, detector_protocol, d2, cv, external, legacy, external_protocol = d2_eval.load_modules()
    protocol = common.load_protocol()
    cohort = common.read_csv(common.verify_asset(protocol["assets"]["d2_cohort"], "D2 cohort"))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, provenance = c010.load_model(seed, device)
    detector_seed = {0: 20260921, 1: 20260922, 2: 20260923}[seed]
    bins = int(detector_protocol["detectors"]["histogram_bins"])
    threshold = float(detector_protocol["detectors"]["fixed_threshold"])
    parent_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in DISCOVERY_TUNNELS:
        fold_rows = evenly_spaced_rows(
            [row for row in cohort if row["tunnel"] == fold],
            maximum_rows_per_tunnel,
        )
        detectors = d2_eval.load_detectors(
            fold,
            detector_seed,
            multi,
            detector_protocol,
            d2,
            legacy,
            external_protocol,
            device,
        )
        dataset = external.ExternalSyntheticDataset(fold_rows, external_protocol)
        loader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0, pin_memory=True)
        accumulators = {
            detector: collections.defaultdict(
                lambda: {condition: external.Accumulator(bins) for condition in CONDITIONS}
            )
            for detector in common.DETECTORS
        }
        for batch_index, (image_tensor, degraded_tensor, mask_tensor, parents, metadata) in enumerate(loader, start=1):
            raw = image_tensor.numpy().astype(np.uint8, copy=False)
            masks = mask_tensor.numpy().astype(np.uint8, copy=False)
            normalized = raw.astype(np.float32) / np.float32(255.0)
            zeros = np.zeros(len(raw), dtype=np.float32)
            tensor = torch.from_numpy(c009_common.prompt_tensor(normalized, zeros, zeros)).to(device)
            candidates = encode_candidates(make_candidates(tensor, model))
            for detector_name, detector in detectors.items():
                probabilities = {
                    condition: external.detect_batch(candidate, detector, device)
                    for condition, candidate in candidates.items()
                }
                for sample_index, parent in enumerate(parents):
                    for condition in CONDITIONS:
                        external.update_accumulator(
                            accumulators[detector_name][parent][condition],
                            probabilities[condition][sample_index : sample_index + 1],
                            masks[sample_index : sample_index + 1],
                            raw[sample_index : sample_index + 1],
                            candidates[condition][sample_index : sample_index + 1],
                            cv,
                            threshold,
                            False,
                        )
            if batch_index == 1 or batch_index == len(loader):
                print(json.dumps({"stage": "d2", "fold": fold, "batch": batch_index, "batches": len(loader)}), flush=True)
        for detector_name, by_parent in accumulators.items():
            for parent, by_condition in by_parent.items():
                metrics = {
                    condition: external.finalize_accumulator(accumulator, d2, cv)
                    for condition, accumulator in by_condition.items()
                }
                raw_metric = metrics["raw"]
                for condition in CANDIDATES:
                    burden = d2_eval.matched_sensitivity_burden(
                        by_condition[condition],
                        raw_metric,
                    )
                    parent_rows.append(
                        {
                            "fold": fold,
                            "parent": parent,
                            "detector": detector_name,
                            "condition": condition,
                            "delta_auprc": metrics[condition]["pixel_auprc"] - raw_metric["pixel_auprc"],
                            "delta_f1": metrics[condition]["f1_at_0_5"] - raw_metric["f1_at_0_5"],
                            "delta_matched_fpr": burden["delta_false_positive_pixel_rate_restored_minus_degraded"],
                        }
                    )
        del detectors, dataset, loader, accumulators
        torch.cuda.empty_cache()
    summary: dict[str, Any] = {}
    for condition in CANDIDATES:
        selected = [row for row in parent_rows if row["condition"] == condition]
        by_detector = {}
        for detector in common.DETECTORS:
            detector_rows = [row for row in selected if row["detector"] == detector]
            by_detector[detector] = {
                key: float(np.mean([row[key] for row in detector_rows]))
                for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
            }
        summary[condition] = {
            "panel_delta_auprc": float(np.mean([row["delta_auprc"] for row in selected])),
            "panel_delta_f1": float(np.mean([row["delta_f1"] for row in selected])),
            "panel_delta_matched_fpr": float(np.mean([row["delta_matched_fpr"] for row in selected])),
            "detectors": by_detector,
        }
    result = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "scientific_status": "UNBLINDED_DEVELOPMENT_SCREEN_ONLY",
        "split": {"tunnels": list(DISCOVERY_TUNNELS), "maximum_rows_per_tunnel": maximum_rows_per_tunnel},
        "restorer_seed": seed,
        "candidates": CANDIDATES,
        "summary": summary,
        "parent_rows": parent_rows,
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["payload_sha256"] = common.payload_sha256(result)
    write_json(RUN_ROOT / "d2_screen.json", result)
    return result


@torch.inference_mode()
def tiled_candidates(
    gray: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    c009_common: Any,
    c010: Any,
    tile_size: int = 448,
    overlap: int = 112,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    source = gray.astype(np.float32) / np.float32(255.0)
    padded, original_shape = c010.pad_to_tile(source, tile_size)
    coordinates = [(y, x) for y in c010.axis_starts(padded.shape[0], tile_size, overlap) for x in c010.axis_starts(padded.shape[1], tile_size, overlap)]
    window = c010.blend_window(tile_size)
    accumulators = {condition: np.zeros_like(padded, dtype=np.float32) for condition in CONDITIONS}
    weights = np.zeros_like(padded, dtype=np.float32)
    for start in range(0, len(coordinates), batch_size):
        chunk = coordinates[start : start + batch_size]
        tiles = np.stack([padded[y : y + tile_size, x : x + tile_size] for y, x in chunk]).astype(np.float32)
        zeros = np.zeros(len(tiles), dtype=np.float32)
        tensor = torch.from_numpy(c009_common.prompt_tensor(tiles, zeros, zeros)).to(device)
        outputs = make_candidates(tensor, model)
        for condition, output in outputs.items():
            values = output[:, 0].float().cpu().numpy()
            for value, (y, x) in zip(values, chunk):
                accumulators[condition][y : y + tile_size, x : x + tile_size] += value * window
        for y, x in chunk:
            weights[y : y + tile_size, x : x + tile_size] += window
    height, width = original_shape
    return {
        condition: np.rint(np.clip(value[:height, :width] / weights[:height, :width], 0.0, 1.0) * 255.0).astype(np.uint8)
        for condition, value in accumulators.items()
    }


def screen_width(maximum_sources: int, seed: int) -> dict[str, Any]:
    common.load_protocol()
    c009_common, _, width_eval = common.load_c009_modules()
    c010 = common.load_c010_inference()
    contract = c009_common.load_contract()
    ledger, traces = width_eval.load_inputs(contract)
    by_source: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in ledger:
        by_source[row["source_id"]].append(row)
    source_ids = list(DISCOVERY_WIDTH_SOURCES[:maximum_sources])
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, provenance = c010.load_model(seed, device)
    v1, v2 = width_eval.load_measurement_modules()
    rows_out: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, source_id in enumerate(source_ids, start=1):
        rows = by_source[source_id]
        source_path = width_eval.SOURCE_ROOT / rows[0]["filename"]
        image = v1.read_image(source_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = tiled_candidates(gray, model, device, c009_common, c010)
        bgr = {name: cv2.cvtColor(value, cv2.COLOR_GRAY2BGR) for name, value in candidates.items()}
        for row in sorted(rows, key=lambda value: float(value["nominal_width_mm"])):
            nominal = float(row["nominal_width_mm"])
            polygon = v1.polygon(row)
            for condition in CONDITIONS:
                measurement = width_eval.measure_variant(
                    v1.rectify(bgr[condition], polygon),
                    traces[(source_id, nominal)],
                    source_id,
                    nominal,
                    v1,
                    v2,
                )
                rows_out.append({"source_id": source_id, "condition": condition, **measurement})
        print(json.dumps({"stage": "width", "source": source_id, "completed": index, "total": len(source_ids)}), flush=True)
    summary: dict[str, Any] = {}
    raw_lookup = {(row["source_id"], row["nominal_width_mm"]): row for row in rows_out if row["condition"] == "raw"}
    for condition in CANDIDATES:
        selected = [row for row in rows_out if row["condition"] == condition and row["status"] == "ESTIMABLE"]
        deltas = []
        for row in selected:
            raw = raw_lookup[(row["source_id"], row["nominal_width_mm"])]
            if raw["status"] == "ESTIMABLE":
                deltas.append(float(row["absolute_error_mm"]) - float(raw["absolute_error_mm"]))
        summary[condition] = {
            "paired_estimable_rows": len(deltas),
            "delta_absolute_error_mm": float(np.mean(deltas)) if deltas else None,
            "candidate_nonestimable_rate": 1.0 - len(selected) / (len(source_ids) * 8),
        }
    result = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "scientific_status": "UNBLINDED_DEVELOPMENT_SCREEN_ONLY",
        "split": {"source_ids": source_ids, "primary_nominal_width_mm": [0.3, 1.0]},
        "restorer_seed": seed,
        "candidates": CANDIDATES,
        "summary": summary,
        "measurements": rows_out,
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["payload_sha256"] = common.payload_sha256(result)
    write_json(RUN_ROOT / "width_screen.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("d2", "width"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--maximum-rows-per-tunnel", type=int, default=100)
    parser.add_argument("--maximum-sources", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "d2":
        result = screen_d2(args.maximum_rows_per_tunnel, args.seed)
    else:
        result = screen_width(args.maximum_sources, args.seed)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
