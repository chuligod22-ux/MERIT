"""MERIT v2.4 cross-fitted development and evaluation pipeline.

The supplied tunnels A-D are treated as four outer folds.  For each target
tunnel, route selection uses seed-0 outcomes from the other three tunnels
only.  The selected route is then evaluated on the excluded tunnel with three
frozen restorer/detector seeds.  This is a development-time cross-fitted
estimate on existing data; it is not a prospective new-site study.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
C016 = HERE.parent
MODIFIED = C016.parent
C011_SCRIPTS = MODIFIED / "c011_field_validation" / "scripts"
C014_SCRIPTS = MODIFIED / "c014_existing_four_tunnel_validation" / "scripts"
C010_SCRIPTS = MODIFIED / "c010_merit_v2_development" / "scripts"
for directory in (HERE, C011_SCRIPTS, C014_SCRIPTS, C010_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import c010_infer_v3 as c010  # noqa: E402
import c011_common as field_common  # noqa: E402
import c014_common as width_common  # noqa: E402
import c016_profiles as profiles  # noqa: E402


TUNNELS = ("A", "B", "C", "D")
SEEDS = (0, 1, 2)
DETECTOR_SEEDS = {0: 20260921, 1: 20260922, 2: 20260923}
PROTOCOL = C016 / "protocol" / "v24_crossfit_protocol_v1.json"
SELECTION = C016 / "protocol" / "v24_crossfit_selection_v1.json"
RUNS = C016 / "runs"
D2_COHORT = MODIFIED / "c007_d2_external_synthetic" / "artifacts" / "external_v1" / "cohort.csv"
CROSS_SECTIONS = (
    MODIFIED
    / "c014_existing_four_tunnel_validation"
    / "protocol"
    / "natural_width_cross_sections_v1.csv"
)
C012_SUMMARY = MODIFIED / "c012_field_model" / "runs" / "c012_final_summary.json"
C014_SUMMARY = (
    MODIFIED
    / "c014_existing_four_tunnel_validation"
    / "runs"
    / "c014_final_summary.json"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261013


class C016Error(RuntimeError):
    """Fail-closed v2.4 contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def asset_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise C016Error(f"Missing asset: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_asset(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise C016Error(f"Missing {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise C016Error(f"Byte-size mismatch for {label}: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise C016Error(f"SHA-256 mismatch for {label}: expected={record['sha256']} actual={actual}")
    return path


def mean(values: Iterable[float]) -> float:
    materialized = np.asarray(list(values), dtype=np.float64)
    if not len(materialized):
        raise C016Error("Cannot average an empty collection")
    return float(materialized.mean())


def freeze_protocol() -> dict[str, Any]:
    """Freeze candidates, cross-fitting rules, endpoints, and code hashes."""

    if PROTOCOL.exists():
        return verify_protocol()
    test_path = HERE / "test_c016.py"
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-CROSSFIT-PROTOCOL-v1",
        "status": "FROZEN_BEFORE_V24_OUTCOMES",
        "scientific_role": (
            "Development-time outer-tunnel cross-fitting on the four supplied tunnels. "
            "Each target tunnel is excluded from its route selection. This is not a "
            "prospective physically new-site confirmation."
        ),
        "tunnels": list(TUNNELS),
        "selection_seed": 0,
        "evaluation_seeds": list(SEEDS),
        "candidate_profiles": profiles.CANDIDATES,
        "detection_selection": {
            "unit": "detector within excluded target tunnel",
            "training_folds": "the other three tunnels only",
            "aggregation": "parent within tunnel, equal tunnel mean",
            "eligibility": (
                "mean delta AUPRC >0; mean matched-sensitivity delta FPR <=0; "
                "at least 2/3 training tunnels have positive delta AUPRC"
            ),
            "objective": "maximum mean delta AUPRC; ties prefer smaller profile magnitude",
            "fallback": "exact identity",
        },
        "width_selection": {
            "unit": "excluded target tunnel",
            "training_folds": "the other three tunnels only",
            "aggregation": "ROI within tile, tile within parent, equal tunnel mean",
            "eligibility": (
                "mean delta absolute error <0; at least 2/3 training tunnels improve; "
                "effective non-estimable rate no greater than raw"
            ),
            "objective": "minimum mean delta absolute error; ties prefer smaller profile magnitude",
            "fallback": "exact identity",
            "safe_routing": (
                "When raw is ESTIMABLE and the active profile is NON_ESTIMABLE, use the raw "
                "measurement. The reference width is never consulted by this routing rule."
            ),
        },
        "final_gates": {
            "detection": (
                "panel delta AUPRC mean and 95% CI lower >0; all four tunnels positive; "
                "matched-sensitivity delta FPR mean <=0; every detector mean delta AUPRC >=0; "
                "panel mean exceeds the frozen v2.3 four-tunnel descriptive mean"
            ),
            "natural_apparent_width": (
                ">=300 ROI, >=100 tiles, >=10 parents, four tunnels; delta absolute-error "
                "mean and 95% CI upper <0; >=3/4 tunnels improve; effective non-estimable "
                "increase <=2 percentage points; paired estimable rate >=70%"
            ),
        },
        "claim_boundary": (
            "Detection uses existing annotated tunnel data. Width is expert-mask-defined natural "
            "crack apparent width in pixels, not physical millimetre width or traceable metrology."
        ),
        "assets": {
            "d2_cohort": asset_record(D2_COHORT),
            "natural_width_cross_sections": asset_record(CROSS_SECTIONS),
            "v23_summary": asset_record(C012_SUMMARY),
            "c014_summary": asset_record(C014_SUMMARY),
        },
        "code": {
            "profiles": asset_record(HERE / "c016_profiles.py"),
            "pipeline": asset_record(Path(__file__)),
            "tests": asset_record(test_path),
        },
    }
    payload["payload_sha256"] = payload_sha256(payload)
    write_json_exclusive(PROTOCOL, payload)
    return payload


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL.is_file():
        raise C016Error(f"Missing frozen v2.4 protocol: {PROTOCOL}")
    payload = load_json(PROTOCOL)
    if payload.get("schema_version") != "MERIT-C016-V24-CROSSFIT-PROTOCOL-v1":
        raise C016Error("Unexpected v2.4 protocol schema")
    if payload.get("status") != "FROZEN_BEFORE_V24_OUTCOMES":
        raise C016Error("v2.4 protocol is not frozen")
    if payload_sha256(payload) != payload.get("payload_sha256"):
        raise C016Error("v2.4 protocol payload hash mismatch")
    for group in ("assets", "code"):
        for label, record in payload[group].items():
            verify_asset(record, f"{group}:{label}")
    if payload["candidate_profiles"] != profiles.CANDIDATES:
        raise C016Error("Candidate definitions changed after protocol freeze")
    return payload


def _valid_payload(path: Path, schema: str) -> bool:
    if not path.is_file():
        return False
    payload = load_json(path)
    return payload.get("schema_version") == schema and payload_sha256(payload) == payload.get("payload_sha256")


def encode_candidates(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: np.rint(value[:, 0].float().cpu().numpy() * np.float32(255.0)).astype(np.uint8)
        for name, value in outputs.items()
    }


def detect_candidate_map(
    candidates: dict[str, np.ndarray],
    detector: torch.nn.Module,
    device: torch.device,
    external: Any,
    conditions_per_call: int = 3,
) -> dict[str, np.ndarray]:
    """Batch several candidate conditions into each detector call."""

    names = list(candidates)
    batch = len(next(iter(candidates.values())))
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(names), conditions_per_call):
        chunk = names[start : start + conditions_per_call]
        stacked = np.concatenate([candidates[name] for name in chunk], axis=0)
        probabilities = external.detect_batch(stacked, detector, device)
        for index, name in enumerate(chunk):
            result[name] = probabilities[index * batch : (index + 1) * batch]
    return result


def _detection_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    c009_common, d2_eval, _ = field_common.load_c009_modules()
    multi, detector_protocol, d2, cv, external, legacy, external_protocol = d2_eval.load_modules()
    return c009_common, d2_eval, multi, detector_protocol, d2, cv, external, (legacy, external_protocol)


def _detection_screen_unit(fold: str, batch_size: int = 12) -> dict[str, Any]:
    c009_common, d2_eval, multi, detector_protocol, d2, cv, external, tail = _detection_modules()
    legacy, external_protocol = tail
    protocol = field_common.load_protocol()
    cohort = field_common.read_csv(D2_COHORT)
    fold_rows = [row for row in cohort if row["tunnel"] == fold]
    if not fold_rows:
        raise C016Error(f"Empty D2 fold: {fold}")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise C016Error("CUDA is required for v2.4")
    model, provenance = c010.load_model(0, device)
    detectors = d2_eval.load_detectors(
        fold,
        DETECTOR_SEEDS[0],
        multi,
        detector_protocol,
        d2,
        legacy,
        external_protocol,
        device,
    )
    dataset = external.ExternalSyntheticDataset(fold_rows, external_protocol)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    bins = int(detector_protocol["detectors"]["histogram_bins"])
    threshold = float(detector_protocol["detectors"]["fixed_threshold"])
    condition_names = ("raw", *profiles.CANDIDATES)
    accumulators = {
        detector: collections.defaultdict(
            lambda: {condition: external.Accumulator(bins) for condition in condition_names}
        )
        for detector in field_common.DETECTORS
    }
    started = time.perf_counter()
    for batch_index, (image_tensor, degraded_tensor, mask_tensor, parents, metadata) in enumerate(loader, 1):
        del degraded_tensor, metadata
        raw = image_tensor.numpy().astype(np.uint8, copy=False)
        masks = mask_tensor.numpy().astype(np.uint8, copy=False)
        normalized = raw.astype(np.float32) / np.float32(255.0)
        zeros = np.zeros(len(raw), dtype=np.float32)
        tensor = torch.from_numpy(c009_common.prompt_tensor(normalized, zeros, zeros)).to(device)
        with c010.pilot.autocast(device):
            transformed = profiles.make_candidates(tensor, model.base)
        candidates = {"raw": raw, **encode_candidates(transformed)}
        for detector_name, detector in detectors.items():
            probabilities = detect_candidate_map(candidates, detector, device, external)
            for sample_index, parent in enumerate(parents):
                for condition in condition_names:
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
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(json.dumps({"stage": "screen_detection", "fold": fold, "batch": batch_index, "batches": len(loader)}), flush=True)
    parent_rows: list[dict[str, Any]] = []
    for detector_name, by_parent in accumulators.items():
        for parent, by_condition in by_parent.items():
            metrics = {
                condition: external.finalize_accumulator(accumulator, d2, cv)
                for condition, accumulator in by_condition.items()
            }
            raw_metric = metrics["raw"]
            parent_rows.append(
                {
                    "tunnel": fold,
                    "detector": detector_name,
                    "parent": parent,
                    "condition": profiles.IDENTITY,
                    "delta_auprc": 0.0,
                    "delta_f1": 0.0,
                    "delta_matched_fpr": 0.0,
                }
            )
            for condition in profiles.CANDIDATES:
                burden = d2_eval.matched_sensitivity_burden(by_condition[condition], raw_metric)
                parent_rows.append(
                    {
                        "tunnel": fold,
                        "detector": detector_name,
                        "parent": parent,
                        "condition": condition,
                        "delta_auprc": float(metrics[condition]["pixel_auprc"] - raw_metric["pixel_auprc"]),
                        "delta_f1": float(metrics[condition]["f1_at_0_5"] - raw_metric["f1_at_0_5"]),
                        "delta_matched_fpr": float(burden["delta_false_positive_pixel_rate_restored_minus_degraded"]),
                    }
                )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-SCREEN-UNIT-v1",
        "status": "PASS",
        "fold": fold,
        "selection_seed": 0,
        "tile_count": len(dataset),
        "parents": len({row["parent"] for row in parent_rows}),
        "parent_rows": parent_rows,
        "base_provenance": provenance,
        "c011_protocol_payload_sha256": protocol["payload_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    del model, detectors, dataset, loader, accumulators
    torch.cuda.empty_cache()
    return payload


def screen_detection(batch_size: int = 12) -> dict[str, Any]:
    protocol = verify_protocol()
    progress = RUNS / "screen_v1" / "detection" / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    units = []
    started = time.perf_counter()
    for fold in TUNNELS:
        path = progress / f"fold_{fold}_seed0.json"
        if not _valid_payload(path, "MERIT-C016-DETECTION-SCREEN-UNIT-v1"):
            write_json_atomic(path, _detection_screen_unit(fold, batch_size))
        units.append(load_json(path))
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-SCREEN-v1",
        "status": "PASS_COMPLETE",
        "scientific_status": "UNBLINDED_CROSSFIT_DEVELOPMENT_SCREEN",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "folds": list(TUNNELS),
        "selection_seed": 0,
        "tiles": sum(int(unit["tile_count"]) for unit in units),
        "parent_rows": [row for unit in units for row in unit["parent_rows"]],
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    write_json_atomic(RUNS / "screen_v1" / "detection_screen.json", payload)
    return payload


def _coordinates(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[key]) for key in ("x1_px", "y1_px", "x2_px", "y2_px"))


def _load_width_image(row: dict[str, str]) -> np.ndarray:
    path = Path(row["image_path"])
    if sha256_file(path) != row["image_sha256"]:
        raise C016Error(f"Natural-width image hash changed: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (512, 512):
        raise C016Error(f"Invalid natural-width image: {path}")
    return image.astype(np.float32) / np.float32(255.0)


def _width_row(
    annotation: dict[str, str],
    condition: str,
    raw_measurement: dict[str, Any],
    candidate_measurement: dict[str, Any],
) -> dict[str, Any]:
    effective, fallback = profiles.safe_width_measurement(raw_measurement, candidate_measurement)
    reference = float(annotation["reference_width_px"])
    delta = None
    if raw_measurement["status"] == "ESTIMABLE" and effective["status"] == "ESTIMABLE":
        delta = abs(float(effective["estimate_px"]) - reference) - abs(
            float(raw_measurement["estimate_px"]) - reference
        )
    return {
        "roi_id": annotation["roi_id"],
        "image_id": annotation["image_id"],
        "tunnel": annotation["tunnel"],
        "parent": annotation["parent"],
        "condition": condition,
        "reference_width_px": reference,
        "raw_status": raw_measurement["status"],
        "raw_estimate_px": raw_measurement.get("estimate_px"),
        "candidate_status": candidate_measurement["status"],
        "candidate_estimate_px": candidate_measurement.get("estimate_px"),
        "effective_status": effective["status"],
        "effective_estimate_px": effective.get("estimate_px"),
        "safe_fallback": fallback,
        "delta_absolute_error_px": delta,
    }


def _width_screen_unit(
    tunnel: str,
    parent: str,
    rows: list[dict[str, str]],
    model: Any,
    provenance: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    by_image: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    image_groups = sorted(by_image.items())
    measurements: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(image_groups), batch_size):
        chunk = image_groups[start : start + batch_size]
        raw_images = [_load_width_image(group_rows[0]) for _, group_rows in chunk]
        batch = np.stack(raw_images).astype(np.float32, copy=False)
        zeros = np.zeros(len(batch), dtype=np.float32)
        inputs = c010.pilot.common.prompt_tensor(batch, zeros, zeros)
        tensor = torch.from_numpy(inputs).to(device)
        with c010.pilot.autocast(device):
            outputs = profiles.make_candidates(tensor, model.base)
        candidate_images = {
            name: value[:, 0].float().cpu().numpy().astype(np.float32)
            for name, value in outputs.items()
        }
        for item_index, ((image_id, group_rows), raw) in enumerate(zip(chunk, raw_images)):
            for annotation in group_rows:
                coordinates = _coordinates(annotation)
                raw_measurement = width_common.estimate_dark_width(raw, coordinates)
                measurements.append(
                    _width_row(annotation, profiles.IDENTITY, raw_measurement, raw_measurement)
                )
                for condition in profiles.CANDIDATES:
                    candidate_measurement = width_common.estimate_dark_width(
                        candidate_images[condition][item_index], coordinates
                    )
                    measurements.append(
                        _width_row(annotation, condition, raw_measurement, candidate_measurement)
                    )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-WIDTH-SCREEN-UNIT-v1",
        "status": "PASS",
        "tunnel": tunnel,
        "parent": parent,
        "selection_seed": 0,
        "tiles": len(image_groups),
        "rois": len(rows),
        "measurements": measurements,
        "base_provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def screen_width(batch_size: int = 12) -> dict[str, Any]:
    protocol = verify_protocol()
    with CROSS_SECTIONS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_unit: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_unit[(row["tunnel"], row["parent"])].append(row)
    if tuple(sorted({key[0] for key in by_unit})) != TUNNELS:
        raise C016Error("Natural-width screen does not cover all four tunnels")
    progress = RUNS / "screen_v1" / "width" / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise C016Error("CUDA is required for v2.4")
    missing = [
        unit
        for unit in sorted(by_unit)
        if not _valid_payload(
            progress / f"tunnel_{unit[0]}_parent_{unit[1]}_seed0.json",
            "MERIT-C016-WIDTH-SCREEN-UNIT-v1",
        )
    ]
    model = provenance = None
    if missing:
        model, provenance = c010.load_model(0, device)
    started = time.perf_counter()
    for index, (tunnel, parent) in enumerate(sorted(by_unit), 1):
        path = progress / f"tunnel_{tunnel}_parent_{parent}_seed0.json"
        if not _valid_payload(path, "MERIT-C016-WIDTH-SCREEN-UNIT-v1"):
            assert model is not None and provenance is not None
            write_json_atomic(
                path,
                _width_screen_unit(
                    tunnel,
                    parent,
                    by_unit[(tunnel, parent)],
                    model,
                    provenance,
                    device,
                    batch_size,
                ),
            )
        print(json.dumps({"stage": "screen_width", "completed": index, "total": len(by_unit), "tunnel": tunnel, "parent": parent}), flush=True)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    units = [load_json(path) for path in sorted(progress.glob("tunnel_*_seed0.json"))]
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-WIDTH-SCREEN-v1",
        "status": "PASS_COMPLETE",
        "scientific_status": "UNBLINDED_CROSSFIT_DEVELOPMENT_SCREEN",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "units": len(units),
        "tiles": sum(int(unit["tiles"]) for unit in units),
        "rois": sum(int(unit["rois"]) for unit in units),
        "measurements": [row for unit in units for row in unit["measurements"]],
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    write_json_atomic(RUNS / "screen_v1" / "width_screen.json", payload)
    return payload


def _equal_tunnel_detection_score(
    rows: list[dict[str, Any]], target: str, detector: str, condition: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["tunnel"] != target
        and row["detector"] == detector
        and row["condition"] == condition
    ]
    per_tunnel = {}
    for tunnel in TUNNELS:
        if tunnel == target:
            continue
        tunnel_rows = [row for row in selected if row["tunnel"] == tunnel]
        per_tunnel[tunnel] = {
            key: mean(row[key] for row in tunnel_rows)
            for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
        }
    score = {
        key: mean(value[key] for value in per_tunnel.values())
        for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
    }
    score["positive_tunnels"] = sum(value["delta_auprc"] > 0.0 for value in per_tunnel.values())
    score["per_tunnel"] = per_tunnel
    score["eligible"] = bool(
        condition == profiles.IDENTITY
        or (
            score["delta_auprc"] > 0.0
            and score["delta_matched_fpr"] <= 0.0
            and score["positive_tunnels"] >= 2
        )
    )
    return score


def _width_training_score(
    rows: list[dict[str, Any]], target: str, condition: str
) -> dict[str, Any]:
    selected = [
        row for row in rows if row["tunnel"] != target and row["condition"] == condition
    ]
    by_image: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for row in selected:
        if row["delta_absolute_error_px"] is not None:
            by_image[(row["tunnel"], row["parent"], row["image_id"])].append(
                float(row["delta_absolute_error_px"])
            )
    by_parent: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for (tunnel, parent, image_id), values in by_image.items():
        del image_id
        by_parent[(tunnel, parent)].append(mean(values))
    per_tunnel = {
        tunnel: mean(
            mean(values)
            for (row_tunnel, parent), values in by_parent.items()
            if row_tunnel == tunnel
        )
        for tunnel in TUNNELS
        if tunnel != target
    }
    raw_nonestimable = mean(row["raw_status"] != "ESTIMABLE" for row in selected)
    effective_nonestimable = mean(row["effective_status"] != "ESTIMABLE" for row in selected)
    paired_rate = mean(row["delta_absolute_error_px"] is not None for row in selected)
    delta = mean(per_tunnel.values())
    score = {
        "delta_absolute_error_px": delta,
        "per_tunnel": per_tunnel,
        "improved_tunnels": sum(value < 0.0 for value in per_tunnel.values()),
        "raw_nonestimable_rate": raw_nonestimable,
        "effective_nonestimable_rate": effective_nonestimable,
        "paired_estimable_rate": paired_rate,
    }
    score["eligible"] = bool(
        condition == profiles.IDENTITY
        or (
            delta < 0.0
            and score["improved_tunnels"] >= 2
            and effective_nonestimable <= raw_nonestimable + 1e-12
        )
    )
    return score


def select_routes() -> dict[str, Any]:
    protocol = verify_protocol()
    detection_path = RUNS / "screen_v1" / "detection_screen.json"
    width_path = RUNS / "screen_v1" / "width_screen.json"
    if not _valid_payload(detection_path, "MERIT-C016-DETECTION-SCREEN-v1"):
        raise C016Error("Detection screen is missing or invalid")
    if not _valid_payload(width_path, "MERIT-C016-WIDTH-SCREEN-v1"):
        raise C016Error("Width screen is missing or invalid")
    detection_rows = load_json(detection_path)["parent_rows"]
    width_rows = load_json(width_path)["measurements"]
    conditions = (profiles.IDENTITY, *profiles.CANDIDATES)
    detection_routes: dict[str, dict[str, Any]] = {}
    for target in TUNNELS:
        detection_routes[target] = {}
        for detector in field_common.DETECTORS:
            scores = {
                condition: _equal_tunnel_detection_score(
                    detection_rows, target, detector, condition
                )
                for condition in conditions
            }
            eligible = [condition for condition, score in scores.items() if score["eligible"]]
            selected = max(
                eligible,
                key=lambda condition: (
                    float(scores[condition]["delta_auprc"]),
                    -profiles.profile_magnitude(condition),
                    condition,
                ),
            )
            detection_routes[target][detector] = {
                "selected": selected,
                "training_tunnels": [tunnel for tunnel in TUNNELS if tunnel != target],
                "scores": scores,
            }
    width_routes: dict[str, Any] = {}
    for target in TUNNELS:
        scores = {
            condition: _width_training_score(width_rows, target, condition)
            for condition in conditions
        }
        eligible = [condition for condition, score in scores.items() if score["eligible"]]
        selected = min(
            eligible,
            key=lambda condition: (
                float(scores[condition]["delta_absolute_error_px"]),
                profiles.profile_magnitude(condition),
                condition,
            ),
        )
        width_routes[target] = {
            "selected": selected,
            "training_tunnels": [tunnel for tunnel in TUNNELS if tunnel != target],
            "scores": scores,
        }
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-CROSSFIT-SELECTION-v1",
        "status": "FROZEN_AFTER_TRAINING_FOLDS_BEFORE_OUTER_FOLD_AGGREGATION",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "screen_hashes": {
            "detection": sha256_file(detection_path),
            "width": sha256_file(width_path),
        },
        "detection_routes": detection_routes,
        "width_routes": width_routes,
        "independence_check": {
            target: [tunnel for tunnel in TUNNELS if tunnel != target]
            for target in TUNNELS
        },
    }
    payload["payload_sha256"] = payload_sha256(payload)
    if SELECTION.exists():
        existing = load_json(SELECTION)
        if existing != payload:
            raise C016Error("Frozen v2.4 route selection differs from recomputed selection")
        return existing
    write_json_exclusive(SELECTION, payload)
    return payload


def verify_selection() -> dict[str, Any]:
    protocol = verify_protocol()
    if not SELECTION.is_file():
        raise C016Error("Missing frozen v2.4 selection manifest")
    payload = load_json(SELECTION)
    if payload.get("schema_version") != "MERIT-C016-V24-CROSSFIT-SELECTION-v1":
        raise C016Error("Unexpected v2.4 selection schema")
    if payload_sha256(payload) != payload.get("payload_sha256"):
        raise C016Error("v2.4 selection payload hash mismatch")
    if payload["protocol_payload_sha256"] != protocol["payload_sha256"]:
        raise C016Error("v2.4 selection points to a different protocol")
    for target, training in payload["independence_check"].items():
        if target in training or sorted(training) != sorted(set(TUNNELS) - {target}):
            raise C016Error(f"Target leakage in selection manifest: {target}")
    return payload


def _derive_detection_seed0(fold: str, selection: dict[str, Any]) -> dict[str, Any]:
    screen = load_json(RUNS / "screen_v1" / "detection_screen.json")
    rows = []
    for detector in field_common.DETECTORS:
        condition = selection["detection_routes"][fold][detector]["selected"]
        rows.extend(
            row
            for row in screen["parent_rows"]
            if row["tunnel"] == fold
            and row["detector"] == detector
            and row["condition"] == condition
        )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-EVAL-UNIT-v1",
        "status": "PASS",
        "fold": fold,
        "seed": 0,
        "source": "crossfit-independent row materialized from frozen seed-0 screen",
        "routes": {
            detector: selection["detection_routes"][fold][detector]["selected"]
            for detector in field_common.DETECTORS
        },
        "parent_rows": rows,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _detection_eval_unit(
    fold: str, seed: int, selection: dict[str, Any], batch_size: int
) -> dict[str, Any]:
    c009_common, d2_eval, multi, detector_protocol, d2, cv, external, tail = _detection_modules()
    legacy, external_protocol = tail
    cohort = field_common.read_csv(D2_COHORT)
    fold_rows = [row for row in cohort if row["tunnel"] == fold]
    device = torch.device("cuda")
    model, provenance = c010.load_model(seed, device)
    detectors = d2_eval.load_detectors(
        fold,
        DETECTOR_SEEDS[seed],
        multi,
        detector_protocol,
        d2,
        legacy,
        external_protocol,
        device,
    )
    dataset = external.ExternalSyntheticDataset(fold_rows, external_protocol)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    bins = int(detector_protocol["detectors"]["histogram_bins"])
    threshold = float(detector_protocol["detectors"]["fixed_threshold"])
    accumulators = {
        detector: collections.defaultdict(
            lambda: {condition: external.Accumulator(bins) for condition in ("raw", "field")}
        )
        for detector in field_common.DETECTORS
    }
    routes = {
        detector: selection["detection_routes"][fold][detector]["selected"]
        for detector in field_common.DETECTORS
    }
    active = sorted({condition for condition in routes.values() if condition != profiles.IDENTITY})
    started = time.perf_counter()
    for batch_index, (image_tensor, degraded_tensor, mask_tensor, parents, metadata) in enumerate(loader, 1):
        del degraded_tensor, metadata
        raw = image_tensor.numpy().astype(np.uint8, copy=False)
        masks = mask_tensor.numpy().astype(np.uint8, copy=False)
        normalized = raw.astype(np.float32) / np.float32(255.0)
        zeros = np.zeros(len(raw), dtype=np.float32)
        tensor = torch.from_numpy(c009_common.prompt_tensor(normalized, zeros, zeros)).to(device)
        with c010.pilot.autocast(device):
            outputs = profiles.make_candidates(tensor, model.base, active)
        candidates = encode_candidates(outputs)
        for detector_name, detector in detectors.items():
            condition = routes[detector_name]
            field_image = raw if condition == profiles.IDENTITY else candidates[condition]
            images = {"raw": raw, "field": field_image}
            probabilities = {
                name: external.detect_batch(image, detector, device)
                for name, image in images.items()
            }
            for sample_index, parent in enumerate(parents):
                for name in ("raw", "field"):
                    external.update_accumulator(
                        accumulators[detector_name][parent][name],
                        probabilities[name][sample_index : sample_index + 1],
                        masks[sample_index : sample_index + 1],
                        raw[sample_index : sample_index + 1],
                        images[name][sample_index : sample_index + 1],
                        cv,
                        threshold,
                        False,
                    )
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(json.dumps({"stage": "eval_detection", "fold": fold, "seed": seed, "batch": batch_index, "batches": len(loader)}), flush=True)
    parent_rows = []
    for detector_name, by_parent in accumulators.items():
        for parent, by_condition in by_parent.items():
            raw_metric = external.finalize_accumulator(by_condition["raw"], d2, cv)
            field_metric = external.finalize_accumulator(by_condition["field"], d2, cv)
            burden = d2_eval.matched_sensitivity_burden(by_condition["field"], raw_metric)
            parent_rows.append(
                {
                    "tunnel": fold,
                    "detector": detector_name,
                    "parent": parent,
                    "condition": routes[detector_name],
                    "delta_auprc": float(field_metric["pixel_auprc"] - raw_metric["pixel_auprc"]),
                    "delta_f1": float(field_metric["f1_at_0_5"] - raw_metric["f1_at_0_5"]),
                    "delta_matched_fpr": float(burden["delta_false_positive_pixel_rate_restored_minus_degraded"]),
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-EVAL-UNIT-v1",
        "status": "PASS",
        "fold": fold,
        "seed": seed,
        "source": "fresh outer-fold evaluation",
        "routes": routes,
        "tile_count": len(dataset),
        "parent_rows": parent_rows,
        "base_provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    del model, detectors, dataset, loader, accumulators
    torch.cuda.empty_cache()
    return payload


def evaluate_detection(batch_size: int = 12) -> dict[str, Any]:
    selection = verify_selection()
    progress = RUNS / "evaluation_v1" / "detection" / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for seed in SEEDS:
        for fold in TUNNELS:
            path = progress / f"fold_{fold}_seed{seed}.json"
            if not _valid_payload(path, "MERIT-C016-DETECTION-EVAL-UNIT-v1"):
                payload = (
                    _derive_detection_seed0(fold, selection)
                    if seed == 0
                    else _detection_eval_unit(fold, seed, selection, batch_size)
                )
                write_json_atomic(path, payload)
    files = sorted(progress.glob("fold_*_seed*.json"))
    if len(files) != len(TUNNELS) * len(SEEDS):
        raise C016Error(f"Expected 12 detection evaluation units, found {len(files)}")
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-EVAL-COMPLETE-v1",
        "status": "PASS_COMPLETE",
        "units": len(files),
        "selection_payload_sha256": selection["payload_sha256"],
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    write_json_atomic(RUNS / "evaluation_v1" / "detection" / "RUN_COMPLETE.json", payload)
    return payload


def _derive_width_seed0(
    tunnel: str, parent: str, selection: dict[str, Any]
) -> dict[str, Any]:
    condition = selection["width_routes"][tunnel]["selected"]
    path = RUNS / "screen_v1" / "width" / "progress" / f"tunnel_{tunnel}_parent_{parent}_seed0.json"
    screen = load_json(path)
    measurements = [
        row for row in screen["measurements"] if row["condition"] == condition
    ]
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-WIDTH-EVAL-UNIT-v1",
        "status": "PASS",
        "tunnel": tunnel,
        "parent": parent,
        "seed": 0,
        "condition": condition,
        "source": "crossfit-independent row materialized from frozen seed-0 screen",
        "measurements": measurements,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _width_eval_unit(
    tunnel: str,
    parent: str,
    seed: int,
    rows: list[dict[str, str]],
    selection: dict[str, Any],
    model: Any,
    provenance: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    condition = selection["width_routes"][tunnel]["selected"]
    by_image: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    image_groups = sorted(by_image.items())
    measurements = []
    started = time.perf_counter()
    for start in range(0, len(image_groups), batch_size):
        chunk = image_groups[start : start + batch_size]
        raw_images = [_load_width_image(group_rows[0]) for _, group_rows in chunk]
        batch = np.stack(raw_images).astype(np.float32, copy=False)
        candidate_images = batch
        if condition != profiles.IDENTITY:
            zeros = np.zeros(len(batch), dtype=np.float32)
            inputs = c010.pilot.common.prompt_tensor(batch, zeros, zeros)
            tensor = torch.from_numpy(inputs).to(device)
            with c010.pilot.autocast(device):
                output = profiles.make_candidates(tensor, model.base, [condition])[condition]
            candidate_images = output[:, 0].float().cpu().numpy().astype(np.float32)
        for item_index, ((image_id, group_rows), raw) in enumerate(zip(chunk, raw_images)):
            del image_id
            candidate = candidate_images[item_index]
            for annotation in group_rows:
                coordinates = _coordinates(annotation)
                raw_measurement = width_common.estimate_dark_width(raw, coordinates)
                candidate_measurement = width_common.estimate_dark_width(candidate, coordinates)
                measurements.append(
                    _width_row(annotation, condition, raw_measurement, candidate_measurement)
                )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-WIDTH-EVAL-UNIT-v1",
        "status": "PASS",
        "tunnel": tunnel,
        "parent": parent,
        "seed": seed,
        "condition": condition,
        "source": "fresh outer-fold evaluation",
        "measurements": measurements,
        "base_provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def evaluate_width(batch_size: int = 12) -> dict[str, Any]:
    selection = verify_selection()
    with CROSS_SECTIONS.open(encoding="utf-8", newline="") as stream:
        annotations = list(csv.DictReader(stream))
    by_unit: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in annotations:
        by_unit[(row["tunnel"], row["parent"])].append(row)
    progress = RUNS / "evaluation_v1" / "width" / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    started = time.perf_counter()
    for seed in SEEDS:
        model = provenance = None
        if seed > 0:
            missing = [
                unit
                for unit in sorted(by_unit)
                if not _valid_payload(
                    progress / f"tunnel_{unit[0]}_parent_{unit[1]}_seed{seed}.json",
                    "MERIT-C016-WIDTH-EVAL-UNIT-v1",
                )
            ]
            if missing:
                model, provenance = c010.load_model(seed, device)
        for tunnel, parent in sorted(by_unit):
            path = progress / f"tunnel_{tunnel}_parent_{parent}_seed{seed}.json"
            if not _valid_payload(path, "MERIT-C016-WIDTH-EVAL-UNIT-v1"):
                if seed == 0:
                    payload = _derive_width_seed0(tunnel, parent, selection)
                else:
                    assert model is not None and provenance is not None
                    payload = _width_eval_unit(
                        tunnel,
                        parent,
                        seed,
                        by_unit[(tunnel, parent)],
                        selection,
                        model,
                        provenance,
                        device,
                        batch_size,
                    )
                write_json_atomic(path, payload)
        if model is not None:
            del model
            torch.cuda.empty_cache()
    files = sorted(progress.glob("tunnel_*_seed*.json"))
    expected = len(by_unit) * len(SEEDS)
    if len(files) != expected:
        raise C016Error(f"Expected {expected} width evaluation units, found {len(files)}")
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-WIDTH-EVAL-COMPLETE-v1",
        "status": "PASS_COMPLETE",
        "units": len(files),
        "selection_payload_sha256": selection["payload_sha256"],
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    write_json_atomic(RUNS / "evaluation_v1" / "width" / "RUN_COMPLETE.json", payload)
    return payload


def stratified_bootstrap(
    rows: list[dict[str, Any]], key: str, seed: int
) -> dict[str, Any]:
    by_tunnel: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        by_tunnel[row["tunnel"]].append(float(row[key]))
    if tuple(sorted(by_tunnel)) != TUNNELS:
        raise C016Error(f"Bootstrap expected {TUNNELS}, found {tuple(sorted(by_tunnel))}")
    per_tunnel = {tunnel: mean(by_tunnel[tunnel]) for tunnel in TUNNELS}
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        tunnel_means = []
        for tunnel in TUNNELS:
            values = np.asarray(by_tunnel[tunnel], dtype=np.float64)
            tunnel_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        samples[index] = mean(tunnel_means)
    return {
        "mean": mean(per_tunnel.values()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "per_tunnel": per_tunnel,
        "positive_tunnels": sum(value > 0.0 for value in per_tunnel.values()),
        "improved_tunnels": sum(value < 0.0 for value in per_tunnel.values()),
        "clusters": len(rows),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
    }


def summarize_detection() -> dict[str, Any]:
    files = sorted((RUNS / "evaluation_v1" / "detection" / "progress").glob("fold_*_seed*.json"))
    if len(files) != len(TUNNELS) * len(SEEDS):
        raise C016Error("Detection evaluation is incomplete")
    raw_rows = []
    for path in files:
        payload = load_json(path)
        if payload_sha256(payload) != payload.get("payload_sha256"):
            raise C016Error(f"Detection evaluation hash mismatch: {path}")
        for row in payload["parent_rows"]:
            raw_rows.append({**row, "seed": int(payload["seed"])})
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_rows:
        grouped[(row["tunnel"], row["detector"], row["parent"])].append(row)
    detector_parent_rows = []
    for (tunnel, detector, parent), rows in sorted(grouped.items()):
        detector_parent_rows.append(
            {
                "tunnel": tunnel,
                "detector": detector,
                "parent": parent,
                **{
                    key: mean(row[key] for row in rows)
                    for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
                },
            }
        )
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in detector_parent_rows:
        by_parent[(row["tunnel"], row["parent"])].append(row)
    panel_rows = [
        {
            "tunnel": tunnel,
            "parent": parent,
            **{
                key: mean(row[key] for row in rows)
                for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
            },
        }
        for (tunnel, parent), rows in sorted(by_parent.items())
    ]
    panel_auprc = stratified_bootstrap(panel_rows, "delta_auprc", BOOTSTRAP_SEED)
    panel_f1 = stratified_bootstrap(panel_rows, "delta_f1", BOOTSTRAP_SEED + 1)
    panel_fpr = stratified_bootstrap(panel_rows, "delta_matched_fpr", BOOTSTRAP_SEED + 2)
    detectors = {}
    for index, detector in enumerate(field_common.DETECTORS):
        rows = [row for row in detector_parent_rows if row["detector"] == detector]
        detectors[detector] = {
            "delta_auprc": stratified_bootstrap(rows, "delta_auprc", BOOTSTRAP_SEED + 10 + index),
            "delta_f1": stratified_bootstrap(rows, "delta_f1", BOOTSTRAP_SEED + 20 + index),
            "delta_matched_fpr": stratified_bootstrap(rows, "delta_matched_fpr", BOOTSTRAP_SEED + 30 + index),
        }
    c014 = load_json(C014_SUMMARY)
    v23_mean = float(
        c014["detection"]["existing_four_tunnel_descriptive_delta_auprc"]["mean"]
    )
    detector_nonnegative = all(value["delta_auprc"]["mean"] >= 0.0 for value in detectors.values())
    gate = bool(
        panel_auprc["mean"] > 0.0
        and panel_auprc["ci95_low"] > 0.0
        and panel_auprc["positive_tunnels"] == 4
        and panel_fpr["mean"] <= 0.0
        and detector_nonnegative
        and panel_auprc["mean"] > v23_mean
    )
    return {
        "panel": {
            "delta_auprc": panel_auprc,
            "delta_f1": panel_f1,
            "delta_matched_fpr": panel_fpr,
        },
        "detectors": detectors,
        "v23_four_tunnel_mean_delta_auprc": v23_mean,
        "delta_over_v23": panel_auprc["mean"] - v23_mean,
        "gate": {
            "pass": gate,
            "requirements": load_json(PROTOCOL)["final_gates"]["detection"],
        },
        "panel_parent_rows": panel_rows,
    }


def summarize_width() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files = sorted((RUNS / "evaluation_v1" / "width" / "progress").glob("tunnel_*_seed*.json"))
    if len(files) != 14 * len(SEEDS):
        raise C016Error(f"Expected 42 width evaluation units, found {len(files)}")
    by_roi: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for path in files:
        payload = load_json(path)
        if payload_sha256(payload) != payload.get("payload_sha256"):
            raise C016Error(f"Width evaluation hash mismatch: {path}")
        for measurement in payload["measurements"]:
            by_roi[measurement["roi_id"]].append(measurement)
    outcomes = []
    for roi_id, rows in sorted(by_roi.items()):
        if len(rows) != len(SEEDS):
            raise C016Error(f"Expected three width seeds for {roi_id}, found {len(rows)}")
        raw_statuses = [row["raw_status"] for row in rows]
        if len(set(raw_statuses)) != 1:
            raise C016Error(f"Raw width status changed across seeds: {roi_id}")
        raw_values = [float(row["raw_estimate_px"]) for row in rows if row["raw_status"] == "ESTIMABLE"]
        if raw_values and max(raw_values) - min(raw_values) > 1e-12:
            raise C016Error(f"Raw width estimate changed across seeds: {roi_id}")
        effective_values = [
            float(row["effective_estimate_px"])
            for row in rows
            if row["effective_status"] == "ESTIMABLE"
        ]
        first = rows[0]
        raw_status = raw_statuses[0]
        effective_status = "ESTIMABLE" if len(effective_values) >= 2 else "NON_ESTIMABLE"
        raw_estimate = raw_values[0] if raw_status == "ESTIMABLE" else None
        effective_estimate = mean(effective_values) if effective_status == "ESTIMABLE" else None
        reference = float(first["reference_width_px"])
        delta = None
        if raw_estimate is not None and effective_estimate is not None:
            delta = abs(effective_estimate - reference) - abs(raw_estimate - reference)
        outcomes.append(
            {
                "roi_id": roi_id,
                "image_id": first["image_id"],
                "tunnel": first["tunnel"],
                "parent": first["parent"],
                "condition": first["condition"],
                "reference_width_px": reference,
                "raw_status": raw_status,
                "effective_status": effective_status,
                "raw_estimate_px": raw_estimate,
                "effective_estimate_px": effective_estimate,
                "delta_absolute_error_px": delta,
                "fallback_seed_count": sum(bool(row["safe_fallback"]) for row in rows),
            }
        )
    by_tile: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for row in outcomes:
        if row["delta_absolute_error_px"] is not None:
            by_tile[(row["tunnel"], row["parent"], row["image_id"])].append(
                float(row["delta_absolute_error_px"])
            )
    by_parent: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for (tunnel, parent, image_id), values in by_tile.items():
        del image_id
        by_parent[(tunnel, parent)].append(mean(values))
    parent_rows = [
        {"tunnel": tunnel, "parent": parent, "delta_absolute_error_px": mean(values)}
        for (tunnel, parent), values in sorted(by_parent.items())
    ]
    delta = stratified_bootstrap(parent_rows, "delta_absolute_error_px", BOOTSTRAP_SEED + 100)
    raw_nonestimable = mean(row["raw_status"] != "ESTIMABLE" for row in outcomes)
    effective_nonestimable = mean(row["effective_status"] != "ESTIMABLE" for row in outcomes)
    paired_rate = mean(row["delta_absolute_error_px"] is not None for row in outcomes)
    coverage = {
        "rois": len(outcomes),
        "tiles": len({row["image_id"] for row in outcomes}),
        "parents": len({(row["tunnel"], row["parent"]) for row in outcomes}),
        "tunnels": len({row["tunnel"] for row in outcomes}),
    }
    coverage_pass = bool(
        coverage["rois"] >= 300
        and coverage["tiles"] >= 100
        and coverage["parents"] >= 10
        and coverage["tunnels"] == 4
    )
    gate = bool(
        coverage_pass
        and delta["mean"] < 0.0
        and delta["ci95_high"] < 0.0
        and delta["improved_tunnels"] >= 3
        and effective_nonestimable - raw_nonestimable <= 0.02
        and paired_rate >= 0.70
    )
    v23 = load_json(C014_SUMMARY)["natural_crack_apparent_width"]
    return (
        {
            "coverage": coverage,
            "coverage_pass": coverage_pass,
            "delta_absolute_error_px": delta,
            "raw_nonestimable_rate": raw_nonestimable,
            "effective_nonestimable_rate": effective_nonestimable,
            "paired_estimable_rate": paired_rate,
            "safe_fallback_rois": sum(row["fallback_seed_count"] > 0 for row in outcomes),
            "v23": {
                "delta_absolute_error_px": float(v23["delta_absolute_error_px"]["mean"]),
                "paired_estimable_rate": float(v23["paired_estimable_rate"]),
            },
            "changes_from_v23": {
                "delta_absolute_error_px": delta["mean"] - float(v23["delta_absolute_error_px"]["mean"]),
                "paired_estimable_rate": paired_rate - float(v23["paired_estimable_rate"]),
            },
            "gate": {
                "pass": gate,
                "requirements": load_json(PROTOCOL)["final_gates"]["natural_apparent_width"],
            },
            "claim_boundary": load_json(PROTOCOL)["claim_boundary"],
        },
        outcomes,
    )


def summarize() -> dict[str, Any]:
    protocol = verify_protocol()
    selection = verify_selection()
    detection = summarize_detection()
    width, width_outcomes = summarize_width()
    outcome_csv = RUNS / "evaluation_v1" / "natural_width_outcomes.csv"
    outcome_csv.parent.mkdir(parents=True, exist_ok=True)
    with outcome_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(width_outcomes[0]))
        writer.writeheader()
        writer.writerows(width_outcomes)
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-CROSSFIT-SUMMARY-v1",
        "status": (
            "PASS_COMPLETE"
            if detection["gate"]["pass"] and width["gate"]["pass"]
            else "COMPLETE_GATE_FAIL"
        ),
        "scientific_status": "CROSS_FITTED_DEVELOPMENT_ON_EXISTING_FOUR_TUNNELS",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "selection_payload_sha256": selection["payload_sha256"],
        "detection": detection,
        "natural_crack_apparent_width": width,
        "global_gate_pass": bool(detection["gate"]["pass"] and width["gate"]["pass"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["payload_sha256"] = payload_sha256(payload)
    write_json_atomic(RUNS / "c016_final_summary.json", payload)
    return payload


def run_all(batch_size: int = 12) -> dict[str, Any]:
    freeze_protocol()
    screen_detection(batch_size)
    screen_width(batch_size)
    select_routes()
    evaluate_detection(batch_size)
    evaluate_width(batch_size)
    return summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "screen-detection",
            "screen-width",
            "select",
            "eval-detection",
            "eval-width",
            "summarize",
            "run",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    commands = {
        "freeze": freeze_protocol,
        "screen-detection": lambda: screen_detection(args.batch_size),
        "screen-width": lambda: screen_width(args.batch_size),
        "select": select_routes,
        "eval-detection": lambda: evaluate_detection(args.batch_size),
        "eval-width": lambda: evaluate_width(args.batch_size),
        "summarize": summarize,
        "run": lambda: run_all(args.batch_size),
    }
    result = commands[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
