#!/usr/bin/env python3
"""Evaluate frozen MERIT restorers on tunnel-held-out D2 crack tiles.

The analysis is deliberately post-hoc and synthetic: expert-labelled D2 source
tiles are degraded with the manuscript's fixed joint blur/noise profile, restored
by frozen C006 checkpoints, and scored by D2 detectors that never saw the held
out tunnel.  All write operations are confined to a versioned artifact folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol.json"
SCHEMA = "MERIT-C007-D2-EXTERNAL-SYNTHETIC-v1"
CONDITIONS = ("reference", "degraded", "task_off", "task_on")


class ProtocolError(RuntimeError):
    """The frozen protocol, data, or runtime environment is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ProtocolError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ProtocolError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ProtocolError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA:
        raise ProtocolError("Unexpected protocol schema")
    expected = protocol["data"]["selection"]["expected_per_parent"]
    if sum(int(value) for value in expected.values()) != int(
        protocol["data"]["selection"]["expected_tiles"]
    ):
        raise ProtocolError("Expected per-parent tile counts do not sum to total")
    degradation = protocol["degradation"]
    if (
        float(degradation["sigma_h_px"]),
        float(degradation["sigma_v_px"]),
        float(degradation["noise_std_dn"]),
        int(degradation["noise_replicates"]),
    ) != (2.6, 2.4, 2.5, 1):
        raise ProtocolError("Frozen joint degradation changed")
    pairings = protocol["detectors"]["seed_pairing"]
    if [int(row["chain"]) for row in pairings] != [0, 1, 2]:
        raise ProtocolError("Seed-chain pairing changed")
    if protocol["statistics"]["conditions"] != list(CONDITIONS):
        raise ProtocolError("Condition list changed")
    return protocol


def import_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ProtocolError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def verify_static_assets(protocol: Mapping[str, Any]) -> dict[str, str]:
    data = protocol["data"]
    restoration = protocol["restoration"]
    assets = {
        "cv_protocol": (Path(data["cv_protocol_path"]), data["cv_protocol_sha256"]),
        "cv_code": (Path(data["cv_code_path"]), data["cv_code_sha256"]),
        "d2_code": (Path(data["d2_code_path"]), data["d2_code_sha256"]),
        "d2_protocol": (Path(data["d2_protocol_path"]), data["d2_protocol_sha256"]),
        "upstream_audit": (
            Path(data["upstream_audit_path"]),
            data["upstream_audit_sha256"],
        ),
        "upstream_manifest": (
            Path(data["upstream_manifest_path"]),
            data["upstream_manifest_sha256"],
        ),
        "restoration_protocol": (
            Path(restoration["source_protocol_path"]),
            restoration["source_protocol_sha256"],
        ),
        "restoration_model_code": (
            Path(restoration["model_code_path"]),
            restoration["model_code_sha256"],
        ),
    }
    return {
        label: require_hash(path, expected, label)
        for label, (path, expected) in assets.items()
    }


def load_source_modules(protocol: Mapping[str, Any]):
    data = protocol["data"]
    d2 = import_module(Path(data["d2_code_path"]), "merit_d2_external_frozen_detector")
    cv = import_module(Path(data["cv_code_path"]), "merit_d2_external_frozen_cv")
    model = import_module(
        Path(protocol["restoration"]["model_code_path"]),
        "merit_d2_external_frozen_restorer",
    )
    return d2, cv, model


def select_pairs(protocol: Mapping[str, Any], d2: Any, cv: Any) -> dict[str, list[Any]]:
    d2_root = Path(protocol["data"]["root"])
    cv_protocol = cv.load_protocol(Path(protocol["data"]["cv_protocol_path"]))
    d2_protocol = d2.load_protocol(Path(protocol["data"]["d2_protocol_path"]))
    pairs = cv.collect_parent_pairs(d2_root, cv_protocol, d2_protocol)
    selected: dict[str, list[Any]] = {}
    expected = protocol["data"]["selection"]["expected_per_parent"]
    for parent, parent_pairs in pairs.items():
        kind = cv.artifact_kind(parent, cv_protocol)
        if kind == "positive_enriched_overlap":
            kept = [
                pair
                for pair in parent_pairs
                if pair.row % 4 == 0 and pair.column % 4 == 0
            ]
        elif kind == "full_grid":
            kept = list(parent_pairs)
        else:
            raise ProtocolError(f"Unexpected artifact kind for {parent}: {kind}")
        if len(kept) != int(expected[parent]):
            raise ProtocolError(
                f"Selected tile count changed for {parent}: {len(kept)} != {expected[parent]}"
            )
        coordinates = [(pair.row, pair.column) for pair in kept]
        if len(coordinates) != len(set(coordinates)):
            raise ProtocolError(f"Duplicate selected coordinates for {parent}")
        if kind == "positive_enriched_overlap":
            coordinate_set = set(coordinates)
            for row, column in coordinates:
                if any(
                    (row + dy, column + dx) in coordinate_set
                    for dy, dx in ((0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0))
                ):
                    raise ProtocolError(f"Selected lattice still overlaps for {parent}")
        selected[parent] = sorted(kept, key=lambda pair: pair.name)
    if sum(map(len, selected.values())) != int(
        protocol["data"]["selection"]["expected_tiles"]
    ):
        raise ProtocolError("Total selected tile count changed")
    return selected


def upstream_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    path = Path(protocol["data"]["upstream_manifest_path"])
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            result[str(row["relative_path"]).replace("\\", "/")] = str(row["sha256"])
    return result


def cohort_rows(
    protocol: Mapping[str, Any], selected: Mapping[str, Sequence[Any]], cv: Any
) -> list[dict[str, Any]]:
    d2_root = Path(protocol["data"]["root"])
    cv_protocol = cv.load_protocol(Path(protocol["data"]["cv_protocol_path"]))
    lookup = upstream_hashes(protocol)
    tunnel_lookup = cv.tunnel_by_parent(cv_protocol)
    rows: list[dict[str, Any]] = []
    index = 0
    for parent in sorted(selected):
        kind = cv.artifact_kind(parent, cv_protocol)
        for pair in selected[parent]:
            image_relative = pair.image_path.relative_to(d2_root).as_posix()
            mask_relative = pair.mask_path.relative_to(d2_root).as_posix()
            if image_relative not in lookup or mask_relative not in lookup:
                raise ProtocolError(f"Selected file absent from upstream manifest: {pair.name}")
            image_hash = require_hash(pair.image_path, lookup[image_relative], pair.name + " image")
            mask_hash = require_hash(pair.mask_path, lookup[mask_relative], pair.name + " mask")
            rows.append(
                {
                    "cohort_index": index,
                    "tunnel": tunnel_lookup[parent],
                    "parent": parent,
                    "artifact_kind": kind,
                    "name": pair.name,
                    "row": pair.row,
                    "column": pair.column,
                    "image_path": str(pair.image_path.resolve()),
                    "mask_path": str(pair.mask_path.resolve()),
                    "image_sha256": image_hash,
                    "mask_sha256": mask_hash,
                }
            )
            index += 1
    return rows


def command_audit(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    assets = verify_static_assets(protocol)
    d2, cv, _ = load_source_modules(protocol)
    selected = select_pairs(protocol, d2, cv)
    rows = cohort_rows(protocol, selected, cv)
    output = args.output_dir.expanduser().resolve()
    cohort_path = output / "cohort.csv"
    write_csv_exclusive(cohort_path, rows)
    content_lines = [
        f"{row['image_sha256']}  {row['image_path']}\n{row['mask_sha256']}  {row['mask_path']}"
        for row in rows
    ]
    freeze = {
        "schema_version": SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "created_utc": utc_now(),
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "static_asset_hashes": assets,
        "cohort_path": str(cohort_path),
        "cohort_sha256": sha256_file(cohort_path),
        "selected_content_set_sha256": sha256_text("\n".join(content_lines)),
        "tile_count": len(rows),
        "parent_count": len({row["parent"] for row in rows}),
        "tunnel_count": len({row["tunnel"] for row in rows}),
        "per_parent": {
            parent: sum(row["parent"] == parent for row in rows)
            for parent in sorted({row["parent"] for row in rows})
        },
    }
    write_json_exclusive(output / "freeze.json", freeze)
    print(json.dumps(freeze, indent=2, ensure_ascii=False), flush=True)
    return 0


def deterministic_seed(base_seed: int, tile_name: str) -> int:
    payload = f"{base_seed}|{tile_name}|joint_ref|0".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError(f"Could not decode image: {path}")
    return image


def read_mask(path: Path, tile_size: int) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != (tile_size, tile_size) or mask.dtype != np.uint8:
        raise ProtocolError(f"Invalid mask: {path}")
    if not np.isin(np.unique(mask), np.asarray([0, 1], dtype=np.uint8)).all():
        raise ProtocolError(f"Mask is not binary: {path}")
    return mask


def degrade_gray(clean: np.ndarray, protocol: Mapping[str, Any], tile_name: str) -> np.ndarray:
    degradation = protocol["degradation"]
    blurred = gaussian_filter(
        clean.astype(np.float32),
        sigma=(float(degradation["sigma_v_px"]), float(degradation["sigma_h_px"])),
        mode="reflect",
    )
    rng = np.random.default_rng(
        deterministic_seed(int(degradation["base_seed"]), tile_name)
    )
    blurred += rng.normal(
        0.0, float(degradation["noise_std_dn"]), size=clean.shape
    ).astype(np.float32)
    return np.rint(np.clip(blurred, 0.0, 255.0)).astype(np.uint8)


class ExternalSyntheticDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> None:
        self.rows = list(rows)
        self.protocol = protocol
        self.tile_size = int(protocol["image_contract"]["tile_size"])
        if not self.rows:
            raise ProtocolError("Evaluation dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = read_image_bgr(Path(str(row["image_path"])))
        if image.shape[:2] != (self.tile_size, self.tile_size):
            raise ProtocolError(f"Unexpected image shape: {row['image_path']}")
        clean = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        degraded = degrade_gray(clean, self.protocol, str(row["name"]))
        mask = read_mask(Path(str(row["mask_path"])), self.tile_size)
        return (
            torch.from_numpy(clean),
            torch.from_numpy(degraded),
            torch.from_numpy(mask),
            str(row["parent"]),
            str(row["name"]),
        )


def worker_seed(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ProtocolError("D2 external synthetic evaluation requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda")


def load_restoration_source(protocol: Mapping[str, Any]) -> dict[str, Any]:
    source_path = Path(protocol["restoration"]["source_protocol_path"])
    return json.loads(source_path.read_text(encoding="utf-8"))


def restoration_record(
    source: Mapping[str, Any], arm: str, seed: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in source["models"]["checkpoints"]
        if row["arm"] == arm and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ProtocolError(f"Expected one restoration checkpoint for {arm} seed {seed}")
    return matches[0]


def load_models(
    protocol: Mapping[str, Any],
    d2: Any,
    restorer_module: Any,
    fold: str,
    pairing: Mapping[str, Any],
    device: torch.device,
):
    detector_seed = str(int(pairing["detector_seed"]))
    detector_record = protocol["detectors"]["checkpoints"][fold][detector_seed]
    detector_path = Path(detector_record["path"])
    require_hash(detector_path, detector_record["sha256"], f"fold {fold} detector {detector_seed}")
    detector_checkpoint = torch.load(detector_path, map_location="cpu", weights_only=True)
    detector = d2.ResNet34UNet()
    detector.load_state_dict(detector_checkpoint["model_state"], strict=True)
    detector = detector.to(device).eval()

    source = load_restoration_source(protocol)
    restorer_seed = int(pairing["restorer_seed"])
    restorers: dict[str, torch.nn.Module] = {}
    with torch.serialization.safe_globals([TorchVersion]):
        for arm in ("task_off", "task_on"):
            record = restoration_record(source, arm, restorer_seed)
            path = Path(record["path"])
            require_hash(path, record["sha256"], f"{arm} seed {restorer_seed}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = restorer_module.NAFNetUNet(
                width=int(protocol["restoration"]["model_width"])
            )
            model.load_state_dict(checkpoint["state"], strict=True)
            restorers[arm] = model.to(device).eval()
    return detector, restorers


@torch.inference_mode()
def restore_batch(
    degraded_u8: torch.Tensor, model: torch.nn.Module, device: torch.device
) -> np.ndarray:
    gray = degraded_u8.to(device, non_blocking=True).float().div_(255.0)
    inputs = torch.zeros(
        (len(gray), 3, gray.shape[-2], gray.shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    inputs[:, 0] = gray
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(inputs)[:, 0].float()
    return np.rint(
        np.clip(prediction.cpu().numpy(), 0.0, 1.0) * np.float32(255.0)
    ).astype(np.uint8)


@torch.inference_mode()
def detect_batch(
    gray_u8: np.ndarray | torch.Tensor,
    detector: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    if isinstance(gray_u8, np.ndarray):
        tensor = torch.from_numpy(gray_u8)
    else:
        tensor = gray_u8
    gray = tensor.to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)
    rgb = gray.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    inputs = (rgb - mean) / std
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = detector(inputs)
    return torch.sigmoid(logits[:, 0].float()).cpu().numpy()


@dataclass
class Accumulator:
    bins: int
    positive: np.ndarray = field(init=False)
    negative: np.ndarray = field(init=False)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    tile_count: int = 0
    negative_tiles: int = 0
    negative_tiles_predicted: int = 0
    psnr_sum: float = 0.0
    gradient_mae_sum: float = 0.0
    paired_metric_count: int = 0

    def __post_init__(self) -> None:
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)


def paired_image_metrics(clean: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    clean_float = clean.astype(np.float32)
    candidate_float = candidate.astype(np.float32)
    residual = candidate_float - clean_float
    mse = float(np.mean(residual * residual))
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    dx = np.diff(candidate_float, axis=1) - np.diff(clean_float, axis=1)
    dy = np.diff(candidate_float, axis=0) - np.diff(clean_float, axis=0)
    gradient = float((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2.0)
    return psnr, gradient


def update_accumulator(
    accumulator: Accumulator,
    probabilities: np.ndarray,
    masks: np.ndarray,
    clean: np.ndarray,
    candidate: np.ndarray,
    cv: Any,
    threshold: float,
    paired_metrics_enabled: bool,
) -> None:
    for probability, mask, clean_tile, candidate_tile in zip(
        probabilities, masks, clean, candidate
    ):
        cv.update_histogram(
            accumulator.positive,
            accumulator.negative,
            probability.ravel(),
            mask.ravel(),
        )
        tp, fp, fn, tn = cv.confusion_at_threshold(probability, mask, threshold)
        accumulator.tp += tp
        accumulator.fp += fp
        accumulator.fn += fn
        accumulator.tn += tn
        accumulator.tile_count += 1
        if not mask.any():
            accumulator.negative_tiles += 1
            accumulator.negative_tiles_predicted += int((probability >= threshold).any())
        if paired_metrics_enabled:
            psnr, gradient = paired_image_metrics(clean_tile, candidate_tile)
            if not math.isfinite(psnr):
                raise ProtocolError("Non-reference paired PSNR is infinite")
            accumulator.psnr_sum += psnr
            accumulator.gradient_mae_sum += gradient
            accumulator.paired_metric_count += 1


def finalize_accumulator(
    accumulator: Accumulator, d2: Any, cv: Any
) -> dict[str, Any]:
    counts = cv.metrics_from_counts(
        accumulator.tp, accumulator.fp, accumulator.fn, accumulator.tn
    )
    positive_pixels = int(accumulator.positive.sum())
    negative_pixels = int(accumulator.negative.sum())
    if positive_pixels == 0:
        raise ProtocolError("Selected parent has no positive GT pixels")
    return {
        "tile_count": accumulator.tile_count,
        "positive_pixels": positive_pixels,
        "negative_pixels": negative_pixels,
        "pixel_auprc": float(
            d2.ParentHistogramAUPRC.average_precision(
                accumulator.positive, accumulator.negative
            )
        ),
        **counts,
        "negative_tiles": accumulator.negative_tiles,
        "negative_tile_any_positive_rate_at_0_5": (
            accumulator.negative_tiles_predicted / accumulator.negative_tiles
            if accumulator.negative_tiles
            else None
        ),
        "paired_psnr_db": (
            accumulator.psnr_sum / accumulator.paired_metric_count
            if accumulator.paired_metric_count
            else None
        ),
        "paired_gradient_mae_dn": (
            accumulator.gradient_mae_sum / accumulator.paired_metric_count
            if accumulator.paired_metric_count
            else None
        ),
    }


def verify_cohort_files(rows: Sequence[Mapping[str, str]]) -> None:
    for row in rows:
        require_hash(Path(row["image_path"]), row["image_sha256"], row["name"] + " image")
        require_hash(Path(row["mask_path"]), row["mask_sha256"], row["name"] + " mask")


def load_frozen_cohort(
    protocol_path: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    protocol = load_protocol(protocol_path)
    freeze_path = output_dir / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require_hash(protocol_path, freeze["protocol_sha256"], "frozen protocol")
    require_hash(Path(freeze["code_path"]), freeze["code_sha256"], "frozen analysis code")
    cohort_path = Path(freeze["cohort_path"])
    require_hash(cohort_path, freeze["cohort_sha256"], "frozen cohort")
    rows = read_csv(cohort_path)
    if len(rows) != int(protocol["data"]["selection"]["expected_tiles"]):
        raise ProtocolError("Frozen cohort row count changed")
    verify_cohort_files(rows)
    return protocol, freeze, rows


def evaluation_rows(
    result: Mapping[str, Any], fold: str, pairing: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent, conditions in result["parents"].items():
        for condition, metrics in conditions.items():
            rows.append(
                {
                    "fold": fold,
                    "chain": int(pairing["chain"]),
                    "detector_seed": int(pairing["detector_seed"]),
                    "restorer_seed": int(pairing["restorer_seed"]),
                    "parent": parent,
                    "condition": condition,
                    **metrics,
                }
            )
    return rows


def run_fold_chain(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    fold: str,
    pairing: Mapping[str, Any],
    output_path: Path,
    *,
    workers: int | None = None,
    maximum_rows: int | None = None,
) -> dict[str, Any]:
    fold_rows = [row for row in rows if row["tunnel"] == fold]
    if maximum_rows is not None:
        fold_rows = fold_rows[:maximum_rows]
    if not fold_rows:
        raise ProtocolError(f"No cohort rows for fold {fold}")
    d2, cv, restorer_module = load_source_modules(protocol)
    device = require_cuda()
    detector, restorers = load_models(
        protocol, d2, restorer_module, fold, pairing, device
    )
    generator = torch.Generator().manual_seed(20260923 + int(pairing["chain"]))
    dataset = ExternalSyntheticDataset(fold_rows, protocol)
    loader = DataLoader(
        dataset,
        batch_size=int(protocol["detectors"]["batch_size"]),
        shuffle=False,
        num_workers=(int(protocol["detectors"]["workers"]) if workers is None else workers),
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    bins = int(protocol["detectors"]["histogram_bins"])
    threshold = float(protocol["detectors"]["fixed_threshold"])
    accumulators: dict[str, dict[str, Accumulator]] = defaultdict(
        lambda: {condition: Accumulator(bins) for condition in CONDITIONS}
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    processed = 0
    total_batches = len(loader)
    for batch_index, (clean_tensor, degraded_tensor, mask_tensor, parents, _) in enumerate(
        loader, start=1
    ):
        clean = clean_tensor.numpy()
        degraded = degraded_tensor.numpy()
        masks = mask_tensor.numpy()
        off = restore_batch(degraded_tensor, restorers["task_off"], device)
        on = restore_batch(degraded_tensor, restorers["task_on"], device)
        candidates = {
            "reference": clean,
            "degraded": degraded,
            "task_off": off,
            "task_on": on,
        }
        probabilities = {
            condition: detect_batch(candidate, detector, device)
            for condition, candidate in candidates.items()
        }
        for sample_index, parent in enumerate(parents):
            for condition in CONDITIONS:
                update_accumulator(
                    accumulators[parent][condition],
                    probabilities[condition][sample_index : sample_index + 1],
                    masks[sample_index : sample_index + 1],
                    clean[sample_index : sample_index + 1],
                    candidates[condition][sample_index : sample_index + 1],
                    cv,
                    threshold,
                    condition != "reference",
                )
        processed += len(clean)
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "chain": pairing["chain"],
                        "processed": processed,
                        "total": len(dataset),
                        "batch": batch_index,
                        "batches": total_batches,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                ),
                flush=True,
            )
    result = {
        "schema_version": SCHEMA,
        "fold": fold,
        "pairing": dict(pairing),
        "tile_count": len(dataset),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu": torch.cuda.get_device_name(device),
        "parents": {
            parent: {
                condition: finalize_accumulator(accumulator, d2, cv)
                for condition, accumulator in conditions.items()
            }
            for parent, conditions in sorted(accumulators.items())
        },
    }
    write_json_exclusive(output_path, result)
    del detector, restorers
    torch.cuda.empty_cache()
    return result


def command_smoke(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    verify_static_assets(protocol)
    d2, cv, _ = load_source_modules(protocol)
    selected = select_pairs(protocol, d2, cv)
    rows = cohort_rows(protocol, selected, cv)
    pairing = protocol["detectors"]["seed_pairing"][0]
    output = args.output_dir.expanduser().resolve()
    result = run_fold_chain(
        protocol,
        rows,
        "A",
        pairing,
        output / "smoke_fold_A_chain0.json",
        workers=0,
        maximum_rows=2,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


def command_run(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    protocol, freeze, rows = load_frozen_cohort(args.protocol, output)
    verify_static_assets(protocol)
    progress_dir = output / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    total = 4 * len(protocol["detectors"]["seed_pairing"])
    completed = 0
    started = time.perf_counter()
    for fold in ("A", "B", "C", "D"):
        for pairing in protocol["detectors"]["seed_pairing"]:
            path = progress_dir / f"fold_{fold}_chain{pairing['chain']}.json"
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("fold") != fold or existing.get("pairing") != pairing:
                    raise ProtocolError(f"Invalid existing progress artifact: {path}")
            else:
                run_fold_chain(protocol, rows, fold, pairing, path)
            completed += 1
            elapsed = time.perf_counter() - started
            write_json_atomic(
                output / "status.json",
                {
                    "schema_version": SCHEMA,
                    "state": "INFERENCE_COMPLETE" if completed == total else "RUNNING",
                    "completed_fold_chains": completed,
                    "total_fold_chains": total,
                    "elapsed_seconds_this_invocation": elapsed,
                    "projected_remaining_seconds": (
                        elapsed / completed * (total - completed) if completed else None
                    ),
                    "freeze_sha256": sha256_file(output / "freeze.json"),
                    "updated_utc": utc_now(),
                },
            )
    print((output / "status.json").read_text(encoding="utf-8"), flush=True)
    return 0


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ProtocolError("Cannot average an empty sequence")
    return float(sum(materialized) / len(materialized))


def numeric_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def stratified_parent_bootstrap(
    parent_rows: Sequence[Mapping[str, Any]],
    metric: str,
    left: str,
    right: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    by_tunnel: dict[str, list[float]] = defaultdict(list)
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in parent_rows:
        grouped[(str(row["tunnel"]), str(row["parent"]))][str(row["condition"])] = float(
            row[metric]
        )
    for (tunnel, _), conditions in grouped.items():
        if left not in conditions or right not in conditions:
            raise ProtocolError("Bootstrap condition missing")
        by_tunnel[tunnel].append(conditions[left] - conditions[right])
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        tunnel_values = []
        for tunnel in ("A", "B", "C", "D"):
            values = np.asarray(by_tunnel[tunnel], dtype=np.float64)
            chosen = rng.choice(values, size=len(values), replace=True)
            tunnel_values.append(float(chosen.mean()))
        samples[index] = float(np.mean(tunnel_values))
    point = mean(mean(values) for values in by_tunnel.values())
    return {
        "point": point,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
    }


def command_summarize(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    protocol, _, _ = load_frozen_cohort(args.protocol, output)
    final_path = output / "external_synthetic_final.json"
    if final_path.exists():
        raise FileExistsError(f"Final result already exists: {final_path}")
    chain_rows: list[dict[str, Any]] = []
    for fold in ("A", "B", "C", "D"):
        for pairing in protocol["detectors"]["seed_pairing"]:
            path = output / "progress" / f"fold_{fold}_chain{pairing['chain']}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            chain_rows.extend(evaluation_rows(result, fold, pairing))
    write_csv_exclusive(output / "parent_chain_metrics.csv", chain_rows)

    metric_names = (
        "pixel_auprc",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
        "negative_tile_any_positive_rate_at_0_5",
        "paired_psnr_db",
        "paired_gradient_mae_dn",
    )
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in chain_rows:
        grouped[(str(row["fold"]), str(row["parent"]), str(row["condition"]))].append(row)
    parent_rows: list[dict[str, Any]] = []
    for (tunnel, parent, condition), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "tunnel": tunnel,
            "parent": parent,
            "condition": condition,
            "chains": len(values),
            "tile_count": int(values[0]["tile_count"]),
            "positive_pixels": int(values[0]["positive_pixels"]),
            "negative_pixels": int(values[0]["negative_pixels"]),
        }
        for metric in metric_names:
            available = [numeric_or_none(value.get(metric)) for value in values]
            materialized = [value for value in available if value is not None]
            row[metric] = mean(materialized) if materialized else None
        parent_rows.append(row)
    write_csv_exclusive(output / "parent_mean_metrics.csv", parent_rows)

    tunnel_rows: list[dict[str, Any]] = []
    for tunnel in ("A", "B", "C", "D"):
        for condition in CONDITIONS:
            subset = [
                row
                for row in parent_rows
                if row["tunnel"] == tunnel and row["condition"] == condition
            ]
            out: dict[str, Any] = {
                "tunnel": tunnel,
                "condition": condition,
                "parent_count": len(subset),
            }
            for metric in metric_names:
                values = [row[metric] for row in subset if row[metric] is not None]
                out[metric] = mean(values) if values else None
            tunnel_rows.append(out)
    write_csv_exclusive(output / "tunnel_macro_metrics.csv", tunnel_rows)

    equal_tunnel: dict[str, dict[str, float | None]] = {}
    for condition in CONDITIONS:
        subset = [row for row in tunnel_rows if row["condition"] == condition]
        equal_tunnel[condition] = {}
        for metric in metric_names:
            values = [row[metric] for row in subset if row[metric] is not None]
            equal_tunnel[condition][metric] = mean(values) if values else None

    bootstrap_resamples = 10000
    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    for name, left, right in (
        ("reference_minus_degraded", "reference", "degraded"),
        ("task_on_minus_degraded", "task_on", "degraded"),
        ("task_off_minus_degraded", "task_off", "degraded"),
        ("task_on_minus_task_off", "task_on", "task_off"),
    ):
        comparisons[name] = {}
        for metric in ("pixel_auprc", "f1_at_0_5"):
            estimate = stratified_parent_bootstrap(
                parent_rows,
                metric,
                left,
                right,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + len(comparisons) * 10 + (0 if metric == "pixel_auprc" else 1),
            )
            positive_tunnels = 0
            per_tunnel: dict[str, float] = {}
            for tunnel in ("A", "B", "C", "D"):
                left_row = next(
                    row
                    for row in tunnel_rows
                    if row["tunnel"] == tunnel and row["condition"] == left
                )
                right_row = next(
                    row
                    for row in tunnel_rows
                    if row["tunnel"] == tunnel and row["condition"] == right
                )
                delta = float(left_row[metric]) - float(right_row[metric])
                per_tunnel[tunnel] = delta
                positive_tunnels += int(delta > 0.0)
            comparisons[name][metric] = {
                **estimate,
                "positive_tunnels": positive_tunnels,
                "per_tunnel": per_tunnel,
            }

    manipulation = comparisons["reference_minus_degraded"]["pixel_auprc"]
    primary = comparisons["task_on_minus_degraded"]["pixel_auprc"]
    task_increment = comparisons["task_on_minus_task_off"]["pixel_auprc"]
    gates = {
        "degradation_manipulation": manipulation["positive_tunnels"] >= 3,
        "task_on_primary": (
            primary["point"] > 0.0
            and primary["ci_lower"] > 0.0
            and primary["positive_tunnels"] >= 3
        ),
        "task_loss_increment": (
            task_increment["point"] > 0.0
            and task_increment["ci_lower"] > 0.0
            and task_increment["positive_tunnels"] >= 3
        ),
    }
    supported = gates["degradation_manipulation"] and gates["task_on_primary"]
    final = {
        "schema_version": SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "post_hoc": True,
        "cohort": {
            "tiles": int(protocol["data"]["selection"]["expected_tiles"]),
            "parents": int(protocol["data"]["selection"]["expected_parents"]),
            "tunnels": int(protocol["data"]["selection"]["expected_tunnels"]),
            "chains": len(protocol["detectors"]["seed_pairing"]),
        },
        "equal_tunnel_macro": equal_tunnel,
        "comparisons": comparisons,
        "gates": gates,
        "verdict": "SUPPORTED_WITH_LIMITS" if supported else "NOT_SUPPORTED",
        "task_loss_verdict": (
            "SUPPORTED_WITH_LIMITS" if gates["task_loss_increment"] else "NOT_SUPPORTED"
        ),
        "claim_boundary": protocol["claim_boundary"],
        "artifacts": {
            "parent_chain_metrics_sha256": sha256_file(output / "parent_chain_metrics.csv"),
            "parent_mean_metrics_sha256": sha256_file(output / "parent_mean_metrics.csv"),
            "tunnel_macro_metrics_sha256": sha256_file(output / "tunnel_macro_metrics.csv"),
            "freeze_sha256": sha256_file(output / "freeze.json"),
        },
        "completed_utc": utc_now(),
    }
    write_json_exclusive(final_path, final)
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.set_defaults(handler=command_audit)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.set_defaults(handler=command_smoke)
    run = commands.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.set_defaults(handler=command_run)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.set_defaults(handler=command_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.protocol = args.protocol.expanduser().resolve()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
