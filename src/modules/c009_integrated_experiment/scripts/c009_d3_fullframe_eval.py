"""Stream one C009 WP1 fixed-final checkpoint over all 974 eligible D3 frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
REVISION = C009.parent.parent
C006 = REVISION / "40_수정본" / "c006_execution"
DEFAULT_OUTPUT = C009 / "evaluation" / "wp6_d3_fullframe"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def load_c006_modules():
    root = str(C006.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    operations = common.import_module(
        "c006_ops_c009_d3", C006 / "c006_operational_inference.py"
    )
    stream = common.import_module(
        "c006_stream_c009_d3", C006 / "c006_d3_streaming.py"
    )
    return operations, stream


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen_inputs(
    contract: dict[str, Any], stream: Any
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    protocol_path = common.verify_asset(contract["assets"]["d3_protocol"], "D3 protocol")
    protocol = stream.load_protocol(protocol_path)
    for name in ("cohort", "consensus", "endpoint_code"):
        record = protocol["assets"][name]
        path = Path(record["path"])
        if common.sha256_file(path) != record["sha256"]:
            raise common.EvaluationError(f"D3 {name} hash mismatch")
    detector = protocol["detector"]
    for name in ("code", "protocol", "checkpoint", "ledger", "imagenet_weight"):
        record = detector[name]
        if common.sha256_file(Path(record["path"])) != record["sha256"]:
            raise common.EvaluationError(f"D3 detector {name} hash mismatch")
    _, eligible, consensus = stream.load_inputs(protocol)
    if len(eligible) != int(contract["wp6_different_tunnel_same_system"]["eligible_rows"]):
        raise common.EvaluationError("D3 eligible count differs from evaluation contract")
    return protocol, eligible, consensus


def load_detector(protocol: dict[str, Any], operations: Any, device: torch.device):
    detector = protocol["detector"]
    detector_class = operations._load_d2_class(Path(detector["code"]["path"]))
    model = detector_class()
    checkpoint = torch.load(
        Path(detector["checkpoint"]["path"]), map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def quality_descriptors(gray: np.ndarray) -> dict[str, float]:
    image = gray.astype(np.float32) / np.float32(255.0)
    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    highpass = image - cv2.GaussianBlur(image, (0, 0), 1.0)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "brightness_mean": float(np.mean(image)),
        "brightness_std": float(np.std(image)),
        "noise_proxy_mad": float(np.median(np.abs(highpass - np.median(highpass))) * 1.4826),
        "saturation_fraction": float(np.mean((gray == 0) | (gray == 255))),
        "local_contrast_laplacian_std": float(np.std(laplacian)),
        "gradient_mean": float(np.mean(np.hypot(gx, gy))),
    }


def progress_path(output_dir: Path, index: int, cohort_id: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in cohort_id)
    return output_dir / "progress" / f"{index:04d}_{safe}.json"


def validate_progress(path: Path, row: dict[str, str], model_name: str) -> None:
    value = common.load_json(path)
    if value.get("cohort_id") != row["cohort_id"] or value.get("model_id") != model_name:
        raise common.EvaluationError(f"D3 progress identity mismatch: {path}")
    if value.get("source_sha256") != row["sha256"]:
        raise common.EvaluationError(f"D3 progress source mismatch: {path}")
    if value.get("status") != "PASS":
        raise common.EvaluationError(f"D3 progress is incomplete: {path}")


def run(specification: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if specification["kind"] != "wp1":
        raise common.EvaluationError("D3 full-frame matrix is restricted to 18 WP1 models")
    contract = common.load_contract()
    operations, stream = load_c006_modules()
    protocol, eligible, consensus = load_frozen_inputs(contract, stream)
    model_name = common.model_id(specification)
    output_dir = output_root / model_name
    marker_path = output_dir / "RUN_COMPLETE.json"
    if marker_path.exists():
        marker = common.load_json(marker_path)
        marker["status"] = "ALREADY_COMPLETE"
        return marker
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for index, row in enumerate(eligible, start=1):
        path = progress_path(output_dir, index, row["cohort_id"])
        if path.exists():
            validate_progress(path, row, model_name)
            completed += 1
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise common.EvaluationError("CUDA is required")
    restorer, provenance = common.build_and_load_model(specification, device)
    detector = load_detector(protocol, operations, device)
    endpoint = stream.load_module(
        Path(protocol["assets"]["endpoint_code"]["path"]), "c009_d3_frozen_endpoint"
    )
    raw_root = Path(protocol["raw_root"])
    tiling = contract["wp6_different_tunnel_same_system"]["tiling"]
    batch_size = int(tiling["batch_size"][specification["family"]])
    started = time.perf_counter()
    session_completed = 0
    for index, row in enumerate(eligible, start=1):
        path = progress_path(output_dir, index, row["cohort_id"])
        if path.exists():
            continue
        frame_started = time.perf_counter()
        source = raw_root / Path(row["relative_path"])
        if common.sha256_file(source) != row["sha256"]:
            raise common.EvaluationError(f"D3 source hash mismatch: {source}")
        image = operations.read_bgr(source)
        if list(image.shape) != list(protocol["input_shape"]):
            raise common.EvaluationError(f"D3 frame shape mismatch: {source}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        torch.cuda.reset_peak_memory_stats(device)
        restored = operations.tiled_restore_gray(
            gray,
            restorer,
            device,
            tile_size=int(tiling["tile_size"]),
            overlap=int(tiling["overlap"]),
            batch_size=batch_size,
            blend_floor=0.05,
            amp_dtype="bfloat16",
        )
        encoded = np.rint(restored * np.float32(255.0)).astype(np.uint8)
        restored_bgr = cv2.cvtColor(encoded, cv2.COLOR_GRAY2BGR)
        detector_started = time.perf_counter()
        probability = stream.detector_probability(restored_bgr, detector, device, protocol)
        event = stream.endpoint_event(
            probability,
            consensus[row["cohort_id"]],
            row,
            model_name,
            str(specification["seed"]),
            endpoint,
            protocol,
            time.perf_counter() - detector_started,
            array_sha256(restored_bgr),
        )
        payload = {
            "schema_version": "MERIT-C009-D3-FULLFRAME-PROGRESS-v1",
            "status": "PASS",
            "model_id": model_name,
            "cohort_id": row["cohort_id"],
            "eligible_index": index,
            "source_relative_path": row["relative_path"],
            "source_sha256": row["sha256"],
            "consensus_label": consensus[row["cohort_id"]]["label"],
            "cell": {
                key: row.get(key)
                for key in ("camera", "pass_id", "region", "side", "iso")
                if key in row
            },
            "raw_quality": quality_descriptors(gray),
            "restored_quality": quality_descriptors(encoded),
            "endpoint": event,
            "frame_elapsed_seconds": time.perf_counter() - frame_started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "committed_utc": utc_now(),
        }
        payload["payload_sha256"] = hashlib.sha256(common.canonical_bytes(payload)).hexdigest()
        common.write_json_exclusive(path, payload)
        completed += 1
        session_completed += 1
        elapsed = time.perf_counter() - started
        remaining = len(eligible) - completed
        status = {
            "schema_version": "MERIT-C009-D3-FULLFRAME-STATUS-v1",
            "status": "running" if remaining else "inference_complete",
            "model_id": model_name,
            "completed": completed,
            "eligible": len(eligible),
            "remaining": remaining,
            "last_cohort_id": row["cohort_id"],
            "session_average_seconds": elapsed / session_completed,
            "session_projected_remaining_seconds": elapsed / session_completed * remaining,
            "updated_utc": utc_now(),
        }
        common.write_json_atomic(output_dir / "status.json", status)
        print(json.dumps(status, sort_keys=True), flush=True)
        del restored, encoded, restored_bgr, probability
    progress_files = sorted((output_dir / "progress").glob("*.json"))
    if len(progress_files) != len(eligible):
        raise common.EvaluationError("D3 final progress coverage mismatch")
    manifest_hash = hashlib.sha256()
    for path in progress_files:
        manifest_hash.update(path.name.encode("utf-8"))
        manifest_hash.update(common.sha256_file(path).encode("ascii"))
    legacy_raw = C006 / "artifacts" / "d3_stream_v1" / "d3_endpoint_streaming_final.json"
    marker: dict[str, Any] = {
        "schema_version": "MERIT-C009-D3-FULLFRAME-RUN-COMPLETE-v1",
        "status": "PASS",
        "model_id": model_name,
        "specification": specification,
        "provenance": provenance,
        "contract_sha256": common.EXPECTED_CONTRACT_SHA256,
        "eligible_frames": len(eligible),
        "progress_files": len(progress_files),
        "progress_manifest_sha256": manifest_hash.hexdigest(),
        "legacy_raw_baseline_path": str(legacy_raw.resolve()),
        "legacy_raw_baseline_sha256": common.sha256_file(legacy_raw),
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
    specification = find_specification(args.model_id, contract)
    result = run(specification, args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
