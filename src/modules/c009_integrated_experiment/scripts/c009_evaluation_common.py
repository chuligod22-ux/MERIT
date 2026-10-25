"""Shared fail-closed helpers for the frozen C009 evaluation queues."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
DEFAULT_CONTRACT = C009 / "protocol" / "full_evaluation_contract_v1.json"
EXPECTED_CONTRACT_SHA256 = "c08e955ad9d52a3a2fc63336c95683bf472fefb5ab26c0eea20b88652cc313f8"


class EvaluationError(RuntimeError):
    """Raised when a frozen evaluation invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_asset(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise EvaluationError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != str(record["sha256"]).lower():
        raise EvaluationError(
            f"{label} hash mismatch: expected={record['sha256']} actual={actual}"
        )
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise EvaluationError(f"{label} byte-size mismatch: {path}")
    return path


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    path = path.resolve()
    actual = sha256_file(path)
    if actual != EXPECTED_CONTRACT_SHA256:
        raise EvaluationError(
            f"Evaluation contract changed: expected={EXPECTED_CONTRACT_SHA256} actual={actual}"
        )
    contract = load_json(path)
    if contract.get("status") != "FROZEN_BEFORE_ANY_C009_FIXED_FINAL_OR_ENDPOINT_RESULT":
        raise EvaluationError("Evaluation contract is not frozen")
    payload = dict(contract)
    recorded = payload.pop("design_payload_sha256")
    actual_payload = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if recorded != actual_payload:
        raise EvaluationError("Evaluation contract payload hash mismatch")
    return contract


def verify_completion_marker(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    marker_path = run_dir / "RUN_COMPLETE.json"
    checkpoint_path = run_dir / "final.pt"
    if not marker_path.is_file() or not checkpoint_path.is_file():
        raise EvaluationError(f"Incomplete fixed-final run: {run_dir}")
    marker = load_json(marker_path)
    payload = dict(marker)
    recorded = payload.pop("payload_sha256", None)
    if recorded != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise EvaluationError(f"Completion marker payload mismatch: {marker_path}")
    if marker.get("status") != "PASS" or int(marker.get("global_step", -1)) != 8000:
        raise EvaluationError(f"Run is not a passing 8000-step fixed-final: {run_dir}")
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if marker.get("final_checkpoint_sha256") != actual_checkpoint_hash:
        raise EvaluationError(f"Checkpoint hash mismatch: {checkpoint_path}")
    return checkpoint_path, marker


def model_id(specification: dict[str, Any]) -> str:
    if specification["kind"] == "wp1":
        return (
            f"wp1__{specification['family']}__{specification['arm']}"
            f"__seed{specification['seed']}"
        )
    return (
        f"{specification['wp']}__{specification['structure']}"
        f"__seed{specification['seed']}"
    )


def wp1_specifications(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"kind": "wp1", **record}
        for record in contract["model_matrix"]["wp1"]
    ]


def wp23_specifications(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"kind": "followon", **record}
        for record in contract["model_matrix"]["wp23_additional"]
    ]


def run_directory(specification: dict[str, Any]) -> Path:
    if specification["kind"] == "wp1":
        return (
            C009
            / "runs"
            / "wp1"
            / specification["family"]
            / f"{specification['arm']}_seed{specification['seed']}"
        )
    return (
        C009
        / "runs"
        / specification["wp"]
        / f"{specification['structure']}_seed{specification['seed']}"
    )


def build_and_load_model(
    specification: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path, marker = verify_completion_marker(run_directory(specification))
    if specification["kind"] == "wp1":
        factory = import_module(
            f"g11_factory_eval_{specification['family']}",
            HERE / "g11_official_model_factory.py",
        )
        model, metadata = factory.build_official_adapter(specification["family"])
    elif specification["wp"] == "wp2":
        factory = import_module(
            "g11_factory_eval_wp2", HERE / "g11_official_model_factory.py"
        )
        model, metadata = factory.build_official_adapter("nafnet")
    else:
        structures = import_module("wp23_models_eval", HERE / "c009_wp23_models.py")
        model, metadata = structures.build_structure(specification["structure"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "model_id": model_id(specification),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "completion_payload_sha256": marker["payload_sha256"],
        "model": metadata,
    }


def require_all_training_complete(contract: dict[str, Any]) -> None:
    wp1_status = load_json(C009 / "runs" / "queue" / "full_queue_status.json")
    wp23_status = load_json(C009 / "runs" / "queue" / "wp23_queue_status.json")
    if wp1_status.get("status") != "complete" or int(wp1_status.get("completed_runs", -1)) != 18:
        raise EvaluationError("WP1 is not complete")
    if wp23_status.get("status") != "complete" or len(wp23_status.get("completed_full_runs", [])) != 9:
        raise EvaluationError("WP2/WP3 is not complete")
    for specification in wp1_specifications(contract) + wp23_specifications(contract):
        verify_completion_marker(run_directory(specification))


def prompt_tensor(
    degraded: np.ndarray,
    blur: Iterable[float] | float,
    noise: Iterable[float] | float,
) -> np.ndarray:
    if degraded.ndim != 3:
        raise EvaluationError(f"Expected degraded [B,H,W], got {degraded.shape}")
    batch = degraded.shape[0]
    blur_values = np.broadcast_to(np.asarray(blur, dtype=np.float32), (batch,))
    noise_values = np.broadcast_to(np.asarray(noise, dtype=np.float32), (batch,))
    blur_maps = np.empty_like(degraded, dtype=np.float32)
    noise_maps = np.empty_like(degraded, dtype=np.float32)
    for index in range(batch):
        blur_maps[index].fill(float(np.clip(blur_values[index], 0.0, 1.0)))
        noise_maps[index].fill(float(np.clip(noise_values[index], 0.0, 1.0)))
    return np.stack([degraded, blur_maps, noise_maps], axis=1).astype(np.float32)


def autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )
