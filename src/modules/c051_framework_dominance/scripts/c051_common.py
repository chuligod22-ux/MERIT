"""Shared paths, hashing, and model construction for C051."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
C051 = HERE.parent
MODIFIED = C051.parent
REVISION = MODIFIED.parent
WORKSPACE = REVISION.parent
PROTOCOL = C051 / "protocol" / "framework_dominance_protocol_v1.json"
ASSETS = C051 / "assets"
RUNS = C051 / "runs"
TRAINING = RUNS / "training_v1"
EVALUATION = RUNS / "evaluation_v1"

TUNNELSCANNING = Path(r"<LOCAL_PATH_REDACTED>")
RESTORATION = TUNNELSCANNING / "03_restoration"
NRIQA = TUNNELSCANNING / "02_nriqa"
LEGACY_DATA = RESTORATION / "04_data"

C007_EXTERNAL = MODIFIED / "c007_d2_external_synthetic"
C007_MULTI = MODIFIED / "c007_d2_multidetector"
C017 = MODIFIED / "c017_transformer_mamba_validation"

BACKBONES = ("nafnet", "plainunet", "restormer", "mamba")
ARMS = ("off", "on")
SEEDS = (0, 1, 2)
WIDTHS = {"nafnet": 32, "plainunet": 48, "restormer": 48, "mamba": 128}


class C051Error(RuntimeError):
    """Raised when a C051 invariant is violated."""


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    clean = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return json.dumps(
        clean,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol() -> dict[str, Any]:
    protocol = load_json(PROTOCOL)
    if protocol.get("status") != "FROZEN_BEFORE_C051_TRAINING_OR_EVALUATION":
        raise C051Error("C051 protocol is not frozen")
    if payload_sha256(protocol) != protocol.get("payload_sha256"):
        raise C051Error("C051 protocol payload mismatch")
    return protocol


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(backbone: str, in_ch: int) -> torch.nn.Module:
    if backbone not in BACKBONES:
        raise C051Error(f"Unknown backbone: {backbone}")
    from c051_backbones import MambaLite, PlainUNet, RestormerLite
    from c051_nafnet import NAFNetUNet

    width = WIDTHS[backbone]
    if backbone == "nafnet":
        return NAFNetUNet(in_ch=in_ch, width=width)
    if backbone == "plainunet":
        return PlainUNet(in_ch=in_ch, width=width)
    if backbone == "restormer":
        return RestormerLite(in_ch=in_ch, width=width)
    return MambaLite(in_ch=in_ch, width=width)


def paired_initial_models(
    backbone: str, seed: int
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Construct OFF/ON models with identical shared parameters.

    The only shape mismatch is the first image-input convolution. Its degraded
    channel and bias are copied from OFF; prompt-channel weights start at zero.
    """

    init_seed = 202610510 + int(seed)
    torch.manual_seed(init_seed)
    off = build_model(backbone, 1)
    torch.manual_seed(init_seed)
    on = build_model(backbone, 3)
    off_state = off.state_dict()
    on_state = on.state_dict()
    unmatched: list[str] = []
    with torch.no_grad():
        for name, target in on_state.items():
            source = off_state.get(name)
            if source is None:
                unmatched.append(name)
                continue
            if source.shape == target.shape:
                target.copy_(source)
            elif (
                source.ndim == 4
                and target.ndim == 4
                and source.shape[0] == target.shape[0]
                and source.shape[1] == 1
                and target.shape[1] == 3
                and source.shape[2:] == target.shape[2:]
            ):
                target.zero_()
                target[:, :1].copy_(source)
            else:
                unmatched.append(name)
    if unmatched:
        raise C051Error(f"Unpaired initialization parameters: {unmatched}")
    on.load_state_dict(on_state, strict=True)
    return off, on


def model_id(backbone: str, arm: str, seed: int) -> str:
    return f"{backbone}__{arm}__seed{seed}"


def run_dir(backbone: str, arm: str, seed: int) -> Path:
    return TRAINING / model_id(backbone, arm, seed)
