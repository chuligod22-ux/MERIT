"""Shared contracts for the C039 independent C036 confirmation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C039 = HERE.parent
REVISION = C039.parent
C038 = REVISION / "c038_mprnet_narrow_margin_repair"
C036 = REVISION / "c036_reserved_iso200_confirmation_bank"
C035 = REVISION / "c035_mprnet_adaptive_route_gate"
C032 = REVISION / "c032_mprnet_hybrid_frequency_route"
FREEZE = C039 / "protocol" / "confirmation_freeze_v1.json"
OUTPUT_ROOT = C039 / "runs" / "confirmation_v1"


class C039Error(RuntimeError):
    """Raised when the frozen C039 confirmation contract is violated."""


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise C039Error(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c038_common = import_path("c039_c038_common", C038 / "scripts" / "c038_common.py")
c023 = c038_common.c023_common
c022_common = c038_common.c037_common.c022_common


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_freeze() -> dict[str, Any]:
    value = json.loads(FREEZE.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    expected = hashlib.sha256(c023.canonical_bytes(payload)).hexdigest()
    if recorded != expected:
        raise C039Error("C039 confirmation freeze payload mismatch")
    if value.get("status") != "FROZEN_BEFORE_ANY_C036_MODEL_OUTPUT":
        raise C039Error("C039 confirmation is not frozen")
    for item in value["source_assets"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise C039Error(f"C039 confirmation source mismatch: {path}")
    return value
