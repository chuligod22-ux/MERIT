"""Shared paths and immutable checks for C049 independent confirmation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C049 = HERE.parent
REVISION = C049.parent
C040 = REVISION / "c040_reserved_iso400_restormer_confirmation"
C048 = REVISION / "c048_restormer_psnr_integrated_development"
PROTOCOL = C049 / "protocol" / "confirmation_protocol_v1.json"
OUTPUT = C049 / "runs" / "confirmation_v1"

for directory in (
    C048 / "scripts",
    REVISION / "c041_restormer_cross_condition_development" / "scripts",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import c048_common  # noqa: E402
import c048_evaluate  # noqa: E402


class C049Error(RuntimeError):
    """Raised when a frozen C049 invariant is violated."""


def canonical_bytes(value: Any) -> bytes:
    return c048_common.canonical_bytes(value)


def sha256_file(path: Path) -> str:
    return c048_common.sha256_file(path)


def write_json(path: Path, value: Any) -> None:
    c048_common.write_json(path, value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    c048_common.write_jsonl(path, rows)


def load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    expected = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if recorded != expected:
        raise C049Error("C049 confirmation protocol payload mismatch")
    if value.get("status") != "FROZEN_BEFORE_ANY_C040_MODEL_INFERENCE":
        raise C049Error("C049 protocol is not frozen")
    for item in value["source_assets"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise C049Error(f"Frozen source mismatch: {path}")
    return value
