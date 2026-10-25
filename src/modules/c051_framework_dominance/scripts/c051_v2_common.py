"""Frozen-path and input-scale contract helpers for the C051 v2 evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

import c051_common as common

PROTOCOL_V2 = (
    common.C051 / "protocol" / "framework_dominance_evaluation_addendum_v2.json"
)
EVALUATION_V2 = common.RUNS / "evaluation_v2"
SUMMARY_V2 = common.RUNS / "factorial_summary_v2.json"
ROWS_V2 = common.RUNS / "factorial_rows_v2.csv"
AUDIT_V2 = common.RUNS / "independent_audit_v2.json"


def to_detector_u8(restored_01: np.ndarray) -> np.ndarray:
    """Convert a finite [0,1] restorer output to the detector's uint8 contract."""
    array = np.asarray(restored_01, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise common.C051Error("Non-finite restoration output")
    return np.rint(np.clip(array, 0.0, 1.0) * np.float32(255.0)).astype(
        np.uint8
    )


def validate_protocol_v2() -> dict[str, Any]:
    """Fail closed unless the v1 protocol and all frozen v2 sources still match."""
    protocol_v1 = common.validate_protocol()
    if not PROTOCOL_V2.is_file():
        raise common.C051Error(f"Missing C051 v2 protocol: {PROTOCOL_V2}")
    payload = common.load_json(PROTOCOL_V2)
    if payload.get("status") != "FROZEN_BEFORE_CORRECTED_EVALUATION_V2":
        raise common.C051Error("C051 v2 protocol is not frozen")
    if common.payload_sha256(payload) != payload.get("payload_sha256"):
        raise common.C051Error("C051 v2 protocol payload mismatch")
    if payload.get("supersedes_protocol_payload_sha256") != protocol_v1.get(
        "payload_sha256"
    ):
        raise common.C051Error("C051 v2 does not reference the frozen v1 protocol")
    for name, expected in payload.get("code_sha256", {}).items():
        path = common.HERE / name
        if not path.is_file() or common.sha256_file(path) != expected:
            raise common.C051Error(f"C051 v2 code hash mismatch: {name}")
    return payload
