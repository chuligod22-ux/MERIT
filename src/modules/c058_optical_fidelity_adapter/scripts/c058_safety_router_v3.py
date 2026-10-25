"""C058 v3g full-RAW public validation and one-shot holdout runner.

The v3 route was designed after the v2 existing-condition confirmation.  It
therefore makes no universal claim: the MERIT residual route is active only at
80 km/h, ISO 100/200, and 2.5/3.5/4.5 m.  Every other stratum is a model-free,
byte-exact RAW bypass.  Before any unexposed PNG is decoded, this module
freezes a parent-disjoint, name-only 280-frame inventory, the public
native-to-canonical end-to-end development, and the complete execution stack.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
C057 = BASE.parent / "c057_mtf50_estimator_v2"
RAW_ROOT = Path("<LOCAL_PATH_REDACTED>")
EXPOSED_COHORT = (
    BASE.parent
    / "c007_d0_condition_eval"
    / "artifacts"
    / "cam1_v1"
    / "d0_cam1_cohort.csv"
)
LOCATOR_CODE = BASE / "scripts" / "c058_raw_locator.py"
LOCATOR_FREEZE = (
    BASE / "artifacts" / "locator_development_v1" / "locator_execution_freeze_v2.json"
)
ADAPTER_CODE = BASE / "scripts" / "c058_adapter.py"
TEST_CODE = BASE / "tests" / "test_c058_safety_router_v3.py"
V3B_INVENTORY = BASE / "protocol" / "holdout_inventory_v3b.json"
INVENTORY = BASE / "protocol" / "holdout_inventory_v3g.json"
CONTRACT = BASE / "protocol" / "safety_router_development_contract_v3g.json"
SELECTION = BASE / "protocol" / "selected_safety_router_v3g.json"
EXECUTION_FREEZE = BASE / "protocol" / "holdout_execution_freeze_v3g.json"
PUBLIC_RAW_MANIFEST = BASE / "protocol" / "public_full_raw_manifest_v3e.json"
V3E_PUBLIC_FINAL = (
    BASE
    / "artifacts"
    / "safety_router_v3e"
    / "public_e2e_development"
    / "public_e2e_development_final.json"
)
V3F_DIAGNOSTIC = (
    BASE
    / "artifacts"
    / "safety_router_v3f"
    / "v3f_diagnostic_reconstruction.json"
)
RETROSPECTIVE_OUTPUT = BASE / "artifacts" / "safety_router_v3b" / "retrospective_profile_v3b.json"
DEVELOPMENT_OUTPUT = BASE / "artifacts" / "safety_router_v3g" / "public_e2e_development"
HOLDOUT_OUTPUT = BASE / "artifacts" / "safety_router_v3g" / "holdout_280_one_shot"

ROUTE_ID = "conditioned_expert_band_inverse_s0.75_cap8"
ACTIVE_SPEED = 80
ACTIVE_ISOS = frozenset({100, 200})
ACTIVE_DISTANCES = frozenset({2.5, 3.5, 4.5})
REFERENCE_SPEED = 60
REFERENCE_ISO = 100
HOLDOUT_FRAMES = 280
HOLDOUT_CONDITIONS = 29
HOLDOUT_ACTIVE_FRAMES = 47
HOLDOUT_ACTIVE_CONDITIONS = 6
HOLDOUT_ACTIVE_DISTANCES = 3
HOLDOUT_INACTIVE_CONDITIONS = 23
PREBANK_PARENT_DISJOINT_CANDIDATES = 397
BANK_PARENT_EXCLUSIONS = 117
PUBLIC_FRAMES = 137
PUBLIC_REFERENCE_FRAMES = 52
PUBLIC_ACTIVE_FRAMES = 51
PUBLIC_INACTIVE_SAFETY_FRAMES = 34
PUBLIC_CONDITIONS = 15
PUBLIC_REFERENCE_CONDITIONS = 5
PUBLIC_ACTIVE_CONDITIONS = 6
PUBLIC_INACTIVE_SAFETY_CONDITIONS = 4
EVIDENCE_LABEL = (
    "strict parent-disjoint endpoint holdout with 6/10 active conditions "
    "and 3/5 active distances"
)
SEEDS = (0, 1, 2)
EXPERT_CONFIGS: dict[tuple[float, int], tuple[int, float]] = {
    (2.5, 100): (1, -0.20),
    (2.5, 200): (1, -1.00),
    (3.5, 100): (1, -1.50),
    (3.5, 200): (1, -1.50),
    (4.5, 100): (0, -0.90),
    (4.5, 200): (1, -1.50),
}
CONTEXT_RADIUS_MULTIPLIER = 1.05
CANONICAL_RADIUS = 320.0
BOOTSTRAP_COUNT = 10_000
BOOTSTRAP_SEED = 20261058
INVENTORY_SCHEMA = "MERIT-C058-SAFETY-ROUTER-HOLDOUT-INVENTORY-v3g"
PUBLIC_RAW_MANIFEST_SCHEMA = "MERIT-C058-PUBLIC-FULL-RAW-MANIFEST-v3e"
CONTRACT_SCHEMA = "MERIT-C058-SAFETY-ROUTER-DEVELOPMENT-v3g"
PROGRESS_SCHEMA = "MERIT-C058-SAFETY-ROUTER-PROGRESS-v3g"
SELECTION_SCHEMA = "MERIT-C058-SAFETY-ROUTER-SELECTION-v3g"
EXECUTION_SCHEMA = "MERIT-C058-SAFETY-ROUTER-HOLDOUT-EXECUTION-v3g"
PUBLIC_FINAL_SCHEMA = "MERIT-C058-SAFETY-ROUTER-PUBLIC-E2E-v3g"
PUBLIC_STUDY = "public_native_e2e_development_v3g"
HOLDOUT_STUDY = "prospective_holdout_280_v3g"

BANK_MANIFESTS = (
    BASE.parent
    / "c030_reserved_confirmation_bank"
    / "artifacts"
    / "confirmatory_v1"
    / "d0_bank_manifest.csv",
    BASE.parent
    / "c033_reserved_confirmation_bank"
    / "artifacts"
    / "confirmatory_v1"
    / "d0_bank_manifest.csv",
    BASE.parent
    / "c036_reserved_iso200_confirmation_bank"
    / "artifacts"
    / "confirmatory_v1"
    / "d0_bank_manifest.csv",
    BASE.parent
    / "c040_reserved_iso400_restormer_confirmation"
    / "artifacts"
    / "confirmatory_v1"
    / "d0_bank_manifest.csv",
)


class ProtocolError(RuntimeError):
    """A frozen input, one-shot rule, or predeclared gate was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def write_json_atomic(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = sidecar_path(path)
    if sidecar.exists():
        raise FileExistsError(sidecar)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def require_sidecar(path: Path) -> str:
    sidecar = sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise ProtocolError(f"missing frozen file or sidecar: {path}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    actual = sha256_file(path)
    if len(fields) != 2 or fields[0] != actual or fields[1] != path.name:
        raise ProtocolError(f"SHA-256 sidecar mismatch: {path}")
    return actual


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def binding(path: Path, *, sidecar: bool = False) -> dict[str, str]:
    digest = require_sidecar(path) if sidecar else sha256_file(path)
    return {"path": relative_or_absolute(path), "sha256": digest}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def adapter_module():
    return load_module("c058_v3_adapter", ADAPTER_CODE)


def locator_module():
    return load_module("c058_v3_locator", LOCATOR_CODE)


def c057_module():
    return load_module("c058_v3_c057", C057 / "run_restoration_mtf50_v2.py")


def parse_condition(condition: str) -> tuple[int, float, int]:
    match = re.fullmatch(
        r"(60|80)km_(2\.5|3\.5|4\.5|5\.5|6\.5)m_ISO(100|200|400|800|1600)",
        condition,
    )
    if match is None:
        raise ProtocolError(f"unexpected condition: {condition}")
    return int(match.group(1)), float(match.group(2)), int(match.group(3))


def all_nonreference_conditions() -> set[str]:
    return {
        f"{speed}km_{distance:.1f}m_ISO{iso}"
        for speed in (60, 80)
        for distance in (2.5, 3.5, 4.5, 5.5, 6.5)
        for iso in (100, 200, 400, 800, 1600)
        if not (speed == REFERENCE_SPEED and iso == REFERENCE_ISO)
    }


def router_active(speed_kmh: int, iso: int, distance_m: float) -> bool:
    return (
        int(speed_kmh) == ACTIVE_SPEED
        and int(iso) in ACTIVE_ISOS
        and float(distance_m) in ACTIVE_DISTANCES
    )


def expert_configuration(distance_m: float, iso: int) -> tuple[int, float]:
    key = (float(distance_m), int(iso))
    if key not in EXPERT_CONFIGS:
        raise ProtocolError(f"no frozen active expert for distance/ISO: {key}")
    return EXPERT_CONFIGS[key]


def expert_table_payload() -> list[dict[str, int | float]]:
    return [
        {
            "distance_m": distance,
            "iso": iso,
            "expert_seed": seed,
            "alpha": alpha,
            "sigma": 0.75,
            "cap_dn": 8.0,
        }
        for (distance, iso), (seed, alpha) in sorted(EXPERT_CONFIGS.items())
    ]


def active_run_key(distance_m: float, iso: int) -> str:
    seed, _ = expert_configuration(distance_m, iso)
    return f"{ROUTE_ID}_{seed}"


def fixed_candidate(distance_m: float, iso: int):
    adapter = adapter_module()
    seed, alpha = expert_configuration(distance_m, iso)
    candidate = adapter.AdapterCandidate(ROUTE_ID, alpha, 0.75, 0.0, 8.0)
    return adapter, candidate, seed


def identity_diagnostics(*, route_id: str, seed: str) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "seed": seed,
        "exact_identity": True,
        "reference_bypass": True,
        "residual_cap_fraction": 0.0,
        "uncapped_residual_mean_abs_dn": 0.0,
        "capped_residual_mean_abs_dn": 0.0,
        "preclip_overshoot_fraction": 0.0,
        "preclip_below_zero_fraction": 0.0,
        "preclip_above_one_fraction": 0.0,
        "introduced_clipping_fraction": 0.0,
        "introduced_low_clipping_fraction": 0.0,
        "introduced_high_clipping_fraction": 0.0,
        "mean_abs_change_dn": 0.0,
        "max_abs_change_dn": 0.0,
    }


def route_native_crop(
    raw_crop: np.ndarray,
    speed_kmh: int,
    iso: int,
    distance_m: float,
    model_callback: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Route a native-coordinate crop; inactive strata never call a model."""

    if raw_crop.ndim != 2 or raw_crop.dtype != np.uint8:
        raise ValueError("raw native crop must be uint8 grayscale")
    outputs: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "raw_none": (raw_crop.copy(), identity_diagnostics(route_id="raw", seed="none"))
    }
    if not router_active(speed_kmh, iso, distance_m):
        output = raw_crop.copy()
        if not np.array_equal(output, raw_crop):
            raise ProtocolError("inactive RAW bypass is not byte-exact")
        outputs["raw_exact_bypass"] = (
            output,
            identity_diagnostics(route_id="raw_exact_bypass", seed="none"),
        )
        return outputs

    adapter, candidate, seed = fixed_candidate(distance_m, iso)
    task_on = model_callback(raw_crop, seed)
    output, diagnostics = adapter.apply_adapter(raw_crop, task_on, candidate)
    record = diagnostics.to_dict()
    record.update(
        {
            "route_id": ROUTE_ID,
            "seed": str(seed),
            "conditioned_alpha": float(candidate.alpha),
        }
    )
    outputs[active_run_key(distance_m, iso)] = (output, record)
    return outputs


def native_context_bounds(shape: tuple[int, int], located: Any) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    half = CONTEXT_RADIUS_MULTIPLIER * float(located.radius)
    x0 = max(0, int(round(float(located.center_x) - half)))
    y0 = max(0, int(round(float(located.center_y) - half)))
    x1 = min(width, int(round(float(located.center_x) + half)))
    y1 = min(height, int(round(float(located.center_y) + half)))
    if x1 <= x0 or y1 <= y0 or min(x1 - x0, y1 - y0) < 64:
        raise ProtocolError("invalid locator-derived native context")
    return x0, y0, x1, y1


def local_locator(located: Any, x0: int, y0: int):
    return located._replace(
        center_x=float(located.center_x) - x0,
        center_y=float(located.center_y) - y0,
    )


def canonical_to_native_mtf(value: float, radius: float) -> float:
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("locator radius must be positive and finite")
    return float(value) * CANONICAL_RADIUS / float(radius)


def enforce_mtf50_nyquist(
    *,
    estimator_valid: bool,
    estimator_reason: str,
    canonical_value: float,
    native_value: float,
) -> tuple[bool, str, float | None, float | None]:
    """Return a physically admissible MTF50 pair or an explicit invalid record."""

    if not estimator_valid:
        return False, str(estimator_reason), None, None
    if not math.isfinite(canonical_value) or canonical_value <= 0.0:
        return False, "canonical_mtf50_nonfinite_or_nonpositive", None, None
    if canonical_value > 0.5:
        return False, "canonical_nyquist_exceeded", None, None
    if not math.isfinite(native_value) or native_value <= 0.0:
        return False, "native_mtf50_nonfinite_or_nonpositive", None, None
    if native_value > 0.5:
        return False, "native_nyquist_exceeded", None, None
    return True, str(estimator_reason), float(canonical_value), float(native_value)


def measure_native_crop(
    crop: np.ndarray,
    located_local: Any,
    locator: Any,
    estimator: Any,
) -> list[dict[str, Any]]:
    canonical, rois = locator.normalize_chart(crop, located_local)
    measured: list[dict[str, Any]] = []
    for roi in rois:
        patch = canonical[
            int(roi["y1"]) : int(roi["y2"]),
            int(roi["x1"]) : int(roi["x2"]),
        ]
        result = estimator.estimate_mtf50_v2(
            patch, str(roi["orientation"]), localization_sigma_px=0.75
        )
        canonical_value = float(result.mtf50) if result.valid else math.nan
        native_value = (
            canonical_to_native_mtf(canonical_value, float(located_local.radius))
            if math.isfinite(canonical_value)
            else math.nan
        )
        valid, reason, canonical_output, native_output = enforce_mtf50_nyquist(
            estimator_valid=bool(result.valid),
            estimator_reason=str(result.reason),
            canonical_value=canonical_value,
            native_value=native_value,
        )
        measured.append(
            {
                "roi_id": int(roi["roi_id"]),
                "orientation": str(roi["orientation"]),
                "valid": valid,
                "reason": reason,
                "mtf50_canonical_cy_px": canonical_output,
                "mtf50_native_cy_px": native_output,
            }
        )
    return measured


def bootstrap_condition_mean_ci(
    values: Iterable[float], *, count: int = BOOTSTRAP_COUNT, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("finite one-dimensional condition values required")
    generator = np.random.default_rng(seed)
    sampled = array[
        generator.integers(0, len(array), size=(int(count), len(array)))
    ].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return float(low), float(high)


def bank_frame_ids() -> set[str]:
    values: set[str] = set()
    for path in BANK_MANIFESTS:
        table = pd.read_csv(path)
        required = {"condition", "parent_name"}
        if not required.issubset(table.columns):
            raise ProtocolError(f"bank manifest columns changed: {path}")
        values.update(
            f"{row.condition}/{row.parent_name}"
            for row in table[["condition", "parent_name"]].itertuples(index=False)
        )
    return values


def discover_holdout_inventory(
    raw_root: Path,
    exposed: pd.DataFrame,
    *,
    bank_frame_ids: set[str],
    expected_frames: int = HOLDOUT_FRAMES,
    expected_conditions: int = HOLDOUT_CONDITIONS,
    allowed_conditions: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover by names only; no candidate PNG bytes, hashes, or shapes are read."""

    allowed = all_nonreference_conditions() if allowed_conditions is None else allowed_conditions
    exposed_ids = set(exposed["frame_id"].astype(str))
    records: list[dict[str, Any]] = []
    excluded_exposed = 0
    excluded_bank = 0
    for condition in sorted(allowed):
        speed, distance, iso = parse_condition(condition)
        directory = raw_root / condition
        if not directory.is_dir():
            raise ProtocolError(f"missing raw condition directory: {condition}")
        for path in sorted(directory.glob("*.png"), key=lambda item: item.name):
            if path.parent != directory:
                raise ProtocolError("holdout inventory admitted a non-top-level PNG")
            frame_id = f"{condition}/{path.name}"
            if frame_id in exposed_ids:
                excluded_exposed += 1
                continue
            if frame_id in bank_frame_ids:
                excluded_bank += 1
                continue
            records.append(
                {
                    "frame_id": frame_id,
                    "condition": condition,
                    "filename": path.name,
                    "relative_path": f"{condition}/{path.name}",
                    "speed_kmh": speed,
                    "distance_m": distance,
                    "iso": iso,
                    "router_active": router_active(speed, iso, distance),
                    "route_id": ROUTE_ID if router_active(speed, iso, distance) else "raw_exact_bypass",
                }
            )
    records.sort(
        key=lambda row: (
            row["speed_kmh"], row["distance_m"], row["iso"], row["filename"]
        )
    )
    for ordinal, record in enumerate(records, start=1):
        record["ordinal"] = ordinal
    frame_ids = [str(record["frame_id"]) for record in records]
    conditions = {str(record["condition"]) for record in records}
    overlap_exposed = set(frame_ids) & exposed_ids
    overlap_bank = set(frame_ids) & bank_frame_ids
    if overlap_exposed:
        raise ProtocolError("holdout frame-id overlap with exposed cohort")
    if overlap_bank:
        raise ProtocolError("holdout frame-id overlap with C006 bank parents")
    if len(records) != expected_frames or len(set(frame_ids)) != expected_frames:
        raise ProtocolError(
            f"holdout inventory is not {expected_frames} unique frames: {len(records)}"
        )
    if len(conditions) != expected_conditions:
        raise ProtocolError(
            f"holdout inventory is not the fixed {expected_conditions}-condition set"
        )
    if allowed_conditions is not None and conditions != allowed:
        raise ProtocolError("custom holdout inventory does not cover its allowed condition set")
    active_conditions = {
        str(record["condition"]) for record in records if bool(record["router_active"])
    }
    active_distances = {
        float(record["distance_m"]) for record in records if bool(record["router_active"])
    }
    audit = {
        "frames": len(records),
        "unique_frame_ids": len(set(frame_ids)),
        "condition_cells": len(conditions),
        "frame_id_overlap_with_exposed": len(overlap_exposed),
        "frame_id_overlap_with_c006_bank_parents": len(overlap_bank),
        "excluded_c057_exposed_frame_ids": excluded_exposed,
        "prebank_parent_disjoint_candidates": len(records) + excluded_bank,
        "excluded_c006_bank_parent_frame_ids": excluded_bank,
        "candidate_png_content_read": False,
        "candidate_png_hashes_computed": False,
        "candidate_png_shapes_decoded": False,
        "active_frames": sum(bool(record["router_active"]) for record in records),
        "inactive_frames": sum(not bool(record["router_active"]) for record in records),
        "active_condition_cells": len(active_conditions),
        "inactive_condition_cells": len(conditions - active_conditions),
        "active_distance_cells": len(active_distances),
        "eligible_conditions": sorted(conditions),
    }
    if allowed_conditions is None:
        required = {
            "prebank_parent_disjoint_candidates": PREBANK_PARENT_DISJOINT_CANDIDATES,
            "excluded_c006_bank_parent_frame_ids": BANK_PARENT_EXCLUSIONS,
            "active_frames": HOLDOUT_ACTIVE_FRAMES,
            "inactive_frames": HOLDOUT_FRAMES - HOLDOUT_ACTIVE_FRAMES,
            "active_condition_cells": HOLDOUT_ACTIVE_CONDITIONS,
            "inactive_condition_cells": HOLDOUT_INACTIVE_CONDITIONS,
            "active_distance_cells": HOLDOUT_ACTIVE_DISTANCES,
        }
        if any(audit[key] != value for key, value in required.items()):
            raise ProtocolError("strict parent-disjoint holdout composition changed")
    return records, audit


def create_inventory(path: Path = INVENTORY) -> dict[str, Any]:
    if path.exists() or sidecar_path(path).exists():
        raise FileExistsError(path)
    exposed = pd.read_csv(EXPOSED_COHORT)
    records, audit = discover_holdout_inventory(
        RAW_ROOT,
        exposed,
        bank_frame_ids=bank_frame_ids(),
    )
    predecessor_hash = require_sidecar(V3B_INVENTORY)
    predecessor = read_json(V3B_INVENTORY)
    if [row["frame_id"] for row in records] != [
        row["frame_id"] for row in predecessor.get("records", [])
    ]:
        raise ProtocolError("v3g name-only inventory differs from sealed v3b frame set")
    payload = {
        "schema_version": INVENTORY_SCHEMA,
        "state": "SEALED_NAME_ONLY_BEFORE_UNEXPOSED_PNG_BYTE_ACCESS",
        "created_utc": utc_now(),
        "raw_root": str(RAW_ROOT).replace("\\", "/"),
        "selection_rule": (
            "All top-level PNG names in non-reference condition directories, minus "
            "the union of exact condition/filename IDs in the frozen C057 exposed "
            "cohort and all four frozen C006 bank-parent manifests. Empty conditions "
            "are absent from the eligible set."
        ),
        "evidence_label": EVIDENCE_LABEL,
        "eligible_conditions": audit["eligible_conditions"],
        "eligible_conditions_sha256": canonical_sha256(audit["eligible_conditions"]),
        "audit": audit,
        "records_sha256": canonical_sha256(records),
        "records": records,
        "exclusion_bindings": {
            "c057_exposed_cohort": binding(EXPOSED_COHORT),
            "c006_bank_manifests": [binding(item) for item in BANK_MANIFESTS],
            "sealed_v3b_predecessor": {
                "path": relative_or_absolute(V3B_INVENTORY),
                "sha256": predecessor_hash,
            },
        },
        "one_shot_post_access_audit": {
            "compute_png_sha256_width_height_dtype_only_after_readiness_freeze": True,
            "require_sha256_overlap_with_exposed_and_bank_parents_equal_zero": True,
            "preserve_all_failed_frames_without_exclusion_or_recalibration": True,
        },
    }
    write_json_atomic(path, payload)
    digest = write_sidecar(path)
    return {"path": str(path), "sha256": digest, **audit}


def validate_locator_freeze() -> str:
    digest = require_sidecar(LOCATOR_FREEZE)
    payload = read_json(LOCATOR_FREEZE)
    expected_state = "FROZEN_AFTER_EXPOSED_479_FRAME_VALIDATION_BEFORE_UNEXPOSED_FRAME_ACCESS"
    checks = (
        payload.get("schema_version") == "MERIT-C058-RAW-LOCATOR-EXECUTION-FREEZE-v2",
        payload.get("state") == expected_state,
        payload.get("result") == "PASS",
        payload.get("validation", {}).get("all_predeclared_gates_pass") is True,
        payload.get("seal", {}).get("unexposed_candidate_frames") == 403,
        payload.get("seal", {}).get("pixel_access_before_this_freeze") is False,
        payload.get("seal", {}).get("status") == "SEALED",
    )
    if not all(checks):
        raise ProtocolError("raw locator execution freeze is not sealed PASS")
    for key, expected in (
        ("source", "1c9dc9627115ada3a57404427a0d921bef95c0017189bf7336f5d9092c6e4755"),
        ("contract", "20f3daace10d22d8eedb6d183d6ac262f5f9ac5ae65d7916248a2354950861cf"),
    ):
        record = payload["files"][key]
        resolved = BASE / str(record["path"])
        if sha256_file(resolved) != expected or str(record["sha256"]) != expected:
            raise ProtocolError(f"locator {key} hash mismatch")
        require_sidecar(resolved)
    return digest


def retrospective_profile(output: Path = RETROSPECTIVE_OUTPUT) -> dict[str, Any]:
    route = "band_inverse_a0.500_s0.75_cap8"
    development = pd.read_csv(
        BASE / "artifacts" / "development_v2" / "development_condition_results.csv"
    )
    confirmation = pd.read_csv(
        BASE
        / "artifacts"
        / "confirmation_existing40_v2"
        / "confirmation_condition_results.csv"
    )
    development = development[
        (development["route_id"] == route)
        & (development["speed_kmh"] == 80)
        & (development["iso"] == 100)
    ].copy()
    confirmation = confirmation[confirmation["route_id"] == route].copy()
    combined = pd.concat([development, confirmation], ignore_index=True)
    combined["active"] = [
        router_active(speed, iso, distance)
        for speed, iso, distance in zip(
            combined["speed_kmh"], combined["iso"], combined["distance_m"]
        )
    ]
    combined["delta_abs_error"] = np.where(
        combined["active"],
        combined["abs_error"] - combined["raw_abs_error"],
        0.0,
    )
    condition = (
        combined.groupby(
            ["condition_cell", "speed_kmh", "distance_m", "iso", "active"],
            as_index=False,
        )["delta_abs_error"]
        .mean()
        .sort_values("condition_cell")
    )
    active = condition[condition["active"]]
    overall_ci = bootstrap_condition_mean_ci(condition["delta_abs_error"])
    active_ci = bootstrap_condition_mean_ci(active["delta_abs_error"])

    diagnostic_paths = (
        BASE / "artifacts" / "development_v2" / "development_pixel_diagnostics.csv",
        BASE
        / "artifacts"
        / "confirmation_existing40_v2"
        / "confirmation_pixel_diagnostics.csv",
    )
    diagnostics = pd.concat([pd.read_csv(path) for path in diagnostic_paths])
    diagnostics = diagnostics[
        (diagnostics["route_id"] == route)
        & (diagnostics["speed_kmh"] == 80)
        & (diagnostics["iso"].isin(sorted(ACTIVE_ISOS)))
        & (diagnostics["distance_m"].isin(sorted(ACTIVE_DISTANCES)))
    ]
    payload = {
        "schema_version": "MERIT-C058-SAFETY-ROUTER-RETROSPECTIVE-v3b",
        "state": "OUTCOME_BLIND_AVAILABILITY_CORRECTION_BEFORE_HOLDOUT_PIXEL_ACCESS",
        "created_utc": utc_now(),
        "evidence_label": (
            "Retrospective engineering profile after an outcome-blind name-availability "
            "correction; not independent confirmation."
        ),
        "router": {
            "active_if": (
                "speed_kmh == 80 and iso in {100,200} and "
                "distance_m in {2.5,3.5,4.5}"
            ),
            "active_route_id": route,
            "otherwise": "byte-exact RAW bypass without model inference",
        },
        "condition_counts": {
            "overall": int(len(condition)),
            "active": int(len(active)),
            "inactive": int(len(condition) - len(active)),
            "active_improved": int((active["delta_abs_error"] < 0).sum()),
            "active_worsened": int((active["delta_abs_error"] > 0).sum()),
        },
        "metrics": {
            "overall_mean_delta_abs_error_cy_px": float(condition["delta_abs_error"].mean()),
            "overall_condition_bootstrap_95pct_ci": list(overall_ci),
            "active_mean_delta_abs_error_cy_px": float(active["delta_abs_error"].mean()),
            "active_condition_bootstrap_95pct_ci": list(active_ci),
            "overall_seed_means": {
                str(int(seed)): float(group["delta_abs_error"].mean())
                for seed, group in combined.groupby("seed")
            },
            "active_seed_means": {
                str(int(seed)): float(group["delta_abs_error"].mean())
                for seed, group in combined[combined["active"]].groupby("seed")
            },
            "maximum_active_preclip_overshoot_fraction": float(
                diagnostics["preclip_overshoot_fraction"].max()
            ),
            "maximum_active_introduced_clipping_fraction": float(
                diagnostics["introduced_clipping_fraction"].max()
            ),
            "maximum_active_abs_change_dn": float(diagnostics["max_abs_change_dn"].max()),
        },
        "decision": {
            "headroom_projection_required": False,
            "reason": (
                "The restricted ISO100/200 active strata have zero observed preclip "
                "overshoot and introduced clipping; the high-ISO strata that triggered "
                "v2 overshoot are exact RAW bypasses."
            ),
        },
        "source_bindings": [binding(path) for path in diagnostic_paths]
        + [
            binding(BASE / "artifacts" / "development_v2" / "development_condition_results.csv"),
            binding(
                BASE
                / "artifacts"
                / "confirmation_existing40_v2"
                / "confirmation_condition_results.csv"
            ),
        ],
    }
    write_json_atomic(output, payload)
    write_sidecar(output)
    return payload


def create_contract(path: Path = CONTRACT) -> dict[str, Any]:
    if path.exists() or sidecar_path(path).exists():
        raise FileExistsError(path)
    inventory_hash = require_sidecar(INVENTORY)
    locator_hash = validate_locator_freeze()
    retrospective_hash = require_sidecar(RETROSPECTIVE_OUTPUT)
    public_cohort = pd.read_csv(EXPOSED_COHORT)
    _, public_raw_manifest_hash = load_public_raw_manifest(public_cohort)
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "state": "V3G_FROZEN_BEFORE_PUBLIC_NATIVE_E2E_MODEL_OUTPUTS_AND_UNEXPOSED_PNG_ACCESS",
        "created_utc": utc_now(),
        "development_timing": (
            "The metadata-conditioned experts were calibrated from the preserved v3e "
            "public full-RAW result. After v3f showed a strong negative mean and CI but "
            "slightly overshot the 4.5m/ISO100 reference, only that condition's alpha "
            "was reduced from -1.5 to -0.9. This remains outcome-driven development; the "
            "parent-disjoint holdout has not been decoded and is reserved for one-shot "
            "independent confirmation after this v3g route is frozen."
        ),
        "router": {
            "active_speed_kmh": 80,
            "active_iso_values": [100, 200],
            "active_distance_m_values": [2.5, 3.5, 4.5],
            "active_route": {
                "route_id": ROUTE_ID,
                "sigma": 0.75,
                "sharp_gain": 0.0,
                "cap_dn": 8.0,
                "expert_table": expert_table_payload(),
            },
            "inactive_route": "model-free byte-exact RAW bypass for all other strata",
            "expert_selection": (
                "Frozen by distance and ISO from v3e public development before any "
                "holdout pixel access; no per-frame or holdout outcome selection"
            ),
        },
        "public_end_to_end_development": {
            "frames": PUBLIC_FRAMES,
            "reference_frames": PUBLIC_REFERENCE_FRAMES,
            "active_frames": PUBLIC_ACTIVE_FRAMES,
            "inactive_safety_frames": PUBLIC_INACTIVE_SAFETY_FRAMES,
            "condition_cells": PUBLIC_CONDITIONS,
            "reference_conditions": "60km/h ISO100 at five distances",
            "active_conditions": "80km/h ISO100/200 at 2.5/3.5/4.5 m (six cells)",
            "inactive_safety_conditions": "80km/h ISO100/200 at 5.5/6.5 m (four exact-bypass cells)",
        },
        "coordinate_pipeline": {
            "localization": "Run frozen RAW locator once on full 8K RAW before restoration",
            "native_context": "center +/- 1.05 locator radii, clipped to sensor bounds",
            "restoration": "C057-frozen C006 task-ON model and adapter in native crop coordinates",
            "leakage_control": "Reuse the same RAW LocatorResult translated to crop coordinates for RAW and all routed outputs; never re-localize outputs",
            "measurement": (
                "Frozen affine normalization to 1024 canonical pixels, fixed four "
                "half-open ROIs, MTF50-v2 sigma 0.75; valid only when canonical "
                "and native MTF50 are finite and in (0,0.5] cy/px"
            ),
            "scale": "mtf_native_cy_px = mtf_canonical_cy_px * (320.0 / raw_locator_radius_px)",
        },
        "gates": {
            "locator_coverage_minimum": 0.95,
            "common_complete_case_minimum": 0.95,
            "each_distance_complete_case_minimum": 0.90,
            "mean_delta_abs_error_lt_zero": True,
            "condition_bootstrap_95pct_ci_upper_lt_zero": True,
            "all_active_condition_means_lt_zero": True,
            "all_active_conditioned_expert_means_lt_zero": True,
            "inactive_exact_identity": True,
            "maximum_preclip_overshoot_fraction": 0.01,
            "maximum_introduced_clipping_fraction": 0.005,
            "maximum_abs_change_dn": 12.0,
            "canonical_mtf50_nyquist_max_cy_px": 0.5,
            "native_mtf50_nyquist_max_cy_px": 0.5,
        },
        "bootstrap": {
            "unit": "condition cell",
            "count": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
        },
        "prospective_holdout": {
            "frames": HOLDOUT_FRAMES,
            "condition_cells": HOLDOUT_CONDITIONS,
            "active_frames": HOLDOUT_ACTIVE_FRAMES,
            "active_condition_cells": HOLDOUT_ACTIVE_CONDITIONS,
            "active_distance_cells": HOLDOUT_ACTIVE_DISTANCES,
            "inactive_condition_cells": HOLDOUT_INACTIVE_CONDITIONS,
            "inventory_sha256": inventory_hash,
            "evidence_label": EVIDENCE_LABEL,
            "one_shot": (
                "One attempt identity; interrupted execution may resume only with the "
                "same execution-freeze, inventory, code, and validated immutable progress."
            ),
            "hash_overlap_audit_after_access": "zero overlap with C057 exposed and C006 bank parent SHA-256 sets",
            "failure_policy": "preserve every failure; no manual ROI, exclusion, threshold change, or outcome-based retry",
        },
        "bindings": {
            "runner": binding(Path(__file__)),
            "tests": binding(TEST_CODE),
            "adapter": binding(ADAPTER_CODE),
            "locator": binding(LOCATOR_CODE, sidecar=True),
            "locator_execution_freeze": {
                "path": relative_or_absolute(LOCATOR_FREEZE),
                "sha256": locator_hash,
            },
            "inventory": {"path": relative_or_absolute(INVENTORY), "sha256": inventory_hash},
            "public_raw_manifest": {
                "path": relative_or_absolute(PUBLIC_RAW_MANIFEST),
                "sha256": public_raw_manifest_hash,
            },
            "retrospective_profile": {
                "path": relative_or_absolute(RETROSPECTIVE_OUTPUT),
                "sha256": retrospective_hash,
            },
            "v3e_public_calibration_final": binding(V3E_PUBLIC_FINAL, sidecar=True),
            "v3f_public_diagnostic": binding(V3F_DIAGNOSTIC, sidecar=True),
            "c057_runner": binding(C057 / "run_restoration_mtf50_v2.py"),
            "c057_restoration_freeze": binding(
                C057 / "protocol" / "mtf50_v2_restoration_execution_freeze.json",
                sidecar=True,
            ),
            "exposed_cohort": binding(EXPOSED_COHORT),
        },
    }
    write_json_atomic(path, payload)
    digest = write_sidecar(path)
    return {"path": str(path), "sha256": digest}


def load_contract(path: Path = CONTRACT) -> tuple[dict[str, Any], str]:
    digest = require_sidecar(path)
    payload = read_json(path)
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ProtocolError("unexpected v3g contract schema")
    if payload.get("state") != "V3G_FROZEN_BEFORE_PUBLIC_NATIVE_E2E_MODEL_OUTPUTS_AND_UNEXPOSED_PNG_ACCESS":
        raise ProtocolError("v3g contract state changed")
    router = payload.get("router", {})
    prospective = payload.get("prospective_holdout", {})
    gates = payload.get("gates", {})
    if (
        router.get("active_speed_kmh") != ACTIVE_SPEED
        or router.get("active_iso_values") != sorted(ACTIVE_ISOS)
        or router.get("active_distance_m_values") != sorted(ACTIVE_DISTANCES)
        or router.get("active_route", {}).get("route_id") != ROUTE_ID
        or router.get("active_route", {}).get("expert_table") != expert_table_payload()
        or prospective.get("frames") != HOLDOUT_FRAMES
        or prospective.get("condition_cells") != HOLDOUT_CONDITIONS
        or prospective.get("active_frames") != HOLDOUT_ACTIVE_FRAMES
        or prospective.get("active_condition_cells") != HOLDOUT_ACTIVE_CONDITIONS
        or prospective.get("active_distance_cells") != HOLDOUT_ACTIVE_DISTANCES
        or prospective.get("inactive_condition_cells") != HOLDOUT_INACTIVE_CONDITIONS
        or prospective.get("evidence_label") != EVIDENCE_LABEL
        or gates.get("canonical_mtf50_nyquist_max_cy_px") != 0.5
        or gates.get("native_mtf50_nyquist_max_cy_px") != 0.5
    ):
        raise ProtocolError("v3g contract router/holdout composition changed")
    contract_bindings = payload.get("bindings", {})
    if set(contract_bindings) != {
        "runner",
        "tests",
        "adapter",
        "locator",
        "locator_execution_freeze",
        "inventory",
        "public_raw_manifest",
        "retrospective_profile",
        "v3e_public_calibration_final",
        "v3f_public_diagnostic",
        "c057_runner",
        "c057_restoration_freeze",
        "exposed_cohort",
    }:
        raise ProtocolError("v3g contract binding set changed")
    for label, record in contract_bindings.items():
        resolved = resolve_path(str(record["path"]))
        if not resolved.is_file() or sha256_file(resolved) != record["sha256"]:
            raise ProtocolError(f"v3g contract binding mismatch: {label}")
    validate_locator_freeze()
    inventory = read_json(INVENTORY)
    if inventory.get("records_sha256") != canonical_sha256(inventory.get("records")):
        raise ProtocolError("holdout inventory record self-hash mismatch")
    _, public_raw_manifest_hash = load_public_raw_manifest(pd.read_csv(EXPOSED_COHORT))
    if public_raw_manifest_hash != contract_bindings["public_raw_manifest"]["sha256"]:
        raise ProtocolError("v3g public full-RAW manifest binding mismatch")
    return payload, digest


def _model_callback(module: Any, protocol: dict[str, Any], models: Any, tiled: Any, device: Any):
    tiling = protocol["restoration"]["tiling"]

    def run(crop: np.ndarray, seed: int) -> np.ndarray:
        return tiled(
            crop,
            models[("on", str(seed))],
            device,
            tile_size=int(tiling["tile_size"]),
            overlap=int(tiling["overlap"]),
            batch_size=int(tiling["batch_size"]),
            blend_floor=float(tiling["positive_hann_floor"]),
            amp_dtype=str(protocol["restoration"]["amp_dtype"]),
        )

    return run


def _evaluate_image(
    image: np.ndarray,
    *,
    speed: int,
    iso: int,
    distance_m: float,
    locator: Any,
    estimator: Any,
    model_callback: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    located = locator.locate_chart(image)
    x0, y0, x1, y1 = native_context_bounds(image.shape, located)
    raw_crop = image[y0:y1, x0:x1]
    located_local = local_locator(located, x0, y0)
    routed = route_native_crop(raw_crop, speed, iso, distance_m, model_callback)
    events: list[dict[str, Any]] = []
    for run_key, (output, diagnostics) in routed.items():
        if run_key == "raw_none" and not np.array_equal(output, raw_crop):
            raise ProtocolError("RAW event changed pixels")
        if not router_active(speed, iso, distance_m) and not np.array_equal(output, raw_crop):
            raise ProtocolError("inactive route changed pixels")
        events.append(
            {
                "run_key": run_key,
                "diagnostics": diagnostics,
                "rois": measure_native_crop(
                    output, located_local, locator, estimator
                ),
            }
        )
    locator_record = {
        "score": float(located.score),
        "center_x_raw": float(located.center_x),
        "center_y_raw": float(located.center_y),
        "radius_raw": float(located.radius),
        "angle_deg": float(located.angle_deg),
        "native_context": [x0, y0, x1, y1],
        "locator_reused_for_all_outputs": True,
    }
    return locator_record, events


def validate_event_coverage(payload: dict[str, Any]) -> None:
    if payload.get("locator_reused_for_all_outputs") is not True:
        raise ProtocolError("RAW locator was not reused for every output")
    active = bool(payload["route_active"])
    expected = [
        "raw_none",
        active_run_key(float(payload["distance_m"]), int(payload["iso"]))
        if active
        else "raw_exact_bypass",
    ]
    observed = [str(event["run_key"]) for event in payload.get("events", [])]
    if observed != expected:
        raise ProtocolError("progress route/event coverage mismatch")


def _validate_roi_and_diagnostics(payload: dict[str, Any]) -> None:
    """Validate all scientific fields in one successful progress payload."""

    validate_event_coverage(payload)
    expected_identity = [
        (1, "vertical_edge"),
        (2, "vertical_edge"),
        (3, "horizontal_edge"),
        (4, "horizontal_edge"),
    ]
    active = bool(payload["route_active"])
    fraction_fields = (
        "residual_cap_fraction",
        "preclip_overshoot_fraction",
        "preclip_below_zero_fraction",
        "preclip_above_one_fraction",
        "introduced_clipping_fraction",
        "introduced_low_clipping_fraction",
        "introduced_high_clipping_fraction",
    )
    magnitude_fields = (
        "uncapped_residual_mean_abs_dn",
        "capped_residual_mean_abs_dn",
        "mean_abs_change_dn",
        "max_abs_change_dn",
    )
    for index, event in enumerate(payload["events"]):
        observed_identity = [
            (int(roi.get("roi_id", -1)), str(roi.get("orientation")))
            for roi in event.get("rois", [])
        ]
        if observed_identity != expected_identity:
            raise ProtocolError("progress ROI identity/orientation mismatch")
        for roi in event["rois"]:
            if not isinstance(roi.get("valid"), bool):
                raise ProtocolError("progress ROI valid flag malformed")
            canonical = roi.get("mtf50_canonical_cy_px")
            native = roi.get("mtf50_native_cy_px")
            if roi["valid"]:
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in (canonical, native)
                ):
                    raise ProtocolError("progress valid ROI MTF is non-finite")
                if not (0.0 < float(canonical) <= 0.5 and 0.0 < float(native) <= 0.5):
                    raise ProtocolError("progress valid ROI MTF outside physical range")
            elif canonical is not None or native is not None:
                raise ProtocolError("progress invalid ROI must have null MTF")
        diagnostic = event.get("diagnostics", {})
        if not isinstance(diagnostic.get("exact_identity"), bool):
            raise ProtocolError("progress diagnostic identity flag malformed")
        for field in fraction_fields:
            value = diagnostic.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ProtocolError(f"progress diagnostic fraction malformed: {field}")
        for field in magnitude_fields:
            value = diagnostic.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ProtocolError(f"progress diagnostic magnitude malformed: {field}")
        if index == 0 and (event["run_key"] != "raw_none" or not diagnostic["exact_identity"]):
            raise ProtocolError("progress RAW event malformed")
        if not active and (
            not diagnostic["exact_identity"]
            or float(diagnostic["max_abs_change_dn"]) != 0.0
            or not str(event["run_key"]).startswith("raw_")
        ):
            raise ProtocolError("inactive progress is not exact RAW identity")


def validate_progress_payload(
    payload: dict[str, Any],
    expected: dict[str, Any],
    *,
    study: str,
    contract_hash: str,
    c057_hash: str,
    locator_freeze_hash: str,
    public_raw_manifest_hash: str | None = None,
    execution_freeze_hash: str | None = None,
    inventory_hash: str | None = None,
    selection_hash: str | None = None,
) -> None:
    """Fail closed on frame identity, frozen bindings, routing, ROI, and diagnostics."""

    scalar_fields = (
        "ordinal",
        "frame_id",
        "condition_cell",
        "speed_kmh",
        "distance_m",
        "iso",
        "route_active",
    )
    if payload.get("schema_version") != PROGRESS_SCHEMA or payload.get("study") != study:
        raise ProtocolError("progress schema/study mismatch")
    for field in scalar_fields:
        if payload.get(field) != expected.get(field):
            raise ProtocolError(f"progress expected field mismatch: {field}")
    bindings = {
        "contract_sha256": contract_hash,
        "c057_restoration_freeze_sha256": c057_hash,
        "locator_execution_freeze_sha256": locator_freeze_hash,
        "public_raw_manifest_sha256": public_raw_manifest_hash,
        "execution_freeze_sha256": execution_freeze_hash,
        "inventory_sha256": inventory_hash,
        "selection_sha256": selection_hash,
    }
    for field, expected_value in bindings.items():
        if expected_value is not None and payload.get(field) != expected_value:
            raise ProtocolError(f"progress frozen binding mismatch: {field}")
    if expected.get("image_sha256") is not None and payload.get("image_sha256") != expected["image_sha256"]:
        raise ProtocolError("progress image hash mismatch")
    status = payload.get("status")
    if status == "failed":
        if payload.get("events") != [] or not str(payload.get("reason", "")):
            raise ProtocolError("failed progress must preserve reason and zero events")
        return
    if status != "ok":
        raise ProtocolError("progress status malformed")
    locator_record = payload.get("locator", {})
    if (
        payload.get("locator_reused_for_all_outputs") is not True
        or locator_record.get("locator_reused_for_all_outputs") is not True
        or not isinstance(locator_record.get("score"), (int, float))
        or float(locator_record["score"]) < 0.45
    ):
        raise ProtocolError("progress locator evidence malformed")
    _validate_roi_and_diagnostics(payload)


def _progress_path(output: Path, ordinal: int, frame_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", frame_id)
    return output / "progress" / f"{ordinal:04d}_{safe}.json"


def _public_frames(cohort: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    selected = cohort[
        ((cohort["speed_kmh"].astype(int) == 60) & (cohort["iso"].astype(int) == 100))
        | ((cohort["speed_kmh"].astype(int) == 80) & cohort["iso"].astype(int).isin([100, 200]))
    ].copy()
    frames = list(selected.sort_values(["condition", "frame", "roi_id"]).groupby("frame_id", sort=True))
    frame_meta = selected.drop_duplicates("frame_id").copy()
    active = np.array(
        [
            router_active(speed, iso, distance)
            for speed, iso, distance in zip(
                frame_meta["speed_kmh"], frame_meta["iso"], frame_meta["distance_m"]
            )
        ],
        dtype=bool,
    )
    reference = (
        (frame_meta["speed_kmh"].astype(int) == REFERENCE_SPEED)
        & (frame_meta["iso"].astype(int) == REFERENCE_ISO)
    ).to_numpy()
    inactive_safety = ~(active | reference)
    condition_meta = frame_meta.drop_duplicates("condition")
    active_condition_count = sum(
        router_active(speed, iso, distance)
        for speed, iso, distance in zip(
            condition_meta["speed_kmh"],
            condition_meta["iso"],
            condition_meta["distance_m"],
        )
    )
    reference_condition_count = int(
        (
            (condition_meta["speed_kmh"].astype(int) == REFERENCE_SPEED)
            & (condition_meta["iso"].astype(int) == REFERENCE_ISO)
        ).sum()
    )
    checks = (
        len(frames) == PUBLIC_FRAMES,
        selected["condition"].nunique() == PUBLIC_CONDITIONS,
        int(reference.sum()) == PUBLIC_REFERENCE_FRAMES,
        int(active.sum()) == PUBLIC_ACTIVE_FRAMES,
        int(inactive_safety.sum()) == PUBLIC_INACTIVE_SAFETY_FRAMES,
        reference_condition_count == PUBLIC_REFERENCE_CONDITIONS,
        active_condition_count == PUBLIC_ACTIVE_CONDITIONS,
        PUBLIC_CONDITIONS - reference_condition_count - active_condition_count
        == PUBLIC_INACTIVE_SAFETY_CONDITIONS,
    )
    if not all(checks):
        raise ProtocolError("public v3g development cohort composition changed")
    return frames


def create_public_raw_manifest(path: Path = PUBLIC_RAW_MANIFEST) -> dict[str, Any]:
    """Freeze the exact full-resolution public RAW inputs before v3g model output."""

    if path.exists() or sidecar_path(path).exists():
        raise FileExistsError(path)
    module = c057_module()
    _, _, cohort, c057_hash = module.load_verified_environment(module.DEFAULT_FREEZE)
    frames = _public_frames(cohort)
    locator = locator_module()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, (frame_id, frame) in enumerate(frames, start=1):
        first = frame.iloc[0]
        relative_path = str(frame_id).replace("\\", "/")
        image_path = RAW_ROOT / relative_path
        if not image_path.is_file():
            raise ProtocolError(f"missing public full RAW: {relative_path}")
        image_hash = sha256_file(image_path)
        image = locator.read_gray(image_path)
        if image.shape != (5460, 8192) or image.dtype != np.uint8:
            raise ProtocolError(
                f"public full RAW shape/dtype changed: {relative_path} "
                f"{image.shape} {image.dtype}"
            )
        records.append(
            {
                "ordinal": ordinal,
                "frame_id": str(frame_id),
                "condition_cell": str(first["condition"]),
                "relative_path": relative_path,
                "speed_kmh": int(first["speed_kmh"]),
                "distance_m": float(first["distance_m"]),
                "iso": int(first["iso"]),
                "route_active": router_active(
                    int(first["speed_kmh"]),
                    int(first["iso"]),
                    float(first["distance_m"]),
                ),
                "image_sha256": image_hash,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "dtype": str(image.dtype),
            }
        )
        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(frames):
            print(
                json.dumps(
                    {
                        "state": "FREEZING_PUBLIC_FULL_RAW",
                        "completed_frames": ordinal,
                        "total_frames": len(frames),
                        "session_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len({record["frame_id"] for record in records}) != PUBLIC_FRAMES:
        raise ProtocolError("public full-RAW manifest frame IDs are not unique")
    if len({record["image_sha256"] for record in records}) != PUBLIC_FRAMES:
        raise ProtocolError("public full-RAW manifest image hashes are not unique")
    payload = {
        "schema_version": PUBLIC_RAW_MANIFEST_SCHEMA,
        "state": "FROZEN_BEFORE_V3E_PUBLIC_MODEL_OUTPUTS",
        "created_utc": utc_now(),
        "raw_root": str(RAW_ROOT).replace("\\", "/"),
        "frames": PUBLIC_FRAMES,
        "full_resolution_shape_hw": [5460, 8192],
        "dtype": "uint8",
        "records_sha256": canonical_sha256(records),
        "records": records,
        "bindings": {
            "c057_exposed_cohort": binding(EXPOSED_COHORT),
            "c057_restoration_freeze_sha256": c057_hash,
            "locator_execution_freeze_sha256": validate_locator_freeze(),
        },
    }
    write_json_atomic(path, payload)
    digest = write_sidecar(path)
    return {"path": str(path), "sha256": digest, "frames": len(records)}


def load_public_raw_manifest(
    cohort: pd.DataFrame, path: Path = PUBLIC_RAW_MANIFEST
) -> tuple[dict[str, dict[str, Any]], str]:
    """Validate the frozen public RAW identities without decoding image pixels."""

    digest = require_sidecar(path)
    payload = read_json(path)
    records = payload.get("records")
    if (
        payload.get("schema_version") != PUBLIC_RAW_MANIFEST_SCHEMA
        or payload.get("state") != "FROZEN_BEFORE_V3E_PUBLIC_MODEL_OUTPUTS"
        or payload.get("frames") != PUBLIC_FRAMES
        or payload.get("full_resolution_shape_hw") != [5460, 8192]
        or payload.get("dtype") != "uint8"
        or not isinstance(records, list)
        or payload.get("records_sha256") != canonical_sha256(records)
    ):
        raise ProtocolError("public full-RAW manifest metadata changed")
    expected_frames = _public_frames(cohort)
    if len(records) != len(expected_frames):
        raise ProtocolError("public full-RAW manifest record count changed")
    by_frame: dict[str, dict[str, Any]] = {}
    for ordinal, ((frame_id, frame), record) in enumerate(
        zip(expected_frames, records), start=1
    ):
        first = frame.iloc[0]
        expected = {
            "ordinal": ordinal,
            "frame_id": str(frame_id),
            "condition_cell": str(first["condition"]),
            "relative_path": str(frame_id).replace("\\", "/"),
            "speed_kmh": int(first["speed_kmh"]),
            "distance_m": float(first["distance_m"]),
            "iso": int(first["iso"]),
            "route_active": router_active(
                int(first["speed_kmh"]), int(first["iso"]), float(first["distance_m"])
            ),
            "width": 8192,
            "height": 5460,
            "dtype": "uint8",
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ProtocolError(f"public full-RAW manifest mismatch: {frame_id} {key}")
        image_hash = record.get("image_sha256")
        if not isinstance(image_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            raise ProtocolError(f"public full-RAW hash malformed: {frame_id}")
        if str(frame_id) in by_frame:
            raise ProtocolError("duplicate public full-RAW frame ID")
        by_frame[str(frame_id)] = record
    if len({record["image_sha256"] for record in records}) != PUBLIC_FRAMES:
        raise ProtocolError("duplicate public full-RAW content hash")
    bindings = payload.get("bindings", {})
    if (
        bindings.get("c057_exposed_cohort", {}).get("sha256")
        != sha256_file(EXPOSED_COHORT)
        or bindings.get("locator_execution_freeze_sha256") != validate_locator_freeze()
    ):
        raise ProtocolError("public full-RAW manifest source binding mismatch")
    return by_frame, digest


def run_public_development(
    contract_path: Path = CONTRACT, output: Path = DEVELOPMENT_OUTPUT
) -> None:
    _, contract_hash = load_contract(contract_path)
    module = c057_module()
    protocol, _, cohort, c057_hash = module.load_verified_environment(module.DEFAULT_FREEZE)
    frames = _public_frames(cohort)
    locator = locator_module()
    locator_hash = validate_locator_freeze()
    public_raw_records, public_raw_manifest_hash = load_public_raw_manifest(cohort)
    completed = 0
    for ordinal, (frame_id, frame) in enumerate(frames, start=1):
        path = _progress_path(output, ordinal, frame_id)
        if not path.exists():
            continue
        first = frame.iloc[0]
        payload = read_json(path)
        validate_progress_payload(
            payload,
            {
                "ordinal": ordinal,
                "frame_id": frame_id,
                "condition_cell": str(first["condition"]),
                "speed_kmh": int(first["speed_kmh"]),
                "distance_m": float(first["distance_m"]),
                "iso": int(first["iso"]),
                "route_active": router_active(
                    int(first["speed_kmh"]), int(first["iso"]), float(first["distance_m"])
                ),
                "image_sha256": public_raw_records[frame_id]["image_sha256"],
            },
            study=PUBLIC_STUDY,
            contract_hash=contract_hash,
            c057_hash=c057_hash,
            locator_freeze_hash=locator_hash,
            public_raw_manifest_hash=public_raw_manifest_hash,
        )
        completed += 1
    if completed == len(frames):
        return

    device = module.D0.require_cuda()
    models, tiled = module.D0.load_models(protocol, device)
    models = {key: model for key, model in models.items() if key[0] == "on"}
    model_callback = _model_callback(module, protocol, models, tiled, device)
    started = time.perf_counter()
    processed = 0
    for ordinal, (frame_id, frame) in enumerate(frames, start=1):
        path = _progress_path(output, ordinal, frame_id)
        if path.exists():
            continue
        first = frame.iloc[0]
        frame_started = time.perf_counter()
        raw_record = public_raw_records[frame_id]
        image_path = RAW_ROOT / str(raw_record["relative_path"])
        actual_image_hash = sha256_file(image_path)
        if actual_image_hash != raw_record["image_sha256"]:
            raise ProtocolError(f"public full-RAW hash changed: {frame_id}")
        image = locator.read_gray(image_path)
        if (
            image.shape != (int(raw_record["height"]), int(raw_record["width"]))
            or str(image.dtype) != raw_record["dtype"]
        ):
            raise ProtocolError(f"public full-RAW shape/dtype changed: {frame_id}")
        speed, iso, distance = (
            int(first["speed_kmh"]),
            int(first["iso"]),
            float(first["distance_m"]),
        )
        try:
            locator_record, events = _evaluate_image(
                image,
                speed=speed,
                iso=iso,
                distance_m=distance,
                locator=locator,
                estimator=module,
                model_callback=model_callback,
            )
            payload = {
                "schema_version": PROGRESS_SCHEMA,
                "study": PUBLIC_STUDY,
                "ordinal": ordinal,
                "frame_id": frame_id,
                "condition_cell": str(first["condition"]),
                "speed_kmh": speed,
                "distance_m": float(first["distance_m"]),
                "iso": iso,
                "route_active": router_active(speed, iso, distance),
                "is_reference": speed == REFERENCE_SPEED and iso == REFERENCE_ISO,
                "image_sha256": actual_image_hash,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "image_dtype": str(image.dtype),
                "contract_sha256": contract_hash,
                "c057_restoration_freeze_sha256": c057_hash,
                "locator_execution_freeze_sha256": locator_hash,
                "public_raw_manifest_sha256": public_raw_manifest_hash,
                "locator_reused_for_all_outputs": True,
                "locator": locator_record,
                "events": events,
                "status": "ok",
                "reason": "ok",
                "elapsed_seconds": time.perf_counter() - frame_started,
                "committed_utc": utc_now(),
            }
        except Exception as exc:
            payload = {
                "schema_version": PROGRESS_SCHEMA,
                "study": PUBLIC_STUDY,
                "ordinal": ordinal,
                "frame_id": frame_id,
                "condition_cell": str(first["condition"]),
                "speed_kmh": speed,
                "distance_m": float(first["distance_m"]),
                "iso": iso,
                "route_active": router_active(speed, iso, distance),
                "is_reference": speed == REFERENCE_SPEED and iso == REFERENCE_ISO,
                "image_sha256": actual_image_hash,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "image_dtype": str(image.dtype),
                "contract_sha256": contract_hash,
                "c057_restoration_freeze_sha256": c057_hash,
                "locator_execution_freeze_sha256": locator_hash,
                "public_raw_manifest_sha256": public_raw_manifest_hash,
                "locator_reused_for_all_outputs": True,
                "events": [],
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - frame_started,
                "committed_utc": utc_now(),
            }
        write_json_atomic(path, payload)
        processed += 1
        done = completed + processed
        if processed == 1 or processed % 10 == 0 or done == len(frames):
            print(
                json.dumps(
                    {
                        "state": "RUNNING" if done < len(frames) else "INFERENCE_COMPLETE",
                        "completed_frames": done,
                        "total_frames": len(frames),
                        "session_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def _collect_progress(output: Path, expected: int) -> list[dict[str, Any]]:
    paths = sorted((output / "progress").glob("*.json"))
    if len(paths) != expected:
        raise ProtocolError(f"expected {expected} progress files, found {len(paths)}")
    return [read_json(path) for path in paths]


def collect_public_progress(
    contract_path: Path, output: Path
) -> tuple[list[dict[str, Any]], str]:
    _, contract_hash = load_contract(contract_path)
    module = c057_module()
    _, _, cohort, c057_hash = module.load_verified_environment(module.DEFAULT_FREEZE)
    frames = _public_frames(cohort)
    locator_hash = validate_locator_freeze()
    public_raw_records, public_raw_manifest_hash = load_public_raw_manifest(cohort)
    progress = _collect_progress(output, len(frames))
    for ordinal, ((frame_id, frame), payload) in enumerate(zip(frames, progress), start=1):
        first = frame.iloc[0]
        validate_progress_payload(
            payload,
            {
                "ordinal": ordinal,
                "frame_id": frame_id,
                "condition_cell": str(first["condition"]),
                "speed_kmh": int(first["speed_kmh"]),
                "distance_m": float(first["distance_m"]),
                "iso": int(first["iso"]),
                "route_active": router_active(
                    int(first["speed_kmh"]), int(first["iso"]), float(first["distance_m"])
                ),
                "image_sha256": public_raw_records[frame_id]["image_sha256"],
            },
            study=PUBLIC_STUDY,
            contract_hash=contract_hash,
            c057_hash=c057_hash,
            locator_freeze_hash=locator_hash,
            public_raw_manifest_hash=public_raw_manifest_hash,
        )
    return progress, contract_hash


def _rows_from_progress(progress: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for frame in progress:
        if frame.get("status") != "ok":
            continue
        validate_event_coverage(frame)
        for event in frame["events"]:
            diagnostic_rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "condition_cell": frame["condition_cell"],
                    "speed_kmh": frame["speed_kmh"],
                    "distance_m": frame["distance_m"],
                    "iso": frame["iso"],
                    "route_active": frame["route_active"],
                    "run_key": event["run_key"],
                    **event["diagnostics"],
                }
            )
            for roi in event["rois"]:
                roi_rows.append(
                    {
                        "frame_id": frame["frame_id"],
                        "condition_cell": frame["condition_cell"],
                        "speed_kmh": frame["speed_kmh"],
                        "distance_m": frame["distance_m"],
                        "iso": frame["iso"],
                        "route_active": frame["route_active"],
                        "is_reference": frame.get("is_reference", False),
                        "run_key": event["run_key"],
                        **roi,
                    }
                )
    return pd.DataFrame(roi_rows), pd.DataFrame(diagnostic_rows)


def _common_case(
    roi: pd.DataFrame,
    expected_frames: int,
    *,
    expected_frames_by_distance: dict[float, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    identities = ["frame_id", "condition_cell", "distance_m", "roi_id", "orientation"]
    required_runs = 2
    valid = roi[roi["valid"] == True].copy()  # noqa: E712
    counts = valid.groupby(identities)["run_key"].nunique()
    common_ids = counts[counts == required_runs].reset_index()[identities]
    common = roi.merge(common_ids, on=identities, how="inner")
    denominator = expected_frames * 4
    fraction = len(common_ids) / denominator
    per_distance = []
    observed_distances = {float(value) for value in roi["distance_m"].unique()}
    distances = (
        sorted(expected_frames_by_distance)
        if expected_frames_by_distance is not None
        else sorted(observed_distances)
    )
    for distance in distances:
        frames = (
            int(expected_frames_by_distance[distance])
            if expected_frames_by_distance is not None
            else int(roi[roi["distance_m"] == distance]["frame_id"].nunique())
        )
        count = len(common_ids[common_ids["distance_m"] == distance])
        per_distance.append(
            {
                "distance_m": float(distance),
                "common_roi_identities": int(count),
                "expected_roi_identities": int(frames * 4),
                "fraction": float(count / (frames * 4)) if frames else 0.0,
            }
        )
    return common, {
        "common_roi_identities": int(len(common_ids)),
        "expected_roi_identities": int(denominator),
        "fraction": float(fraction),
        "minimum_distance_fraction": float(min(item["fraction"] for item in per_distance)),
        "per_distance": per_distance,
    }


def _reference_values(common: pd.DataFrame) -> dict[str, float]:
    reference = common[(common["is_reference"] == True) & (common["run_key"] == "raw_none")].copy()  # noqa: E712
    frames = (
        reference.groupby(["frame_id", "distance_m"], as_index=False)["mtf50_native_cy_px"]
        .mean()
    )
    values = frames.groupby("distance_m")["mtf50_native_cy_px"].median()
    if len(values) != 5 or not np.isfinite(values.to_numpy()).all():
        raise ProtocolError("could not freeze five canonicalized RAW references")
    return {f"{float(distance):.1f}": float(value) for distance, value in values.items()}


def _condition_table(
    common: pd.DataFrame, references: dict[str, float], *, include_reference: bool
) -> pd.DataFrame:
    working = common.copy()
    if not include_reference:
        working = working[working["is_reference"] != True].copy()  # noqa: E712
    frame_values = (
        working.groupby(
            [
                "frame_id",
                "condition_cell",
                "speed_kmh",
                "distance_m",
                "iso",
                "route_active",
                "run_key",
            ],
            as_index=False,
        )["mtf50_native_cy_px"]
        .mean()
    )
    condition_values = (
        frame_values.groupby(
            ["condition_cell", "speed_kmh", "distance_m", "iso", "route_active", "run_key"],
            as_index=False,
        )["mtf50_native_cy_px"]
        .median()
    )
    raw = condition_values[condition_values["run_key"] == "raw_none"][
        ["condition_cell", "mtf50_native_cy_px"]
    ].rename(columns={"mtf50_native_cy_px": "raw_mtf50"})
    routed = condition_values[condition_values["run_key"] != "raw_none"].copy()
    extracted_seed = routed["run_key"].str.extract(r"_(\d)$")[0]
    routed["seed"] = [
        int(value) if isinstance(value, str) else -1 for value in extracted_seed
    ]
    routed = routed.merge(raw, on="condition_cell", how="left", validate="many_to_one")
    routed["reference_mtf50"] = [
        references[f"{float(distance):.1f}"] for distance in routed["distance_m"]
    ]
    routed["abs_error"] = np.abs(
        routed["mtf50_native_cy_px"] - routed["reference_mtf50"]
    )
    routed["raw_abs_error"] = np.abs(routed["raw_mtf50"] - routed["reference_mtf50"])
    routed["delta_abs_error"] = routed["abs_error"] - routed["raw_abs_error"]
    return routed.sort_values(["condition_cell", "seed"]).reset_index(drop=True)


def evaluate_gate(
    condition: pd.DataFrame,
    *,
    ci: tuple[float, float],
    coverage: float,
    locator_coverage: float,
    maximum_introduced_clipping_fraction: float,
    maximum_abs_change_dn: float,
    inactive_exact_identity: bool,
    active_ci: tuple[float, float] | None = None,
    maximum_preclip_overshoot_fraction: float = 0.0,
    minimum_distance_coverage: float = 1.0,
) -> dict[str, Any]:
    if active_ci is None:
        active_ci = ci
    condition_means = condition.groupby(["condition_cell", "active"])[
        "delta_abs_error"
    ].mean().reset_index()
    active = condition[condition["active"]]
    active_condition = condition_means[condition_means["active"]]
    expert_seed_means = active.groupby("seed")["delta_abs_error"].mean()
    checks = {
        "locator_coverage": locator_coverage >= 0.95,
        "common_complete_case_coverage": coverage >= 0.95,
        "each_distance_complete_case_coverage": minimum_distance_coverage >= 0.90,
        "mean_delta_lt_zero": float(condition_means["delta_abs_error"].mean()) < 0.0,
        "ci_upper_lt_zero": float(ci[1]) < 0.0,
        "active_ci_upper_lt_zero": float(active_ci[1]) < 0.0,
        "active_mean_lt_zero": len(active) > 0 and float(active["delta_abs_error"].mean()) < 0.0,
        "all_active_condition_means_lt_zero": len(active_condition) > 0
        and bool((active_condition["delta_abs_error"] < 0.0).all()),
        "inactive_exact_identity": bool(inactive_exact_identity),
        "preclip_overshoot": maximum_preclip_overshoot_fraction <= 0.01,
        "introduced_clipping": maximum_introduced_clipping_fraction <= 0.005,
        "maximum_abs_change_dn": maximum_abs_change_dn <= 12.0,
    }
    return {
        "gate": "PASS" if all(checks.values()) else "FAIL",
        "gate_checks": checks,
        "condition_mean_delta_abs_error_cy_px": float(condition_means["delta_abs_error"].mean()),
        "condition_cluster_bootstrap_95pct_ci": [float(ci[0]), float(ci[1])],
        "active_condition_cluster_bootstrap_95pct_ci": [
            float(active_ci[0]),
            float(active_ci[1]),
        ],
        "active_mean_delta_abs_error_cy_px": float(active["delta_abs_error"].mean()) if len(active) else None,
        "active_expert_seed_means_descriptive": {
            str(int(seed)): float(value) for seed, value in expert_seed_means.items()
        },
    }


def aggregate_public_development(
    contract_path: Path = CONTRACT, output: Path = DEVELOPMENT_OUTPUT
) -> dict[str, Any]:
    progress, contract_hash = collect_public_progress(contract_path, output)
    roi, diagnostics = _rows_from_progress(progress)
    expected_by_distance = {
        float(distance): len({item["frame_id"] for item in progress if float(item["distance_m"]) == float(distance)})
        for distance in (2.5, 3.5, 4.5, 5.5, 6.5)
    }
    common, coverage = _common_case(
        roi, PUBLIC_FRAMES, expected_frames_by_distance=expected_by_distance
    )
    references = _reference_values(common)
    condition = _condition_table(common, references, include_reference=False)
    condition = condition[condition["route_active"]].copy()
    condition["active"] = True
    condition_means = condition.groupby("condition_cell")["delta_abs_error"].mean()
    ci = bootstrap_condition_mean_ci(condition_means)
    active_diagnostics = diagnostics[diagnostics["route_active"] == True]  # noqa: E712
    locator_coverage = (
        sum(item.get("status") == "ok" for item in progress) / PUBLIC_FRAMES
    )
    inactive_diagnostics = diagnostics[diagnostics["route_active"] == False]  # noqa: E712
    inactive_exact_identity = bool(
        len(inactive_diagnostics)
        and inactive_diagnostics["exact_identity"].astype(bool).all()
        and (inactive_diagnostics["max_abs_change_dn"].astype(float) == 0.0).all()
    )
    summary = evaluate_gate(
        condition,
        ci=ci,
        coverage=coverage["fraction"],
        locator_coverage=locator_coverage,
        minimum_distance_coverage=coverage["minimum_distance_fraction"],
        maximum_preclip_overshoot_fraction=float(active_diagnostics["preclip_overshoot_fraction"].max()),
        maximum_introduced_clipping_fraction=float(active_diagnostics["introduced_clipping_fraction"].max()),
        maximum_abs_change_dn=float(active_diagnostics["max_abs_change_dn"].max()),
        inactive_exact_identity=inactive_exact_identity,
    )
    output.mkdir(parents=True, exist_ok=True)
    roi_path = output / "public_e2e_roi_results.csv"
    condition_path = output / "public_e2e_condition_results.csv"
    coverage_path = output / "public_e2e_complete_case.json"
    diagnostics_path = output / "public_e2e_pixel_diagnostics.csv"
    final_path = output / "public_e2e_development_final.json"
    for path in (roi_path, condition_path, coverage_path, diagnostics_path, final_path):
        if path.exists() or sidecar_path(path).exists():
            raise FileExistsError(path)
    roi.to_csv(roi_path, index=False)
    condition.to_csv(condition_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    write_json_atomic(coverage_path, coverage)
    for path in (roi_path, condition_path, diagnostics_path, coverage_path):
        write_sidecar(path)
    final = {
        "schema_version": PUBLIC_FINAL_SCHEMA,
        "state": "PUBLIC_NATIVE_E2E_DEVELOPMENT_COMPLETE",
        "evidence_label": "Outcome-driven public development; not independent confirmation",
        "completed_utc": utc_now(),
        "contract_sha256": contract_hash,
        "counts": {
            "frames": PUBLIC_FRAMES,
            "located_frames": int(sum(item.get("status") == "ok" for item in progress)),
            "active_condition_cells": int(condition["condition_cell"].nunique()),
        },
        "reference_mtf50_native_cy_px_by_distance": references,
        "coverage": coverage,
        "summary": summary,
        "artifacts": {
            "roi": binding(roi_path, sidecar=True),
            "condition": binding(condition_path, sidecar=True),
            "diagnostics": binding(diagnostics_path, sidecar=True),
            "coverage": binding(coverage_path, sidecar=True),
        },
    }
    write_json_atomic(final_path, final)
    write_sidecar(final_path)
    return final


def assert_nested_close(actual: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ProtocolError(f"release structure mismatch at {path}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ProtocolError(f"release list mismatch at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_nested_close(left, right, path=f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-13, abs_tol=1e-15
        ):
            raise ProtocolError(f"release numeric mismatch at {path}")
        return
    if actual != expected:
        raise ProtocolError(f"release value mismatch at {path}")


def verify_public_release(
    contract_path: Path = CONTRACT, output: Path = DEVELOPMENT_OUTPUT
) -> tuple[dict[str, Any], str]:
    progress, contract_hash = collect_public_progress(contract_path, output)
    roi, diagnostics = _rows_from_progress(progress)
    expected_by_distance = {
        float(distance): len({item["frame_id"] for item in progress if float(item["distance_m"]) == float(distance)})
        for distance in (2.5, 3.5, 4.5, 5.5, 6.5)
    }
    common, coverage = _common_case(
        roi, 137, expected_frames_by_distance=expected_by_distance
    )
    references = _reference_values(common)
    condition = _condition_table(common, references, include_reference=False)
    condition = condition[condition["route_active"]].copy()
    condition["active"] = True
    condition_means = condition.groupby("condition_cell")["delta_abs_error"].mean()
    ci = bootstrap_condition_mean_ci(condition_means)
    active_diagnostics = diagnostics[diagnostics["route_active"] == True]  # noqa: E712
    inactive_diagnostics = diagnostics[diagnostics["route_active"] == False]  # noqa: E712
    inactive_exact = bool(
        len(inactive_diagnostics)
        and inactive_diagnostics["exact_identity"].astype(bool).all()
        and (inactive_diagnostics["max_abs_change_dn"].astype(float) == 0.0).all()
    )
    summary = evaluate_gate(
        condition,
        ci=ci,
        coverage=coverage["fraction"],
        locator_coverage=sum(item.get("status") == "ok" for item in progress) / 137,
        minimum_distance_coverage=coverage["minimum_distance_fraction"],
        maximum_preclip_overshoot_fraction=float(active_diagnostics["preclip_overshoot_fraction"].max()),
        maximum_introduced_clipping_fraction=float(active_diagnostics["introduced_clipping_fraction"].max()),
        maximum_abs_change_dn=float(active_diagnostics["max_abs_change_dn"].max()),
        inactive_exact_identity=inactive_exact,
    )
    released_paths = {
        "roi": output / "public_e2e_roi_results.csv",
        "condition": output / "public_e2e_condition_results.csv",
        "diagnostics": output / "public_e2e_pixel_diagnostics.csv",
        "coverage": output / "public_e2e_complete_case.json",
    }
    recomputed_frames = {"roi": roi, "condition": condition, "diagnostics": diagnostics}
    for label, frame in recomputed_frames.items():
        released = pd.read_csv(released_paths[label])
        try:
            pd.testing.assert_frame_equal(
                released.reset_index(drop=True),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-13,
                atol=1e-15,
            )
        except AssertionError as exc:
            raise ProtocolError(f"public released {label} differs from progress reconstruction") from exc
    assert_nested_close(read_json(released_paths["coverage"]), coverage, path="coverage")
    final_path = output / "public_e2e_development_final.json"
    final_hash = require_sidecar(final_path)
    final = read_json(final_path)
    if (
        final.get("schema_version") != PUBLIC_FINAL_SCHEMA
        or final.get("contract_sha256") != contract_hash
    ):
        raise ProtocolError("public final contract binding mismatch")
    assert_nested_close(
        final.get("reference_mtf50_native_cy_px_by_distance"), references, path="references"
    )
    assert_nested_close(final.get("coverage"), coverage, path="final.coverage")
    assert_nested_close(final.get("summary"), summary, path="final.summary")
    if set(final.get("artifacts", {})) != set(released_paths):
        raise ProtocolError("public final artifact set mismatch")
    for label, record in final.get("artifacts", {}).items():
        artifact = resolve_path(str(record["path"]))
        if artifact.resolve() != released_paths[label].resolve() or require_sidecar(artifact) != record.get("sha256"):
            raise ProtocolError(f"public release artifact binding mismatch: {label}")
    return final, final_hash


def freeze_readiness(
    contract_path: Path = CONTRACT,
    selection_path: Path = SELECTION,
    execution_path: Path = EXECUTION_FREEZE,
) -> dict[str, Any]:
    _, contract_hash = load_contract(contract_path)
    final, final_hash = verify_public_release(contract_path, DEVELOPMENT_OUTPUT)
    if final.get("summary", {}).get("gate") != "PASS":
        raise ProtocolError("public native E2E development did not PASS")
    counts = final.get("counts", {})
    if (
        counts.get("frames") != PUBLIC_FRAMES
        or int(counts.get("located_frames", 0)) / PUBLIC_FRAMES < 0.95
        or counts.get("active_condition_cells") != PUBLIC_ACTIVE_CONDITIONS
    ):
        raise ProtocolError("public release counts are not the full predeclared cohort")
    if len(final.get("reference_mtf50_native_cy_px_by_distance", {})) != 5:
        raise ProtocolError("public release does not contain five frozen references")
    inventory_hash = require_sidecar(INVENTORY)
    locator_hash = validate_locator_freeze()
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "state": "V3G_FROZEN_AFTER_PUBLIC_E2E_PASS_BEFORE_UNEXPOSED_PNG_ACCESS",
        "created_utc": utc_now(),
        "router": {
            "active_if": (
                "speed_kmh == 80 and iso in {100,200} and "
                "distance_m in {2.5,3.5,4.5}"
            ),
            "active_route_id": ROUTE_ID,
            "expert_table": expert_table_payload(),
            "otherwise": "raw_exact_bypass",
        },
        "reference_mtf50_native_cy_px_by_distance": final[
            "reference_mtf50_native_cy_px_by_distance"
        ],
        "bindings": {
            "contract_sha256": contract_hash,
            "public_e2e_development_final_sha256": final_hash,
            "holdout_inventory_sha256": inventory_hash,
            "locator_execution_freeze_sha256": locator_hash,
        },
    }
    write_json_atomic(selection_path, selection)
    selection_hash = write_sidecar(selection_path)
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "state": "V3G_READY_SEALED_BEFORE_FIRST_UNEXPOSED_PNG_BYTE_ACCESS",
        "result": "PASS",
        "created_utc": utc_now(),
        "one_shot_policy": (
            "First run atomically claims one attempt. Interrupted execution may resume "
            "only under identical freeze/inventory/code with exact progress validation; "
            "a completed attempt or changed binding can never rerun."
        ),
        "expected": {
            "frames": HOLDOUT_FRAMES,
            "condition_cells": HOLDOUT_CONDITIONS,
            "active_frames": HOLDOUT_ACTIVE_FRAMES,
            "active_condition_cells": HOLDOUT_ACTIVE_CONDITIONS,
            "active_distance_cells": HOLDOUT_ACTIVE_DISTANCES,
            "inactive_condition_cells": HOLDOUT_INACTIVE_CONDITIONS,
            "locator_coverage_minimum": 0.95,
            "common_complete_case_minimum": 0.95,
        },
        "bindings": {
            "runner": binding(Path(__file__)),
            "tests": binding(TEST_CODE),
            "contract": {"path": relative_or_absolute(contract_path), "sha256": contract_hash},
            "selection": {"path": relative_or_absolute(selection_path), "sha256": selection_hash},
            "inventory": {"path": relative_or_absolute(INVENTORY), "sha256": inventory_hash},
            "locator_execution_freeze": {
                "path": relative_or_absolute(LOCATOR_FREEZE),
                "sha256": locator_hash,
            },
        },
    }
    write_json_atomic(execution_path, execution)
    execution_hash = write_sidecar(execution_path)
    return {
        "state": execution["state"],
        "selection_sha256": selection_hash,
        "execution_freeze_sha256": execution_hash,
    }


def load_execution_freeze(path: Path = EXECUTION_FREEZE) -> tuple[dict[str, Any], str]:
    digest = require_sidecar(path)
    payload = read_json(path)
    if (
        payload.get("schema_version") != EXECUTION_SCHEMA
        or payload.get("state") != "V3G_READY_SEALED_BEFORE_FIRST_UNEXPOSED_PNG_BYTE_ACCESS"
        or payload.get("result") != "PASS"
    ):
        raise ProtocolError("holdout execution freeze is not READY/PASS")
    expected = payload.get("expected", {})
    required = {
        "frames": HOLDOUT_FRAMES,
        "condition_cells": HOLDOUT_CONDITIONS,
        "active_frames": HOLDOUT_ACTIVE_FRAMES,
        "active_condition_cells": HOLDOUT_ACTIVE_CONDITIONS,
        "active_distance_cells": HOLDOUT_ACTIVE_DISTANCES,
        "inactive_condition_cells": HOLDOUT_INACTIVE_CONDITIONS,
        "locator_coverage_minimum": 0.95,
        "common_complete_case_minimum": 0.95,
    }
    if any(expected.get(key) != value for key, value in required.items()):
        raise ProtocolError("holdout execution freeze composition/gates changed")
    execution_bindings = payload.get("bindings", {})
    if set(execution_bindings) != {
        "runner",
        "tests",
        "contract",
        "selection",
        "inventory",
        "locator_execution_freeze",
    }:
        raise ProtocolError("holdout execution freeze binding set changed")
    for label, record in execution_bindings.items():
        resolved = resolve_path(str(record["path"]))
        if sha256_file(resolved) != record["sha256"]:
            raise ProtocolError(f"holdout execution binding mismatch: {label}")
    validate_locator_freeze()
    return payload, digest


def claim_one_shot(
    marker: Path,
    execution_freeze_hash: str,
    *,
    inventory_hash: str = "test-inventory",
    runner_hash: str = "test-runner",
    progress_validator: Callable[[], None] | None = None,
) -> str:
    identity = canonical_sha256(
        {
            "execution_freeze_sha256": execution_freeze_hash,
            "inventory_sha256": inventory_hash,
            "runner_sha256": runner_hash,
        }
    )
    payload = {
        "schema_version": "MERIT-C058-SAFETY-ROUTER-ONE-SHOT-ATTEMPT-v3g",
        "state": "STARTED",
        "attempt_identity": identity,
        "execution_freeze_sha256": execution_freeze_hash,
        "inventory_sha256": inventory_hash,
        "runner_sha256": runner_hash,
        "started_utc": utc_now(),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    if (marker.parent / "attempt_complete.json").exists():
        raise ProtocolError("one-shot attempt is already complete")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        existing = read_json(marker)
        if existing.get("attempt_identity") != identity:
            raise ProtocolError("one-shot attempt has already been claimed by different bindings")
        if existing.get("state") == "COMPLETE":
            raise ProtocolError("one-shot attempt is already complete")
        if existing.get("state") != "STARTED" or progress_validator is None:
            raise ProtocolError("one-shot attempt already been claimed; validated resume required")
        progress_validator()
        return "RESUME"
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return "STARTED"


def load_inventory_strict(path: Path = INVENTORY) -> tuple[dict[str, Any], str]:
    digest = require_sidecar(path)
    payload = read_json(path)
    if (
        payload.get("schema_version") != INVENTORY_SCHEMA
        or payload.get("state") != "SEALED_NAME_ONLY_BEFORE_UNEXPOSED_PNG_BYTE_ACCESS"
    ):
        raise ProtocolError("holdout inventory schema/state mismatch")
    records = payload.get("records", [])
    if payload.get("records_sha256") != canonical_sha256(records):
        raise ProtocolError("holdout inventory records self-hash mismatch")
    if len(records) != HOLDOUT_FRAMES or len({row.get("frame_id") for row in records}) != HOLDOUT_FRAMES:
        raise ProtocolError(f"holdout inventory is not exactly {HOLDOUT_FRAMES} unique frames")
    conditions = {str(row.get("condition")) for row in records}
    eligible_conditions = payload.get("eligible_conditions", [])
    if (
        len(conditions) != HOLDOUT_CONDITIONS
        or sorted(conditions) != eligible_conditions
        or payload.get("eligible_conditions_sha256") != canonical_sha256(eligible_conditions)
    ):
        raise ProtocolError(
            f"holdout inventory is not exactly the frozen {HOLDOUT_CONDITIONS} eligible conditions"
        )
    expected_sorted = sorted(
        records,
        key=lambda row: (
            int(row["speed_kmh"]),
            float(row["distance_m"]),
            int(row["iso"]),
            str(row["filename"]),
        ),
    )
    if records != expected_sorted:
        raise ProtocolError("holdout inventory ordering changed")
    for ordinal, row in enumerate(records, start=1):
        speed, distance, iso = parse_condition(str(row["condition"]))
        active = router_active(speed, iso, distance)
        expected_route = ROUTE_ID if active else "raw_exact_bypass"
        if (
            row.get("ordinal") != ordinal
            or row.get("frame_id") != f"{row['condition']}/{row['filename']}"
            or row.get("relative_path") != row.get("frame_id")
            or row.get("speed_kmh") != speed
            or float(row.get("distance_m")) != distance
            or row.get("iso") != iso
            or row.get("router_active") is not active
            or row.get("route_id") != expected_route
        ):
            raise ProtocolError("holdout inventory metadata/router mismatch")
    audit = payload.get("audit", {})
    expected_active = sum(
        router_active(speed, iso, distance)
        for speed, distance, iso in (
            parse_condition(str(row["condition"])) for row in records
        )
    )
    if (
        audit.get("frames") != HOLDOUT_FRAMES
        or audit.get("condition_cells") != HOLDOUT_CONDITIONS
        or audit.get("prebank_parent_disjoint_candidates") != PREBANK_PARENT_DISJOINT_CANDIDATES
        or audit.get("excluded_c006_bank_parent_frame_ids") != BANK_PARENT_EXCLUSIONS
        or audit.get("active_frames") != HOLDOUT_ACTIVE_FRAMES
        or audit.get("active_condition_cells") != HOLDOUT_ACTIVE_CONDITIONS
        or audit.get("active_distance_cells") != HOLDOUT_ACTIVE_DISTANCES
        or audit.get("inactive_condition_cells") != HOLDOUT_INACTIVE_CONDITIONS
        or audit.get("active_frames") != expected_active
        or audit.get("inactive_frames") != HOLDOUT_FRAMES - expected_active
        or audit.get("candidate_png_content_read") is not False
        or audit.get("candidate_png_hashes_computed") is not False
        or audit.get("candidate_png_shapes_decoded") is not False
    ):
        raise ProtocolError("holdout inventory audit fields changed")
    return payload, digest


def load_selection_strict(path: Path = SELECTION) -> tuple[dict[str, Any], str]:
    digest = require_sidecar(path)
    payload = read_json(path)
    if (
        payload.get("schema_version") != SELECTION_SCHEMA
        or payload.get("state") != "V3G_FROZEN_AFTER_PUBLIC_E2E_PASS_BEFORE_UNEXPOSED_PNG_ACCESS"
        or payload.get("router", {}).get("active_if")
        != "speed_kmh == 80 and iso in {100,200} and distance_m in {2.5,3.5,4.5}"
        or payload.get("router", {}).get("active_route_id") != ROUTE_ID
        or payload.get("router", {}).get("expert_table") != expert_table_payload()
        or payload.get("router", {}).get("otherwise") != "raw_exact_bypass"
        or len(payload.get("reference_mtf50_native_cy_px_by_distance", {})) != 5
    ):
        raise ProtocolError("v3g router selection freeze mismatch")
    contract_hash = require_sidecar(CONTRACT)
    inventory_hash = require_sidecar(INVENTORY)
    locator_hash = validate_locator_freeze()
    bindings = payload.get("bindings", {})
    if (
        bindings.get("contract_sha256") != contract_hash
        or bindings.get("holdout_inventory_sha256") != inventory_hash
        or bindings.get("locator_execution_freeze_sha256") != locator_hash
    ):
        raise ProtocolError("v3g selection upstream binding mismatch")
    development_final = DEVELOPMENT_OUTPUT / "public_e2e_development_final.json"
    if bindings.get("public_e2e_development_final_sha256") != require_sidecar(development_final):
        raise ProtocolError("v3g selection development-final binding mismatch")
    return payload, digest


def known_exposed_hashes() -> set[str]:
    cohort = pd.read_csv(EXPOSED_COHORT)
    return set(cohort["image_sha256"].dropna().astype(str).str.lower())


def known_bank_parent_hashes() -> set[str]:
    values: set[str] = set()
    for path in BANK_MANIFESTS:
        table = pd.read_csv(path)
        if "parent_sha256" not in table.columns:
            raise ProtocolError(f"bank parent SHA column missing: {path}")
        values.update(table["parent_sha256"].dropna().astype(str).str.lower())
    return values


def validate_decoded_holdout_image(image: np.ndarray) -> tuple[int, int, str]:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ProtocolError("holdout PNG must decode to 2D uint8 grayscale")
    height, width = int(image.shape[0]), int(image.shape[1])
    if (height, width) != (5460, 8192):
        raise ProtocolError(f"holdout PNG shape changed: {(height, width)}")
    return width, height, str(image.dtype)


def collect_holdout_progress(
    *,
    output: Path,
    inventory: dict[str, Any],
    contract_hash: str,
    c057_hash: str,
    locator_hash: str,
    execution_hash: str,
    inventory_hash: str,
    selection_hash: str,
    require_all: bool,
) -> list[dict[str, Any]]:
    records = inventory["records"]
    progress_directory = output / "progress"
    paths = sorted(progress_directory.glob("*.json")) if progress_directory.exists() else []
    if require_all and len(paths) != HOLDOUT_FRAMES:
        raise ProtocolError(
            f"holdout progress is incomplete: {len(paths)}/{HOLDOUT_FRAMES}"
        )
    if len(paths) > HOLDOUT_FRAMES:
        raise ProtocolError("holdout progress contains extra files")
    progress: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        record = records[index]
        expected_path = _progress_path(output, int(record["ordinal"]), str(record["frame_id"]))
        if path.resolve() != expected_path.resolve():
            raise ProtocolError("holdout progress filename/order mismatch")
        payload = read_json(path)
        validate_progress_payload(
            payload,
            {
                "ordinal": int(record["ordinal"]),
                "frame_id": str(record["frame_id"]),
                "condition_cell": str(record["condition"]),
                "speed_kmh": int(record["speed_kmh"]),
                "distance_m": float(record["distance_m"]),
                "iso": int(record["iso"]),
                "route_active": bool(record["router_active"]),
                "image_sha256": None,
            },
            study=HOLDOUT_STUDY,
            contract_hash=contract_hash,
            c057_hash=c057_hash,
            locator_freeze_hash=locator_hash,
            execution_freeze_hash=execution_hash,
            inventory_hash=inventory_hash,
            selection_hash=selection_hash,
        )
        image_hash = payload.get("image_sha256")
        if not isinstance(image_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            raise ProtocolError("holdout progress PNG hash malformed")
        if payload.get("relative_path") != record["relative_path"]:
            raise ProtocolError("holdout progress path differs from frozen inventory")
        if payload.get("status") == "ok" and (
            payload.get("image_dtype") != "uint8"
            or int(payload.get("image_width", 0)) <= 0
            or int(payload.get("image_height", 0)) <= 0
        ):
            raise ProtocolError("holdout progress decoded image metadata malformed")
        progress.append(payload)
    return progress


def expected_holdout_execution_manifest(
    progress: list[dict[str, Any]],
    *,
    execution_hash: str,
    inventory_hash: str,
    exposed_hashes: set[str],
    bank_hashes: set[str],
) -> dict[str, Any]:
    hashes = [str(item["image_sha256"]) for item in progress]
    return {
        "schema_version": "MERIT-C058-SAFETY-ROUTER-HOLDOUT-EXECUTION-MANIFEST-v3g",
        "state": "INFERENCE_COMPLETE",
        "execution_freeze_sha256": execution_hash,
        "inventory_sha256": inventory_hash,
        "frames": len(progress),
        "condition_cells": len({item["condition_cell"] for item in progress}),
        "sha256_overlap_with_c057_exposed": len(set(hashes) & exposed_hashes),
        "sha256_overlap_with_c006_bank_parents": len(set(hashes) & bank_hashes),
        "duplicate_sha256_within_holdout": len(hashes) - len(set(hashes)),
        "records": [
            {
                "ordinal": item["ordinal"],
                "frame_id": item["frame_id"],
                "condition": item["condition_cell"],
                "relative_path": item["relative_path"],
                "image_sha256": item["image_sha256"],
                "width": item.get("image_width"),
                "height": item.get("image_height"),
                "dtype": item.get("image_dtype"),
                "status": item["status"],
                "reason": item["reason"],
            }
            for item in progress
        ],
    }


def ensure_holdout_execution_manifest(
    output: Path,
    progress: list[dict[str, Any]],
    *,
    execution_hash: str,
    inventory_hash: str,
    exposed_hashes: set[str],
    bank_hashes: set[str],
) -> tuple[dict[str, Any], str]:
    expected = expected_holdout_execution_manifest(
        progress,
        execution_hash=execution_hash,
        inventory_hash=inventory_hash,
        exposed_hashes=exposed_hashes,
        bank_hashes=bank_hashes,
    )
    path = output / "holdout_execution_manifest.json"
    if path.exists() or sidecar_path(path).exists():
        digest = require_sidecar(path)
        if read_json(path) != expected:
            raise ProtocolError("holdout execution manifest differs from exact progress reconstruction")
        return expected, digest
    write_json_atomic(path, expected)
    return expected, write_sidecar(path)


def run_holdout(
    execution_path: Path = EXECUTION_FREEZE,
    output: Path = HOLDOUT_OUTPUT,
) -> None:
    """Execute the sealed 280-frame attempt; safe interruption may resume exactly."""

    # These validations intentionally precede the one-shot claim and every PNG byte read.
    execution, execution_hash = load_execution_freeze(execution_path)
    inventory, inventory_hash = load_inventory_strict(INVENTORY)
    selection, selection_hash = load_selection_strict(SELECTION)
    _, contract_hash = load_contract(CONTRACT)
    if execution["bindings"]["inventory"]["sha256"] != inventory_hash:
        raise ProtocolError("execution freeze does not bind the active inventory")
    if execution["bindings"]["selection"]["sha256"] != selection_hash:
        raise ProtocolError("execution freeze does not bind the active selection")
    locator_hash = validate_locator_freeze()
    module = c057_module()
    protocol, _, _, c057_hash = module.load_verified_environment(module.DEFAULT_FREEZE)

    def validate_resume() -> None:
        collect_holdout_progress(
            output=output,
            inventory=inventory,
            contract_hash=contract_hash,
            c057_hash=c057_hash,
            locator_hash=locator_hash,
            execution_hash=execution_hash,
            inventory_hash=inventory_hash,
            selection_hash=selection_hash,
            require_all=False,
        )

    claim_one_shot(
        output / "attempt_started.json",
        execution_hash,
        inventory_hash=inventory_hash,
        runner_hash=sha256_file(Path(__file__)),
        progress_validator=validate_resume,
    )
    existing = collect_holdout_progress(
        output=output,
        inventory=inventory,
        contract_hash=contract_hash,
        c057_hash=c057_hash,
        locator_hash=locator_hash,
        execution_hash=execution_hash,
        inventory_hash=inventory_hash,
        selection_hash=selection_hash,
        require_all=False,
    )
    exposed_hashes = known_exposed_hashes()
    bank_hashes = known_bank_parent_hashes()
    completed = len(existing)
    if completed == HOLDOUT_FRAMES:
        ensure_holdout_execution_manifest(
            output,
            existing,
            execution_hash=execution_hash,
            inventory_hash=inventory_hash,
            exposed_hashes=exposed_hashes,
            bank_hashes=bank_hashes,
        )
        return

    locator = locator_module()
    device = module.D0.require_cuda()
    models, tiled = module.D0.load_models(protocol, device)
    models = {key: model for key, model in models.items() if key[0] == "on"}
    model_callback = _model_callback(module, protocol, models, tiled, device)
    for record in inventory["records"][completed:]:
        ordinal = int(record["ordinal"])
        path = _progress_path(output, ordinal, str(record["frame_id"]))
        image_path = RAW_ROOT / str(record["relative_path"])
        started = time.perf_counter()
        image_hash = sha256_file(image_path).lower()
        overlap_labels = []
        if image_hash in exposed_hashes:
            overlap_labels.append("C057_EXPOSED")
        if image_hash in bank_hashes:
            overlap_labels.append("C006_BANK_PARENT")
        decoded_width: int | None = None
        decoded_height: int | None = None
        decoded_dtype: str | None = None
        try:
            if overlap_labels:
                raise ProtocolError("PNG_SHA256_OVERLAP:" + ",".join(overlap_labels))
            image = locator.read_gray(image_path)
            decoded_width, decoded_height, decoded_dtype = validate_decoded_holdout_image(image)
            locator_record, events = _evaluate_image(
                image,
                speed=int(record["speed_kmh"]),
                iso=int(record["iso"]),
                distance_m=float(record["distance_m"]),
                locator=locator,
                estimator=module,
                model_callback=model_callback,
            )
            payload = {
                "status": "ok",
                "reason": "ok",
                "image_width": decoded_width,
                "image_height": decoded_height,
                "image_dtype": decoded_dtype,
                "locator": locator_record,
                "events": events,
            }
        except Exception as exc:
            payload = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "image_width": decoded_width,
                "image_height": decoded_height,
                "image_dtype": decoded_dtype,
                "events": [],
            }
        payload.update(
            {
                "schema_version": PROGRESS_SCHEMA,
                "study": HOLDOUT_STUDY,
                "ordinal": ordinal,
                "frame_id": str(record["frame_id"]),
                "condition_cell": str(record["condition"]),
                "relative_path": str(record["relative_path"]),
                "speed_kmh": int(record["speed_kmh"]),
                "distance_m": float(record["distance_m"]),
                "iso": int(record["iso"]),
                "route_active": bool(record["router_active"]),
                "is_reference": False,
                "image_sha256": image_hash,
                "contract_sha256": contract_hash,
                "c057_restoration_freeze_sha256": c057_hash,
                "locator_execution_freeze_sha256": locator_hash,
                "execution_freeze_sha256": execution_hash,
                "inventory_sha256": inventory_hash,
                "selection_sha256": selection_hash,
                "locator_reused_for_all_outputs": True,
                "elapsed_seconds": time.perf_counter() - started,
                "committed_utc": utc_now(),
            }
        )
        write_json_atomic(path, payload)
        if ordinal == completed + 1 or ordinal % 10 == 0 or ordinal == HOLDOUT_FRAMES:
            print(
                json.dumps(
                    {
                        "state": "RUNNING" if ordinal < HOLDOUT_FRAMES else "INFERENCE_COMPLETE",
                        "completed_frames": ordinal,
                        "total_frames": HOLDOUT_FRAMES,
                    }
                ),
                flush=True,
            )

    progress = collect_holdout_progress(
        output=output,
        inventory=inventory,
        contract_hash=contract_hash,
        c057_hash=c057_hash,
        locator_hash=locator_hash,
        execution_hash=execution_hash,
        inventory_hash=inventory_hash,
        selection_hash=selection_hash,
        require_all=True,
    )
    ensure_holdout_execution_manifest(
        output,
        progress,
        execution_hash=execution_hash,
        inventory_hash=inventory_hash,
        exposed_hashes=exposed_hashes,
        bank_hashes=bank_hashes,
    )


def aggregate_holdout(
    execution_path: Path = EXECUTION_FREEZE,
    output: Path = HOLDOUT_OUTPUT,
) -> dict[str, Any]:
    execution, execution_hash = load_execution_freeze(execution_path)
    inventory, inventory_hash = load_inventory_strict(INVENTORY)
    selection, selection_hash = load_selection_strict(SELECTION)
    _, contract_hash = load_contract(CONTRACT)
    locator_hash = validate_locator_freeze()
    module = c057_module()
    _, _, _, c057_hash = module.load_verified_environment(module.DEFAULT_FREEZE)
    progress = collect_holdout_progress(
        output=output,
        inventory=inventory,
        contract_hash=contract_hash,
        c057_hash=c057_hash,
        locator_hash=locator_hash,
        execution_hash=execution_hash,
        inventory_hash=inventory_hash,
        selection_hash=selection_hash,
        require_all=True,
    )
    manifest_path = output / "holdout_execution_manifest.json"
    manifest, manifest_hash = ensure_holdout_execution_manifest(
        output,
        progress,
        execution_hash=execution_hash,
        inventory_hash=inventory_hash,
        exposed_hashes=known_exposed_hashes(),
        bank_hashes=known_bank_parent_hashes(),
    )
    hash_overlap_pass = (
        manifest.get("frames") == HOLDOUT_FRAMES
        and manifest.get("condition_cells") == HOLDOUT_CONDITIONS
        and manifest.get("sha256_overlap_with_c057_exposed") == 0
        and manifest.get("sha256_overlap_with_c006_bank_parents") == 0
        and manifest.get("duplicate_sha256_within_holdout") == 0
    )
    roi, diagnostics = _rows_from_progress(progress)
    expected_by_distance = {
        float(distance): len({item["frame_id"] for item in progress if float(item["distance_m"]) == float(distance)})
        for distance in (2.5, 3.5, 4.5, 5.5, 6.5)
    }
    common, coverage = _common_case(
        roi, HOLDOUT_FRAMES, expected_frames_by_distance=expected_by_distance
    )
    references = {
        str(distance): float(value)
        for distance, value in selection["reference_mtf50_native_cy_px_by_distance"].items()
    }
    condition = _condition_table(common, references, include_reference=False)
    condition["active"] = condition["route_active"].astype(bool)
    condition_cells = condition["condition_cell"].nunique()
    condition_means = condition.groupby("condition_cell")["delta_abs_error"].mean()
    ci = bootstrap_condition_mean_ci(condition_means) if len(condition_means) else (math.inf, math.inf)
    active_condition_means = (
        condition[condition["active"]]
        .groupby("condition_cell")["delta_abs_error"]
        .mean()
    )
    active_ci = (
        bootstrap_condition_mean_ci(active_condition_means)
        if len(active_condition_means)
        else (math.inf, math.inf)
    )
    active_diagnostics = diagnostics[diagnostics["route_active"] == True]  # noqa: E712
    inactive_diagnostics = diagnostics[diagnostics["route_active"] == False]  # noqa: E712
    inactive_exact = bool(
        len(inactive_diagnostics)
        and inactive_diagnostics["exact_identity"].astype(bool).all()
        and (inactive_diagnostics["max_abs_change_dn"].astype(float) == 0.0).all()
    )
    summary = evaluate_gate(
        condition,
        ci=ci,
        active_ci=active_ci,
        coverage=coverage["fraction"],
        locator_coverage=sum(item["status"] == "ok" for item in progress) / HOLDOUT_FRAMES,
        minimum_distance_coverage=coverage["minimum_distance_fraction"],
        maximum_preclip_overshoot_fraction=float(active_diagnostics["preclip_overshoot_fraction"].max()) if len(active_diagnostics) else math.inf,
        maximum_introduced_clipping_fraction=float(active_diagnostics["introduced_clipping_fraction"].max()) if len(active_diagnostics) else math.inf,
        maximum_abs_change_dn=float(active_diagnostics["max_abs_change_dn"].max()) if len(active_diagnostics) else math.inf,
        inactive_exact_identity=inactive_exact,
    )
    active_condition_cells = int(
        condition[condition["active"]]["condition_cell"].nunique()
    )
    inactive_condition_cells = int(
        condition[~condition["active"]]["condition_cell"].nunique()
    )
    active_distance_cells = int(
        condition[condition["active"]]["distance_m"].nunique()
    )
    inactive_exact_ties = bool(
        len(condition[~condition["active"]])
        and (condition.loc[~condition["active"], "delta_abs_error"].astype(float) == 0.0).all()
    )
    all_five_distances_covered = (
        len(coverage["per_distance"]) == 5
        and {float(item["distance_m"]) for item in coverage["per_distance"]}
        == {2.5, 3.5, 4.5, 5.5, 6.5}
        and all(int(item["expected_roi_identities"]) > 0 for item in coverage["per_distance"])
    )
    summary["gate_checks"]["exact_280_frames_29_conditions"] = (
        condition_cells == HOLDOUT_CONDITIONS
    )
    summary["gate_checks"]["active_6_conditions_3_distances"] = (
        active_condition_cells == HOLDOUT_ACTIVE_CONDITIONS
        and active_distance_cells == HOLDOUT_ACTIVE_DISTANCES
    )
    summary["gate_checks"]["inactive_23_conditions_exact_tie"] = (
        inactive_condition_cells == HOLDOUT_INACTIVE_CONDITIONS and inactive_exact_ties
    )
    summary["gate_checks"]["all_five_distances_coverage_reported"] = all_five_distances_covered
    summary["gate_checks"]["zero_prior_sha256_overlap"] = hash_overlap_pass
    summary["gate"] = "PASS" if all(summary["gate_checks"].values()) else "FAIL"
    roi_path = output / "holdout_roi_results.csv"
    condition_path = output / "holdout_condition_results.csv"
    diagnostic_path = output / "holdout_pixel_diagnostics.csv"
    coverage_path = output / "holdout_complete_case.json"
    final_path = output / "holdout_final.json"
    for path in (roi_path, condition_path, diagnostic_path, coverage_path, final_path):
        if path.exists() or sidecar_path(path).exists():
            raise FileExistsError(path)
    roi.to_csv(roi_path, index=False)
    condition.to_csv(condition_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)
    write_json_atomic(coverage_path, coverage)
    for path in (roi_path, condition_path, diagnostic_path, coverage_path):
        write_sidecar(path)
    final = {
        "schema_version": "MERIT-C058-SAFETY-ROUTER-HOLDOUT-RESULT-v3g",
        "state": "ONE_SHOT_HOLDOUT_COMPLETE",
        "completed_utc": utc_now(),
        "evidence_label": (
            f"{EVIDENCE_LABEL}; all {HOLDOUT_FRAMES} frames were pixel-unread "
            "before the v3g execution freeze"
        ),
        "execution_freeze_sha256": execution_hash,
        "inventory_sha256": inventory_hash,
        "selection_sha256": selection_hash,
        "execution_manifest_sha256": manifest_hash,
        "counts": {
            "frames": HOLDOUT_FRAMES,
            "condition_cells": int(condition_cells),
            "active_condition_cells": active_condition_cells,
            "inactive_condition_cells": inactive_condition_cells,
            "active_distance_cells": active_distance_cells,
        },
        "coverage": coverage,
        "summary": summary,
        "artifacts": {
            "roi": binding(roi_path, sidecar=True),
            "condition": binding(condition_path, sidecar=True),
            "diagnostics": binding(diagnostic_path, sidecar=True),
            "coverage": binding(coverage_path, sidecar=True),
        },
    }
    write_json_atomic(final_path, final)
    final_hash = write_sidecar(final_path)
    complete = {
        "schema_version": "MERIT-C058-SAFETY-ROUTER-ONE-SHOT-COMPLETE-v3g",
        "state": "COMPLETE",
        "gate": summary["gate"],
        "execution_freeze_sha256": execution_hash,
        "final_sha256": final_hash,
        "completed_utc": utc_now(),
    }
    write_json_atomic(output / "attempt_complete.json", complete)
    write_sidecar(output / "attempt_complete.json")
    return final


def check_readiness(path: Path = EXECUTION_FREEZE) -> dict[str, Any]:
    payload, digest = load_execution_freeze(path)
    inventory = read_json(INVENTORY)
    checks = {
        "inventory_280_frames": inventory["audit"]["frames"] == HOLDOUT_FRAMES,
        "inventory_29_conditions": inventory["audit"]["condition_cells"] == HOLDOUT_CONDITIONS,
        "inventory_47_active_frames": inventory["audit"]["active_frames"] == HOLDOUT_ACTIVE_FRAMES,
        "inventory_6_active_conditions": inventory["audit"]["active_condition_cells"] == HOLDOUT_ACTIVE_CONDITIONS,
        "inventory_3_active_distances": inventory["audit"]["active_distance_cells"] == HOLDOUT_ACTIVE_DISTANCES,
        "inventory_no_frame_id_overlap": inventory["audit"]["frame_id_overlap_with_exposed"] == 0
        and inventory["audit"]["frame_id_overlap_with_c006_bank_parents"] == 0,
        "inventory_png_bytes_unread": inventory["audit"]["candidate_png_content_read"] is False,
        "locator_freeze_valid": bool(validate_locator_freeze()),
        "attempt_not_started": not (HOLDOUT_OUTPUT / "attempt_started.json").exists(),
    }
    return {
        "state": "READY" if all(checks.values()) else "NO_GO",
        "execution_freeze_sha256": digest,
        "checks": checks,
        "freeze": payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "create-inventory",
            "retrospective",
            "create-public-manifest",
            "create-contract",
            "run-public-development",
            "aggregate-public-development",
            "freeze-readiness",
            "check-readiness",
            "run-holdout",
            "aggregate-holdout",
        ),
    )
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "create-inventory":
        result = create_inventory(arguments.output or INVENTORY)
    elif arguments.command == "retrospective":
        result = retrospective_profile(arguments.output or RETROSPECTIVE_OUTPUT)
    elif arguments.command == "create-public-manifest":
        result = create_public_raw_manifest(arguments.output or PUBLIC_RAW_MANIFEST)
    elif arguments.command == "create-contract":
        result = create_contract(arguments.output or arguments.contract)
    elif arguments.command == "run-public-development":
        run_public_development(arguments.contract, arguments.output or DEVELOPMENT_OUTPUT)
        result = {"state": "INFERENCE_COMPLETE"}
    elif arguments.command == "aggregate-public-development":
        result = aggregate_public_development(
            arguments.contract, arguments.output or DEVELOPMENT_OUTPUT
        )
    elif arguments.command == "freeze-readiness":
        result = freeze_readiness(arguments.contract)
    elif arguments.command == "run-holdout":
        run_holdout(EXECUTION_FREEZE, arguments.output or HOLDOUT_OUTPUT)
        result = {"state": "HOLDOUT_INFERENCE_COMPLETE"}
    elif arguments.command == "aggregate-holdout":
        result = aggregate_holdout(EXECUTION_FREEZE, arguments.output or HOLDOUT_OUTPUT)
    else:
        result = check_readiness()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
