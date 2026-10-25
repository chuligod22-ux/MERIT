"""Corrected C058-v2 optical-fidelity development and confirmation runner.

The development endpoint is fixed to the five 80 km/h ISO100 condition cells;
the five 60 km/h ISO100 cells are distance-specific RAW references and exact
bypass safety controls.  Candidate ranking always averages all three frozen
task-ON seeds, so no seed may be selected from outcomes.

Every candidate is compared on one common complete-case ROI set.  After a
development PASS, ``freeze-selection`` independently recomputes the release
from hashed artifacts and seals one route.  Only then may
``confirm-existing-40`` run the remaining ISO200/400/800/1600 cells.  Those
cells are explicitly labelled adapter-level held-out conditions because their
C057 model outputs were already observed; they are not independent data.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd

from c058_adapter import AdapterCandidate, apply_adapter, build_candidate_grid


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
C057 = BASE.parent / "c057_mtf50_estimator_v2"
DEFAULT_CONTRACT = BASE / "protocol" / "development_contract_v2.json"
DEFAULT_DEVELOPMENT_OUTPUT = BASE / "artifacts" / "development_v2"
DEFAULT_SELECTION_FREEZE = BASE / "protocol" / "selected_route_v2.json"
DEFAULT_CONFIRMATION_OUTPUT = BASE / "artifacts" / "confirmation_existing40_v2"
CONTRACT_SCHEMA = "MERIT-C058-OPTICAL-FIDELITY-DEVELOPMENT-v2"
PROGRESS_SCHEMA = "MERIT-C058-OPTICAL-FIDELITY-PROGRESS-v2"
SELECTION_SCHEMA = "MERIT-C058-OPTICAL-FIDELITY-SELECTION-v2"
CONFIRMATION_LABEL = (
    "adapter-level held-out conditions, not independent data; "
    "C057 model outputs were previously observed"
)


class ProtocolError(RuntimeError):
    """Raised when a frozen input or fail-closed protocol rule changes."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    serialised = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def write_hash_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = _sidecar_path(path)
    if sidecar.exists():
        raise FileExistsError(sidecar)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def require_sidecar(path: Path) -> str:
    sidecar = _sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise ProtocolError(f"Missing frozen file or sidecar: {path}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    actual = sha256_file(path)
    if len(fields) != 2 or fields[0].lower() != actual or fields[1] != path.name:
        raise ProtocolError(f"SHA-256 sidecar mismatch: {path}")
    return actual


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _binding(path: Path) -> dict[str, str]:
    return {"path": relative_path(path), "sha256": sha256_file(path)}


def create_contract(path: Path) -> dict[str, Any]:
    """Freeze code, upstream inputs, finite grid, gates, and tie-break rules."""

    if path.exists() or _sidecar_path(path).exists():
        raise FileExistsError(path)
    grid = [candidate.to_dict() for candidate in build_candidate_grid()]
    bindings = {
        "adapter_code": _binding(BASE / "scripts" / "c058_adapter.py"),
        "development_v2_code": _binding(Path(__file__)),
        "adapter_tests": _binding(BASE / "tests" / "test_c058_adapter.py"),
        "development_v2_tests": _binding(
            BASE / "tests" / "test_c058_development_v2.py"
        ),
        "c057_runner": _binding(C057 / "run_restoration_mtf50_v2.py"),
        "c057_estimator": _binding(C057 / "mtf50_estimator.py"),
        "c057_restoration_freeze": _binding(
            C057 / "protocol" / "mtf50_v2_restoration_execution_freeze.json"
        ),
        "c057_restoration_freeze_sidecar": _binding(
            C057 / "protocol" / "mtf50_v2_restoration_execution_freeze.json.sha256"
        ),
    }
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "state": "V2_FROZEN_BEFORE_C058_V2_MODEL_OUTPUTS",
        "created_utc": utc_now(),
        "purpose": (
            "Develop a bounded adapter using ISO100 only, with 80 km/h cells "
            "as the efficacy endpoint and 60 km/h cells as exact-bypass references."
        ),
        "candidate_grid": grid,
        "candidate_grid_sha256": canonical_sha256(grid),
        "expected_development": {
            "condition_cells": 10,
            "reference_condition_cells": 5,
            "endpoint_condition_cells": 5,
            "frames": 92,
            "roi_identities": 335,
            "task_on_seeds": [0, 1, 2],
        },
        "development_endpoint": {
            "conditions": "80km_{2.5,3.5,4.5,5.5,6.5}m_ISO100",
            "distance_specific_reference": "60km_same-distance_ISO100 RAW",
            "metric": "restored-minus-RAW absolute MTF50 error in cycles/pixel",
            "improvement_direction": "negative",
            "aggregation": "ROI mean per frame, frame median per condition, then three-seed mean",
        },
        "safety": {
            "minimum_overall_common_complete_case_fraction": 0.95,
            "minimum_each_distance_common_complete_case_fraction": 0.90,
            "maximum_introduced_clipping_fraction": 0.005,
            "maximum_preclip_overshoot_fraction": 0.01,
            "maximum_abs_change_dn": 12.0,
            "reference_exact_bypass_required": True,
        },
        "success_gate": {
            "mean_delta_lt_zero": True,
            "condition_cluster_bootstrap_95pct_ci_upper_lt_zero": True,
            "minimum_improved_endpoint_conditions": 4,
            "all_three_seed_means_nonpositive": True,
            "merit_residual_dependency_alpha_nonzero": True,
            "incremental_contribution_for_unsharp_combinations": (
                "candidate-minus-matching-raw-only-unsharp mean < 0 and paired "
                "condition-bootstrap 95% CI upper < 0; pure residual routes are N/A PASS"
            ),
        },
        "bootstrap": {"count": 10000, "seed": 20261058, "cluster": "condition"},
        "selection": {
            "seed_outcome_selection": "prohibited",
            "eligibility": "technical PASS and alpha != 0",
            "fixed_tie_break": [
                "mean delta ascending",
                "introduced clipping ascending",
                "maximum absolute change DN ascending",
                "route_id lexicographic ascending",
            ],
            "fallback": (
                "If no MERIT-dependent route passes, preserve the best technical "
                "route for diagnosis but set overall gate to FAIL."
            ),
        },
        "confirmation_existing_40": {
            "allowed_only_after_development_pass_and_selection_freeze": True,
            "condition_cells": 40,
            "iso_values": [200, 400, 800, 1600],
            "evidence_label": CONFIRMATION_LABEL,
            "minimum_improved_conditions": 32,
            "same_mean_ci_seed_safety_gates": True,
        },
        "bindings": bindings,
        "prohibitions": [
            "No selection of one favorable seed.",
            "No confirmation run after a development FAIL.",
            "Do not describe the existing 40-condition confirmation as independent data.",
            "Do not modify C006 or C057 code, checkpoints, cohorts, or outputs.",
            "Do not reuse or modify C058-v1 progress; v2 writes only development_v2.",
        ],
    }
    write_json_atomic(path, payload)
    digest = write_hash_sidecar(path)
    return {"contract": str(path), "sha256": digest, "candidate_count": len(grid)}


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    digest = require_sidecar(path)
    payload = read_json(path)
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ProtocolError("Unexpected C058 development contract schema")
    if payload.get("state") != "V2_FROZEN_BEFORE_C058_V2_MODEL_OUTPUTS":
        raise ProtocolError("C058 development contract state changed")
    grid = payload.get("candidate_grid", [])
    if canonical_sha256(grid) != payload.get("candidate_grid_sha256"):
        raise ProtocolError("Contract candidate-grid self-hash mismatch")
    code_grid = [candidate.to_dict() for candidate in build_candidate_grid()]
    if grid != code_grid:
        raise ProtocolError("Runtime candidate grid differs from frozen contract")
    for label, record in payload.get("bindings", {}).items():
        bound = resolve_path(str(record["path"]))
        if not bound.is_file() or sha256_file(bound) != str(record["sha256"]).lower():
            raise ProtocolError(f"Frozen binding mismatch: {label}")
    return payload, digest


def _load_c057_module():
    path = C057 / "run_restoration_mtf50_v2.py"
    spec = importlib.util.spec_from_file_location("c058_frozen_c057", path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"Could not import C057: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_c057_environment():
    module = _load_c057_module()
    protocol, original_freeze, cohort, freeze_hash = module.load_verified_environment(
        module.DEFAULT_FREEZE
    )
    return module, protocol, original_freeze, cohort, freeze_hash


def bootstrap_condition_mean_ci(
    values: np.ndarray, *, seed: int, count: int = 10000
) -> list[float]:
    """Percentile CI resampling condition cells as the independent clusters."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [math.nan, math.nan]
    if count <= 0:
        raise ValueError("bootstrap count must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(count, finite.size))
    sampled = finite[indices].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


ROI_IDENTITY = [
    "frame_id",
    "condition_cell",
    "speed_kmh",
    "distance_m",
    "iso",
    "roi_id",
    "orientation",
]


def build_common_complete_case(
    roi: pd.DataFrame,
    expected_run_keys: list[str],
    *,
    minimum_overall_fraction: float,
    minimum_each_distance_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict every run to one identical finite ROI identity set."""

    required = set(ROI_IDENTITY) | {"run_key", "mtf50", "valid"}
    missing = required - set(roi.columns)
    if missing:
        raise ValueError(f"ROI table missing complete-case columns: {sorted(missing)}")
    if len(set(expected_run_keys)) != len(expected_run_keys):
        raise ValueError("Expected run keys must be unique")
    observed = set(str(value) for value in roi["run_key"].unique())
    if observed != set(expected_run_keys):
        raise ProtocolError(
            f"Run-key coverage changed: missing={sorted(set(expected_run_keys) - observed)} "
            f"extra={sorted(observed - set(expected_run_keys))}"
        )
    if roi.duplicated(ROI_IDENTITY + ["run_key"]).any():
        raise ProtocolError("Duplicate ROI identity/run rows prevent complete-case locking")
    working = roi.copy()
    working["usable_mtf50"] = working["mtf50"].where(working["valid"].astype(bool))
    wide = working.pivot(index=ROI_IDENTITY, columns="run_key", values="usable_mtf50")
    missing_columns = set(expected_run_keys) - set(wide.columns)
    if missing_columns:
        raise ProtocolError(f"Complete-case pivot missing runs: {sorted(missing_columns)}")
    wide = wide.reset_index()
    complete_identity = wide.dropna(subset=expected_run_keys)[ROI_IDENTITY].copy()
    total = int(len(wide))
    complete_count = int(len(complete_identity))
    per_distance: list[dict[str, Any]] = []
    for distance in sorted(float(value) for value in wide["distance_m"].unique()):
        distance_total = int(np.sum(wide["distance_m"].astype(float) == distance))
        distance_complete = int(
            np.sum(complete_identity["distance_m"].astype(float) == distance)
        )
        per_distance.append(
            {
                "distance_m": distance,
                "roi_identities": distance_total,
                "common_complete_case_roi_identities": distance_complete,
                "common_complete_case_fraction": distance_complete / distance_total,
            }
        )
    overall = complete_count / total if total else 0.0
    minimum_distance = min(
        (row["common_complete_case_fraction"] for row in per_distance), default=0.0
    )
    checks = {
        "overall_common_complete_case_fraction": overall
        >= minimum_overall_fraction,
        "each_distance_common_complete_case_fraction": minimum_distance
        >= minimum_each_distance_fraction,
    }
    locked = working.merge(complete_identity, on=ROI_IDENTITY, validate="many_to_one")
    expected_rows = complete_count * len(expected_run_keys)
    if len(locked) != expected_rows or locked["mtf50"].isna().any():
        raise ProtocolError("Common complete-case row coverage is inconsistent")
    report = {
        "all_roi_identities": total,
        "common_complete_case_roi_identities": complete_count,
        "common_complete_case_fraction": float(overall),
        "minimum_distance_common_complete_case_fraction": float(minimum_distance),
        "per_distance": per_distance,
        "run_keys": list(expected_run_keys),
        "gate_checks": checks,
        "gate": "PASS" if all(checks.values()) else "FAIL",
    }
    return locked.drop(columns=["usable_mtf50"]), report


def summarize_candidate_metrics(
    conditions: pd.DataFrame,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    safety: dict[str, float],
    expected_conditions: int = 5,
    minimum_improved_conditions: int = 4,
    enforce_incremental_contribution: bool = True,
) -> list[dict[str, Any]]:
    """Summarise fixed-route results without selecting a seed by outcome."""

    required = {
        "route_id",
        "alpha",
        "sigma",
        "sharp_gain",
        "cap_dn",
        "seed",
        "condition_cell",
        "abs_error",
        "raw_abs_error",
        "common_complete_case_overall_fraction",
        "common_complete_case_min_distance_fraction",
        "reference_exact",
        "introduced_clipping_fraction",
        "preclip_overshoot_fraction",
        "max_abs_change_dn",
    }
    missing = required - set(conditions.columns)
    if missing:
        raise ValueError(f"Condition table missing columns: {sorted(missing)}")
    summaries: list[dict[str, Any]] = []
    for route_id, route in conditions.groupby("route_id", sort=True):
        seed_ids = sorted(int(seed) for seed in route["seed"].unique())
        cell_ids = sorted(str(cell) for cell in route["condition_cell"].unique())
        if seed_ids != [0, 1, 2] or len(cell_ids) != expected_conditions:
            if expected_conditions == 5:
                raise ValueError(
                    f"{route_id} must contain three seeds and exactly five 80 km/h ISO100 conditions"
                )
            raise ValueError(
                f"{route_id} must contain three seeds and exactly {expected_conditions} conditions"
            )
        if len(route) != expected_conditions * 3:
            raise ValueError(f"{route_id} condition/seed rows are incomplete or duplicated")
        route = route.copy()
        route["delta"] = route["abs_error"] - route["raw_abs_error"]
        seed_means = {
            str(int(seed)): float(group["delta"].mean())
            for seed, group in route.groupby("seed", sort=True)
        }
        averaged = route.groupby("condition_cell", as_index=False).agg(
            delta=("delta", "mean")
        )
        deltas = averaged["delta"].to_numpy(float)
        ci = bootstrap_condition_mean_ci(
            deltas, seed=bootstrap_seed, count=bootstrap_count
        )
        first = route.iloc[0]
        alpha = float(first["alpha"])
        metrics = {
            "mean_restored_minus_raw_abs_error_cy_px": float(np.mean(deltas)),
            "condition_cluster_bootstrap_95pct_ci": ci,
            "conditions_improved": int(np.sum(deltas < 0.0)),
            "conditions_tied": int(np.sum(deltas == 0.0)),
            "conditions_worsened": int(np.sum(deltas > 0.0)),
            "seed_mean_deltas_cy_px": seed_means,
            "common_complete_case_overall_fraction": float(
                route["common_complete_case_overall_fraction"].min()
            ),
            "common_complete_case_min_distance_fraction": float(
                route["common_complete_case_min_distance_fraction"].min()
            ),
            "maximum_introduced_clipping_fraction": float(
                route["introduced_clipping_fraction"].max()
            ),
            "maximum_preclip_overshoot_fraction": float(
                route["preclip_overshoot_fraction"].max()
            ),
            "maximum_abs_change_dn": float(route["max_abs_change_dn"].max()),
        }
        checks = {
            "mean_delta_lt_zero": metrics[
                "mean_restored_minus_raw_abs_error_cy_px"
            ]
            < 0.0,
            "ci_upper_lt_zero": ci[1] < 0.0,
            "minimum_condition_improvements": metrics["conditions_improved"]
            >= minimum_improved_conditions,
            "all_three_seed_means_nonpositive": all(
                value <= 0.0 for value in seed_means.values()
            ),
            "overall_common_complete_case_coverage": metrics[
                "common_complete_case_overall_fraction"
            ]
            >= float(safety["minimum_overall_common_complete_case_fraction"]),
            "each_distance_common_complete_case_coverage": metrics[
                "common_complete_case_min_distance_fraction"
            ]
            >= float(safety["minimum_each_distance_common_complete_case_fraction"]),
            "reference_exact_bypass": bool(route["reference_exact"].all()),
            "introduced_clipping": metrics[
                "maximum_introduced_clipping_fraction"
            ]
            <= float(safety["maximum_introduced_clipping_fraction"]),
            "preclip_overshoot": metrics["maximum_preclip_overshoot_fraction"]
            <= float(safety["maximum_preclip_overshoot_fraction"]),
            "maximum_abs_change_dn": metrics["maximum_abs_change_dn"]
            <= float(safety["maximum_abs_change_dn"]),
            "merit_residual_dependency": alpha != 0.0,
        }
        summaries.append(
            {
                "route_id": str(route_id),
                "alpha": alpha,
                "sigma": float(first["sigma"]),
                "sharp_gain": float(first["sharp_gain"]),
                "cap_dn": float(first["cap_dn"]),
                "condition_cells": len(cell_ids),
                "seeds": len(seed_ids),
                **metrics,
                "all_seed_means_nonpositive": checks[
                    "all_three_seed_means_nonpositive"
                ],
                "gate_checks": checks,
            }
        )
    condition_by_route = {
        str(route_id): group.copy()
        for route_id, group in conditions.groupby("route_id", sort=True)
    }
    for summary in summaries:
        incremental = {
            "applicable": False,
            "matching_raw_only_route_id": None,
            "mean_candidate_minus_raw_only_abs_error_cy_px": None,
            "paired_condition_bootstrap_95pct_ci": None,
            "pass": True,
            "reason": "N/A_PURE_RESIDUAL_OR_NON_MERIT_ROUTE",
        }
        is_unsharp_combination = (
            float(summary["alpha"]) != 0.0
            and float(summary["sharp_gain"]) > 0.0
        )
        if not enforce_incremental_contribution:
            incremental["reason"] = "N/A_ALREADY_GATED_IN_FROZEN_DEVELOPMENT"
        elif is_unsharp_combination:
            incremental["applicable"] = True
            matches = [
                row
                for row in summaries
                if float(row["alpha"]) == 0.0
                and float(row["sharp_gain"]) == float(summary["sharp_gain"])
            ]
            if len(matches) != 1:
                incremental.update(
                    {"pass": False, "reason": "MATCHING_RAW_ONLY_UNSHARP_NOT_UNIQUE"}
                )
            else:
                matching = matches[0]
                incremental["matching_raw_only_route_id"] = matching["route_id"]
                candidate_cells = (
                    condition_by_route[summary["route_id"]]
                    .groupby("condition_cell", as_index=False)
                    .agg(candidate_abs_error=("abs_error", "mean"))
                )
                raw_only_cells = (
                    condition_by_route[matching["route_id"]]
                    .groupby("condition_cell", as_index=False)
                    .agg(raw_only_abs_error=("abs_error", "mean"))
                )
                paired = candidate_cells.merge(
                    raw_only_cells, on="condition_cell", validate="one_to_one"
                )
                if len(paired) != expected_conditions:
                    incremental.update(
                        {"pass": False, "reason": "INCOMPLETE_PAIRED_CONDITIONS"}
                    )
                else:
                    values = (
                        paired["candidate_abs_error"]
                        - paired["raw_only_abs_error"]
                    ).to_numpy(float)
                    stable_offset = int(
                        hashlib.sha256(summary["route_id"].encode("utf-8")).hexdigest()[:8],
                        16,
                    )
                    ci = bootstrap_condition_mean_ci(
                        values,
                        seed=bootstrap_seed + stable_offset,
                        count=bootstrap_count,
                    )
                    mean_value = float(np.mean(values))
                    passed = mean_value < 0.0 and ci[1] < 0.0
                    incremental.update(
                        {
                            "mean_candidate_minus_raw_only_abs_error_cy_px": mean_value,
                            "paired_condition_bootstrap_95pct_ci": ci,
                            "pass": passed,
                            "reason": "PASS" if passed else "NO_INCREMENTAL_CONTRIBUTION",
                        }
                    )
        summary["incremental_contribution"] = incremental
        summary["gate_checks"]["incremental_contribution"] = bool(
            incremental["pass"]
        )
        technical_keys = [
            key
            for key in summary["gate_checks"]
            if key != "merit_residual_dependency"
        ]
        technical_pass = all(summary["gate_checks"][key] for key in technical_keys)
        summary["technical_gate"] = "PASS" if technical_pass else "FAIL"
        summary["gate"] = (
            "PASS"
            if technical_pass
            and summary["gate_checks"]["merit_residual_dependency"]
            else "FAIL"
        )
    return summaries


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row["mean_restored_minus_raw_abs_error_cy_px"]),
        float(row["maximum_introduced_clipping_fraction"]),
        float(row["maximum_abs_change_dn"]),
        str(row["route_id"]),
    )


def rank_and_select_candidates(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    technical = [row for row in summaries if row["technical_gate"] == "PASS"]
    dependent = [
        row for row in technical if row["gate_checks"]["merit_residual_dependency"]
    ]
    if dependent:
        selected = sorted(dependent, key=_ranking_key)[0]
        return {
            "gate": "PASS",
            "selected_route_id": selected["route_id"],
            "merit_dependent_selection": True,
            "failure_reason": None,
            "fixed_tie_break_key": list(_ranking_key(selected)),
        }
    if technical:
        selected = sorted(technical, key=_ranking_key)[0]
        return {
            "gate": "FAIL",
            "selected_route_id": selected["route_id"],
            "merit_dependent_selection": False,
            "failure_reason": "NO_MERIT_DEPENDENT_CANDIDATE_PASSED",
            "fixed_tie_break_key": list(_ranking_key(selected)),
        }
    return {
        "gate": "FAIL",
        "selected_route_id": None,
        "merit_dependent_selection": False,
        "failure_reason": "NO_CANDIDATE_PASSED_TECHNICAL_GATES",
        "fixed_tie_break_key": None,
    }


def _progress_path(output: Path, index: int, frame_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", frame_id)
    return output / "progress" / f"{index:04d}_{safe}.json"


def _candidate_map(contract: dict[str, Any]) -> dict[str, AdapterCandidate]:
    return {
        candidate.route_id: candidate
        for candidate in (
            AdapterCandidate.from_dict(item) for item in contract["candidate_grid"]
        )
    }


def _development_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    selected = cohort[cohort["iso"].astype(int) == 100].copy()
    if (
        selected["frame_id"].nunique() != 92
        or len(selected) != 335
        or selected["condition"].nunique() != 10
    ):
        raise ProtocolError("Frozen ISO100 development cohort counts changed")
    return selected


def _confirmation_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    selected = cohort[cohort["iso"].astype(int).isin([200, 400, 800, 1600])].copy()
    if selected["condition"].nunique() != 40:
        raise ProtocolError("Existing 40-condition adapter confirmation cohort changed")
    return selected


def _run_on_crop(module, protocol, model, tiled_restore, crop, device) -> np.ndarray:
    tiling = protocol["restoration"]["tiling"]
    return tiled_restore(
        crop,
        model,
        device,
        tile_size=int(tiling["tile_size"]),
        overlap=int(tiling["overlap"]),
        batch_size=int(tiling["batch_size"]),
        blend_floor=float(tiling["positive_hann_floor"]),
        amp_dtype=protocol["restoration"]["amp_dtype"],
    )


def _raw_event(module, image: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "kind": "raw",
        "route_id": "raw",
        "seed": "none",
        "rois": module.measure_rois(image, frame),
    }


def _adapter_event(
    module,
    raw_crop: np.ndarray,
    task_on: np.ndarray,
    frame: pd.DataFrame,
    candidate: AdapterCandidate,
    seed: int,
    x0: int,
    y0: int,
    *,
    reference_bypass: bool,
) -> dict[str, Any]:
    output, diagnostics = apply_adapter(
        raw_crop, task_on, candidate, reference_bypass=reference_bypass
    )
    if reference_bypass and not np.array_equal(output, raw_crop):
        raise ProtocolError("60 km/h ISO100 reference bypass is not bit-exact")
    return {
        "kind": "adapter",
        "route_id": candidate.route_id,
        "seed": str(seed),
        "candidate": candidate.to_dict(),
        "diagnostics": diagnostics.to_dict(),
        "rois": module.measure_rois(output, frame, x_offset=x0, y_offset=y0),
    }


def _evaluate_frame(
    module,
    protocol: dict[str, Any],
    models,
    tiled_restore,
    device,
    image: np.ndarray,
    frame: pd.DataFrame,
    candidates: Iterable[AdapterCandidate],
    *,
    force_reference_bypass: bool,
) -> list[dict[str, Any]]:
    events = [_raw_event(module, image, frame)]
    config = protocol["restoration"]
    crop, x0, y0 = module.D0.union_context(
        image, frame, int(config["context_margin_px"])
    )
    candidates = list(candidates)
    if force_reference_bypass:
        # The bypass cannot depend on a model output, so avoid unnecessary GPU work.
        for seed in (0, 1, 2):
            for candidate in candidates:
                events.append(
                    _adapter_event(
                        module,
                        crop,
                        crop,
                        frame,
                        candidate,
                        seed,
                        x0,
                        y0,
                        reference_bypass=True,
                    )
                )
        return events
    for seed in (0, 1, 2):
        task_on = _run_on_crop(
            module,
            protocol,
            models[("on", str(seed))],
            tiled_restore,
            crop,
            device,
        )
        for candidate in candidates:
            events.append(
                _adapter_event(
                    module,
                    crop,
                    task_on,
                    frame,
                    candidate,
                    seed,
                    x0,
                    y0,
                    reference_bypass=False,
                )
            )
    return events


def _validate_progress(
    path: Path,
    frame: pd.DataFrame,
    *,
    contract_hash: str,
    candidates: list[AdapterCandidate],
    c057_freeze_hash: str,
    study: str,
    selection_hash: str | None = None,
) -> None:
    payload = read_json(path)
    if payload.get("schema_version") != PROGRESS_SCHEMA:
        raise ProtocolError(f"Progress schema mismatch: {path}")
    if payload.get("study") != study or payload.get("contract_sha256") != contract_hash:
        raise ProtocolError(f"Progress protocol binding mismatch: {path}")
    if payload.get("selection_freeze_sha256") != selection_hash:
        raise ProtocolError(f"Progress selection-freeze binding mismatch: {path}")
    if payload.get("c057_restoration_freeze_sha256") != c057_freeze_hash:
        raise ProtocolError(f"Progress C057-freeze binding mismatch: {path}")
    if payload.get("frame_id") != str(frame.iloc[0]["frame_id"]):
        raise ProtocolError(f"Progress frame mismatch: {path}")
    if payload.get("image_sha256") != str(frame.iloc[0]["image_sha256"]):
        raise ProtocolError(f"Progress image hash mismatch: {path}")
    route_ids = [candidate.route_id for candidate in candidates]
    candidate_specs = {candidate.route_id: candidate.to_dict() for candidate in candidates}
    expected = ["raw_none"] + [
        f"{route_id}_{seed}" for seed in (0, 1, 2) for route_id in route_ids
    ]
    observed = [
        f"{event['route_id']}_{event['seed']}" for event in payload.get("events", [])
    ]
    if observed != expected:
        raise ProtocolError(f"Progress route/seed coverage mismatch: {path}")
    roi_count = len(frame)
    if any(len(event.get("rois", [])) != roi_count for event in payload["events"]):
        raise ProtocolError(f"Progress ROI coverage mismatch: {path}")
    expected_identity = [
        (int(row.roi_id), str(row.orientation))
        for row in frame.sort_values("roi_id").itertuples(index=False)
    ]
    fraction_fields = (
        "residual_cap_fraction",
        "preclip_overshoot_fraction",
        "preclip_below_zero_fraction",
        "preclip_above_one_fraction",
        "introduced_clipping_fraction",
        "introduced_low_clipping_fraction",
        "introduced_high_clipping_fraction",
    )
    nonnegative_fields = (
        "uncapped_residual_mean_abs_dn",
        "capped_residual_mean_abs_dn",
        "mean_abs_change_dn",
        "max_abs_change_dn",
    )
    for event_index, event in enumerate(payload["events"]):
        if event_index == 0:
            if event.get("kind") != "raw" or event.get("route_id") != "raw":
                raise ProtocolError(f"Progress RAW event malformed: {path}")
        else:
            route_id = str(event["route_id"])
            if event.get("kind") != "adapter" or event.get("candidate") != candidate_specs.get(
                route_id
            ):
                raise ProtocolError(f"Progress candidate spec mismatch: {path}")
            diagnostics = event.get("diagnostics", {})
            if not isinstance(diagnostics.get("exact_identity"), bool) or not isinstance(
                diagnostics.get("reference_bypass"), bool
            ):
                raise ProtocolError(f"Progress diagnostic booleans malformed: {path}")
            expected_bypass = (
                study == "development_iso100"
                and int(frame.iloc[0]["speed_kmh"]) == 60
                and int(frame.iloc[0]["iso"]) == 100
            )
            if bool(diagnostics["reference_bypass"]) != expected_bypass or (
                expected_bypass and not bool(diagnostics["exact_identity"])
            ):
                raise ProtocolError(f"Progress reference-bypass diagnostic mismatch: {path}")
            for field in fraction_fields:
                value = diagnostics.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(
                    float(value)
                ) or not 0.0 <= float(value) <= 1.0:
                    raise ProtocolError(f"Progress diagnostic fraction invalid ({field}): {path}")
            for field in nonnegative_fields:
                value = diagnostics.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(
                    float(value)
                ) or float(value) < 0.0:
                    raise ProtocolError(f"Progress diagnostic value invalid ({field}): {path}")
            if diagnostics["max_abs_change_dn"] + 1e-12 < diagnostics["mean_abs_change_dn"]:
                raise ProtocolError(f"Progress change diagnostics inconsistent: {path}")
            if not math.isclose(
                diagnostics["preclip_overshoot_fraction"],
                diagnostics["preclip_below_zero_fraction"]
                + diagnostics["preclip_above_one_fraction"],
                abs_tol=1e-12,
            ):
                raise ProtocolError(f"Progress overshoot diagnostics inconsistent: {path}")
            if not math.isclose(
                diagnostics["introduced_clipping_fraction"],
                diagnostics["introduced_low_clipping_fraction"]
                + diagnostics["introduced_high_clipping_fraction"],
                abs_tol=1e-12,
            ):
                raise ProtocolError(f"Progress clipping diagnostics inconsistent: {path}")
        observed_identity = [
            (int(result["roi_id"]), str(result["orientation"]))
            for result in event["rois"]
        ]
        if observed_identity != expected_identity:
            raise ProtocolError(f"Progress ROI identity/orientation mismatch: {path}")
        for result in event["rois"]:
            if not isinstance(result.get("valid"), bool):
                raise ProtocolError(f"Progress ROI valid flag malformed: {path}")
            value = result.get("mtf50")
            if result["valid"]:
                if not isinstance(value, (int, float)) or not math.isfinite(
                    float(value)
                ) or not 0.0 < float(value) <= 0.6:
                    raise ProtocolError(f"Progress valid ROI MTF50 malformed: {path}")
            elif value is not None:
                raise ProtocolError(f"Progress invalid ROI must have null MTF50: {path}")


def _run_frames(
    *,
    contract_path: Path,
    output: Path,
    study: str,
    cohort_selector,
    candidates: list[AdapterCandidate],
    selection_hash: str | None = None,
) -> None:
    contract, contract_hash = load_contract(contract_path)
    module, protocol, _, cohort, c057_hash = load_c057_environment()
    cohort = cohort_selector(cohort)
    frames = module.D0.frame_records(cohort)
    completed = 0
    for index, (_, frame) in enumerate(frames, start=1):
        path = _progress_path(output, index, str(frame.iloc[0]["frame_id"]))
        if path.exists():
            _validate_progress(
                path,
                frame,
                contract_hash=contract_hash,
                candidates=candidates,
                c057_freeze_hash=c057_hash,
                study=study,
                selection_hash=selection_hash,
            )
            completed += 1
    if completed == len(frames):
        print(json.dumps({"state": "INFERENCE_COMPLETE", "frames": completed}))
        return
    device = module.D0.require_cuda()
    models, tiled_restore = module.D0.load_models(protocol, device)
    started = time.perf_counter()
    processed = 0
    for index, (_, frame) in enumerate(frames, start=1):
        frame_id = str(frame.iloc[0]["frame_id"])
        path = _progress_path(output, index, frame_id)
        if path.exists():
            continue
        frame_started = time.perf_counter()
        image = module.D0.verify_row_assets(frame)
        reference_bypass = (
            study == "development_iso100"
            and int(frame.iloc[0]["speed_kmh"]) == 60
            and int(frame.iloc[0]["iso"]) == 100
        )
        events = _evaluate_frame(
            module,
            protocol,
            models,
            tiled_restore,
            device,
            image,
            frame,
            candidates,
            force_reference_bypass=reference_bypass,
        )
        payload = {
            "schema_version": PROGRESS_SCHEMA,
            "study": study,
            "evidence_label": (
                "ISO100 adapter development"
                if study == "development_iso100"
                else CONFIRMATION_LABEL
            ),
            "frame_id": frame_id,
            "condition_cell": str(frame.iloc[0]["condition"]),
            "image_sha256": str(frame.iloc[0]["image_sha256"]),
            "contract_sha256": contract_hash,
            "selection_freeze_sha256": selection_hash,
            "c057_restoration_freeze_sha256": c057_hash,
            "events": events,
            "elapsed_seconds": time.perf_counter() - frame_started,
            "committed_utc": utc_now(),
        }
        write_json_atomic(path, payload)
        processed += 1
        done = completed + processed
        if processed == 1 or processed % 10 == 0 or done == len(frames):
            elapsed = time.perf_counter() - started
            status = {
                "state": "RUNNING" if done < len(frames) else "INFERENCE_COMPLETE",
                "study": study,
                "completed_frames": done,
                "total_frames": len(frames),
                "remaining_frames": len(frames) - done,
                "last_frame_id": frame_id,
                "session_average_seconds": elapsed / processed,
                "updated_utc": utc_now(),
            }
            write_json_atomic(output / "status.json", status, overwrite=True)
            print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)


def run_development(contract_path: Path, output: Path) -> None:
    contract, _ = load_contract(contract_path)
    _run_frames(
        contract_path=contract_path,
        output=output,
        study="development_iso100",
        cohort_selector=_development_cohort,
        candidates=list(_candidate_map(contract).values()),
    )


def run_smoke(contract_path: Path) -> dict[str, Any]:
    contract, contract_hash = load_contract(contract_path)
    module, protocol, _, cohort, c057_hash = load_c057_environment()
    development = _development_cohort(cohort)
    _, frame = next(
        item
        for item in module.D0.frame_records(development)
        if int(item[1].iloc[0]["speed_kmh"]) == 80
    )
    image = module.D0.verify_row_assets(frame)
    device = module.D0.require_cuda()
    models, tiled_restore = module.D0.load_models(protocol, device)
    candidate = next(
        candidate
        for candidate in _candidate_map(contract).values()
        if candidate.alpha < 0.0
    )
    crop, x0, y0 = module.D0.union_context(
        image, frame, int(protocol["restoration"]["context_margin_px"])
    )
    restored = _run_on_crop(
        module, protocol, models[("on", "0")], tiled_restore, crop, device
    )
    event = _adapter_event(
        module, crop, restored, frame, candidate, 0, x0, y0, reference_bypass=False
    )
    if len(event["rois"]) != len(frame):
        raise ProtocolError("Smoke ROI count changed")
    return {
        "state": "PASS",
        "frame_id": str(frame.iloc[0]["frame_id"]),
        "route_id": candidate.route_id,
        "seed": 0,
        "roi_count": len(event["rois"]),
        "valid_rois": sum(bool(row["valid"]) for row in event["rois"]),
        "diagnostics": event["diagnostics"],
        "contract_sha256": contract_hash,
        "c057_restoration_freeze_sha256": c057_hash,
        "gpu": str(device),
    }


def _collect_rows(
    module,
    cohort: pd.DataFrame,
    output: Path,
    *,
    contract_hash: str,
    candidates: list[AdapterCandidate],
    c057_freeze_hash: str,
    study: str,
    selection_hash: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    roi_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, str]] = []
    for index, (_, frame) in enumerate(module.D0.frame_records(cohort), start=1):
        frame_id = str(frame.iloc[0]["frame_id"])
        path = _progress_path(output, index, frame_id)
        if not path.is_file():
            raise ProtocolError(f"Missing progress: {path}")
        _validate_progress(
            path,
            frame,
            contract_hash=contract_hash,
            candidates=candidates,
            c057_freeze_hash=c057_freeze_hash,
            study=study,
            selection_hash=selection_hash,
        )
        payload = read_json(path)
        manifest.append(
            {
                "frame_id": frame_id,
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
        )
        metadata = {int(row.roi_id): row for row in frame.itertuples(index=False)}
        for event in payload["events"]:
            route_id = str(event["route_id"])
            seed = str(event["seed"])
            if event["kind"] == "adapter":
                diagnostic_rows.append(
                    {
                        "frame_id": frame_id,
                        "condition_cell": str(frame.iloc[0]["condition"]),
                        "speed_kmh": int(frame.iloc[0]["speed_kmh"]),
                        "distance_m": float(frame.iloc[0]["distance_m"]),
                        "iso": int(frame.iloc[0]["iso"]),
                        "route_id": route_id,
                        "seed": int(seed),
                        **event["candidate"],
                        **event["diagnostics"],
                    }
                )
            for result in event["rois"]:
                source = metadata[int(result["roi_id"])]
                roi_rows.append(
                    {
                        "frame_id": frame_id,
                        "condition_cell": source.condition,
                        "speed_kmh": int(source.speed_kmh),
                        "distance_m": float(source.distance_m),
                        "iso": int(source.iso),
                        "roi_id": int(result["roi_id"]),
                        "orientation": str(result["orientation"]),
                        "route_id": route_id,
                        "seed": seed if seed == "none" else int(seed),
                        "run_key": (
                            "raw_none" if route_id == "raw" else f"{route_id}_{int(seed)}"
                        ),
                        "mtf50": (
                            np.nan
                            if result["mtf50"] is None
                            else float(result["mtf50"])
                        ),
                        "valid": bool(result["valid"]),
                    }
                )
    return pd.DataFrame(roi_rows), pd.DataFrame(diagnostic_rows), manifest


def _condition_table(
    roi: pd.DataFrame,
    diagnostics: pd.DataFrame,
    candidates: list[AdapterCandidate],
    *,
    endpoint_filter,
    coverage: dict[str, Any],
    reference_values: dict[float, float] | None = None,
    reference_exact: dict[tuple[str, int], bool] | None = None,
) -> tuple[pd.DataFrame, dict[float, float]]:
    raw = roi[roi["route_id"] == "raw"].copy()
    raw_frames = (
        raw.groupby(
            ["frame_id", "condition_cell", "speed_kmh", "distance_m", "iso"],
            as_index=False,
        )
        .agg(mtf50=("mtf50", "mean"), valid_rois=("valid", "sum"), rois=("valid", "size"))
    )
    raw_conditions = (
        raw_frames.groupby(
            ["condition_cell", "speed_kmh", "distance_m", "iso"], as_index=False
        )
        .agg(raw_mtf50=("mtf50", "median"))
    )
    if reference_values is None:
        reference = raw_conditions[
            (raw_conditions["speed_kmh"] == 60) & (raw_conditions["iso"] == 100)
        ]
        if len(reference) != 5:
            raise ProtocolError("Distance-specific ISO100 RAW references changed")
        reference_values = {
            float(row.distance_m): float(row.raw_mtf50)
            for row in reference.itertuples(index=False)
        }
    adapters = roi[roi["route_id"] != "raw"].copy()
    adapters["seed"] = adapters["seed"].astype(int)
    adapter_frames = (
        adapters.groupby(
            [
                "frame_id",
                "condition_cell",
                "speed_kmh",
                "distance_m",
                "iso",
                "route_id",
                "seed",
            ],
            as_index=False,
        )
        .agg(mtf50=("mtf50", "mean"), valid_rois=("valid", "sum"), rois=("valid", "size"))
    )
    condition = (
        adapter_frames.groupby(
            ["condition_cell", "speed_kmh", "distance_m", "iso", "route_id", "seed"],
            as_index=False,
        )
        .agg(
            restored_mtf50=("mtf50", "median"),
        )
    )
    condition = condition.merge(
        raw_conditions,
        on=["condition_cell", "speed_kmh", "distance_m", "iso"],
        validate="many_to_one",
    )
    condition["reference_mtf50"] = condition["distance_m"].map(reference_values)
    condition["abs_error"] = np.abs(
        condition["restored_mtf50"] - condition["reference_mtf50"]
    )
    condition["raw_abs_error"] = np.abs(
        condition["raw_mtf50"] - condition["reference_mtf50"]
    )
    condition = condition[endpoint_filter(condition)].copy()
    candidate_frame = pd.DataFrame([candidate.to_dict() for candidate in candidates])
    condition = condition.merge(candidate_frame, on="route_id", validate="many_to_one")
    diagnostic_condition = (
        diagnostics.groupby(["condition_cell", "route_id", "seed"], as_index=False)
        .agg(
            introduced_clipping_fraction=("introduced_clipping_fraction", "max"),
            preclip_overshoot_fraction=("preclip_overshoot_fraction", "max"),
            max_abs_change_dn=("max_abs_change_dn", "max"),
        )
    )
    condition = condition.merge(
        diagnostic_condition,
        on=["condition_cell", "route_id", "seed"],
        validate="one_to_one",
    )
    if reference_exact is None:
        refs = diagnostics[
            (diagnostics["speed_kmh"] == 60) & (diagnostics["iso"] == 100)
        ]
        reference_exact = {
            (str(route_id), int(seed)): bool(group["exact_identity"].all())
            and bool(group["reference_bypass"].all())
            for (route_id, seed), group in refs.groupby(["route_id", "seed"])
        }
    condition["reference_exact"] = [
        reference_exact.get((str(row.route_id), int(row.seed)), False)
        for row in condition.itertuples(index=False)
    ]
    condition["common_complete_case_overall_fraction"] = float(
        coverage["common_complete_case_fraction"]
    )
    condition["common_complete_case_min_distance_fraction"] = float(
        coverage["minimum_distance_common_complete_case_fraction"]
    )
    return condition, reference_values


def aggregate_development(contract_path: Path, output: Path) -> dict[str, Any]:
    final_path = output / "development_final.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    contract, contract_hash = load_contract(contract_path)
    module, _, _, cohort, c057_hash = load_c057_environment()
    cohort = _development_cohort(cohort)
    candidates = list(_candidate_map(contract).values())
    roi, diagnostics, manifest = _collect_rows(
        module,
        cohort,
        output,
        contract_hash=contract_hash,
        candidates=candidates,
        c057_freeze_hash=c057_hash,
        study="development_iso100",
    )
    expected_run_keys = ["raw_none"] + [
        f"{candidate.route_id}_{seed}"
        for seed in (0, 1, 2)
        for candidate in candidates
    ]
    complete, coverage = build_common_complete_case(
        roi,
        expected_run_keys,
        minimum_overall_fraction=float(
            contract["safety"]["minimum_overall_common_complete_case_fraction"]
        ),
        minimum_each_distance_fraction=float(
            contract["safety"][
                "minimum_each_distance_common_complete_case_fraction"
            ]
        ),
    )
    conditions, reference_values = _condition_table(
        complete,
        diagnostics,
        candidates,
        endpoint_filter=lambda frame: (frame["speed_kmh"] == 80)
        & (frame["iso"] == 100),
        coverage=coverage,
    )
    roi_path = output / "development_roi_results.csv"
    complete_path = output / "development_complete_case_roi_results.csv"
    diagnostics_path = output / "development_pixel_diagnostics.csv"
    conditions_path = output / "development_condition_results.csv"
    summaries_path = output / "development_candidate_summaries.json"
    coverage_path = output / "development_complete_case_report.json"
    manifest_path = output / "development_progress_manifest.json"
    write_csv_atomic(roi_path, roi)
    write_csv_atomic(complete_path, complete)
    write_csv_atomic(diagnostics_path, diagnostics)
    write_csv_atomic(conditions_path, conditions)
    # Re-read the release CSV before deriving the frozen summaries.  The same
    # operation is repeated independently before selection can be frozen.
    released_conditions = pd.read_csv(conditions_path)
    summaries = summarize_candidate_metrics(
        released_conditions,
        bootstrap_count=int(contract["bootstrap"]["count"]),
        bootstrap_seed=int(contract["bootstrap"]["seed"]),
        safety=contract["safety"],
    )
    selection = rank_and_select_candidates(summaries)
    selected_summary = next(
        (
            row
            for row in summaries
            if row["route_id"] == selection["selected_route_id"]
        ),
        None,
    )
    raw_only_ablation = None
    if selected_summary is not None and float(selected_summary["sharp_gain"]) > 0.0:
        raw_only_ablation = next(
            (
                row
                for row in summaries
                if float(row["alpha"]) == 0.0
                and float(row["sharp_gain"]) == float(selected_summary["sharp_gain"])
            ),
            None,
        )
    write_json_atomic(summaries_path, summaries)
    write_json_atomic(coverage_path, coverage)
    write_json_atomic(manifest_path, manifest)
    artifact_paths = {
        "roi_results": roi_path,
        "common_complete_case_roi_results": complete_path,
        "pixel_diagnostics": diagnostics_path,
        "condition_results": conditions_path,
        "candidate_summaries": summaries_path,
        "common_complete_case_report": coverage_path,
        "progress_manifest": manifest_path,
    }
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "state": "DEVELOPMENT_COMPLETE",
        "gate": selection["gate"],
        "selection": selection,
        "selected_candidate_summary": selected_summary,
        "raw_only_unsharp_ablation": raw_only_ablation,
        "common_complete_case": coverage,
        "reference_mtf50_by_distance": {
            f"{distance:.1f}": value for distance, value in sorted(reference_values.items())
        },
        "counts": {
            "frames": int(cohort["frame_id"].nunique()),
            "roi_identities": int(len(cohort)),
            "common_complete_case_roi_identities": int(
                coverage["common_complete_case_roi_identities"]
            ),
            "condition_cells": int(cohort["condition"].nunique()),
            "candidate_routes": len(candidates),
            "seeds": 3,
        },
        "inputs": {
            "contract_sha256": contract_hash,
            "c057_restoration_freeze_sha256": c057_hash,
        },
        "artifacts": {label: _binding(path) for label, path in artifact_paths.items()},
        "completed_utc": utc_now(),
    }
    write_json_atomic(final_path, payload)
    write_hash_sidecar(final_path)
    return payload


def _assert_released_frame_equal(
    recomputed: pd.DataFrame, released: pd.DataFrame, label: str
) -> None:
    if list(recomputed.columns) != list(released.columns):
        raise ProtocolError(f"{label} columns/order do not recompute")
    left = recomputed.reset_index(drop=True).copy()
    right = released.reset_index(drop=True).copy()
    for column in (
        "seed",
        "run_key",
        "route_id",
        "frame_id",
        "condition_cell",
        "orientation",
    ):
        if column in left.columns:
            left[column] = left[column].astype(str)
            right[column] = right[column].astype(str)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-13,
            atol=1e-15,
        )
    except AssertionError as error:
        raise ProtocolError(f"{label} rows/values do not recompute: {error}") from error


def verify_development_release(
    contract_path: Path, development_output: Path
) -> tuple[dict[str, Any], dict[str, Any], AdapterCandidate]:
    """Fail closed unless every released development claim recomputes exactly."""

    contract, contract_hash = load_contract(contract_path)
    final_path = development_output / "development_final.json"
    require_sidecar(final_path)
    final = read_json(final_path)
    if final.get("schema_version") != CONTRACT_SCHEMA or final.get("state") != "DEVELOPMENT_COMPLETE":
        raise ProtocolError("Development final schema/state mismatch")
    module, _, _, cohort, c057_hash = load_c057_environment()
    cohort = _development_cohort(cohort)
    if final.get("inputs") != {
        "contract_sha256": contract_hash,
        "c057_restoration_freeze_sha256": c057_hash,
    }:
        raise ProtocolError("Development final contract/C057 bindings mismatch")
    expected_paths = {
        "roi_results": development_output / "development_roi_results.csv",
        "common_complete_case_roi_results": development_output
        / "development_complete_case_roi_results.csv",
        "pixel_diagnostics": development_output / "development_pixel_diagnostics.csv",
        "condition_results": development_output / "development_condition_results.csv",
        "candidate_summaries": development_output
        / "development_candidate_summaries.json",
        "common_complete_case_report": development_output
        / "development_complete_case_report.json",
        "progress_manifest": development_output / "development_progress_manifest.json",
    }
    if set(final.get("artifacts", {})) != set(expected_paths):
        raise ProtocolError("Development final artifact label set mismatch")
    for label, expected_path in expected_paths.items():
        record = final["artifacts"][label]
        if resolve_path(str(record.get("path", ""))).resolve() != expected_path.resolve():
            raise ProtocolError(f"Development artifact path mismatch: {label}")
        if str(record.get("sha256", "")).lower() != sha256_file(expected_path):
            raise ProtocolError(f"Development artifact hash mismatch: {label}")
    candidates = list(_candidate_map(contract).values())
    recomputed_roi, recomputed_diagnostics, recomputed_manifest = _collect_rows(
        module,
        cohort,
        development_output,
        contract_hash=contract_hash,
        candidates=candidates,
        c057_freeze_hash=c057_hash,
        study="development_iso100",
        selection_hash=None,
    )
    released_roi = pd.read_csv(expected_paths["roi_results"])
    released_diagnostics = pd.read_csv(expected_paths["pixel_diagnostics"])
    released_manifest = read_json(expected_paths["progress_manifest"])
    _assert_released_frame_equal(
        recomputed_roi, released_roi, "Progress-derived ROI result table"
    )
    _assert_released_frame_equal(
        recomputed_diagnostics,
        released_diagnostics,
        "Progress-derived pixel diagnostics table",
    )
    if canonical_sha256(recomputed_manifest) != canonical_sha256(released_manifest):
        raise ProtocolError("Progress manifest does not recompute from progress files")
    expected_run_keys = ["raw_none"] + [
        f"{candidate.route_id}_{seed}"
        for seed in (0, 1, 2)
        for candidate in candidates
    ]
    recomputed_complete, recomputed_coverage = build_common_complete_case(
        recomputed_roi,
        expected_run_keys,
        minimum_overall_fraction=float(
            contract["safety"]["minimum_overall_common_complete_case_fraction"]
        ),
        minimum_each_distance_fraction=float(
            contract["safety"][
                "minimum_each_distance_common_complete_case_fraction"
            ]
        ),
    )
    released_coverage = read_json(expected_paths["common_complete_case_report"])
    if canonical_sha256(recomputed_coverage) != canonical_sha256(released_coverage):
        raise ProtocolError("Common complete-case coverage does not recompute")
    released_complete = pd.read_csv(expected_paths["common_complete_case_roi_results"])
    _assert_released_frame_equal(
        recomputed_complete, released_complete, "Common complete-case ROI table"
    )
    recomputed_conditions, reference_values = _condition_table(
        recomputed_complete,
        recomputed_diagnostics,
        candidates,
        endpoint_filter=lambda frame: (frame["speed_kmh"] == 80)
        & (frame["iso"] == 100),
        coverage=recomputed_coverage,
    )
    released_conditions = pd.read_csv(expected_paths["condition_results"])
    _assert_released_frame_equal(
        recomputed_conditions, released_conditions, "Development condition table"
    )
    recomputed_summaries = summarize_candidate_metrics(
        recomputed_conditions,
        bootstrap_count=int(contract["bootstrap"]["count"]),
        bootstrap_seed=int(contract["bootstrap"]["seed"]),
        safety=contract["safety"],
    )
    released_summaries = read_json(expected_paths["candidate_summaries"])
    if canonical_sha256(recomputed_summaries) != canonical_sha256(released_summaries):
        raise ProtocolError("Condition CSV does not reproduce candidate summaries")
    recomputed_selection = rank_and_select_candidates(recomputed_summaries)
    if canonical_sha256(recomputed_selection) != canonical_sha256(final.get("selection")):
        raise ProtocolError("Released route ranking/selection does not recompute")
    if final.get("gate") != recomputed_selection["gate"]:
        raise ProtocolError("Released development gate does not match recomputation")
    route_id = recomputed_selection["selected_route_id"]
    selected_summary = next(
        (row for row in recomputed_summaries if row["route_id"] == route_id), None
    )
    if canonical_sha256(selected_summary) != canonical_sha256(
        final.get("selected_candidate_summary")
    ):
        raise ProtocolError("Released selected-route summary does not recompute")
    recomputed_raw_only = None
    if selected_summary is not None and float(selected_summary["sharp_gain"]) > 0.0:
        recomputed_raw_only = next(
            (
                row
                for row in recomputed_summaries
                if float(row["alpha"]) == 0.0
                and float(row["sharp_gain"]) == float(selected_summary["sharp_gain"])
            ),
            None,
        )
    if canonical_sha256(recomputed_raw_only) != canonical_sha256(
        final.get("raw_only_unsharp_ablation")
    ):
        raise ProtocolError("Released raw-only ablation does not recompute")
    recomputed_reference = {
        f"{distance:.1f}": value for distance, value in sorted(reference_values.items())
    }
    if canonical_sha256(recomputed_reference) != canonical_sha256(
        final.get("reference_mtf50_by_distance")
    ):
        raise ProtocolError("Released distance references do not recompute")
    expected_counts = {
        "frames": int(cohort["frame_id"].nunique()),
        "roi_identities": int(len(cohort)),
        "common_complete_case_roi_identities": int(
            recomputed_coverage["common_complete_case_roi_identities"]
        ),
        "condition_cells": int(cohort["condition"].nunique()),
        "candidate_routes": len(candidates),
        "seeds": 3,
    }
    if final.get("counts") != expected_counts:
        raise ProtocolError("Released development counts do not recompute")
    if canonical_sha256(recomputed_coverage) != canonical_sha256(
        final.get("common_complete_case")
    ):
        raise ProtocolError("Released final common complete-case report does not recompute")
    candidate = _candidate_map(contract).get(str(route_id))
    if candidate is None:
        raise ProtocolError("Released selected route is absent from contract")
    return contract, final, candidate


def freeze_selection(
    contract_path: Path, development_output: Path, selection_path: Path
) -> dict[str, Any]:
    contract, final, candidate = verify_development_release(
        contract_path, development_output
    )
    contract_hash = require_sidecar(contract_path)
    final_path = development_output / "development_final.json"
    if final.get("gate") != "PASS":
        raise ProtocolError("Selection freeze prohibited because development gate is not PASS")
    if candidate is None or candidate.alpha == 0.0:
        raise ProtocolError("Selected route is missing or lacks MERIT residual dependency")
    payload = {
        "schema_version": SELECTION_SCHEMA,
        "state": "ROUTE_FROZEN_AFTER_DEVELOPMENT_PASS_BEFORE_EXISTING40_CONFIRMATION",
        "created_utc": utc_now(),
        "selected_candidate": candidate.to_dict(),
        "development_gate": "PASS",
        "evidence_label_for_confirmation": CONFIRMATION_LABEL,
        "bindings": {
            "contract": {"path": relative_path(contract_path), "sha256": contract_hash},
            "development_final": _binding(final_path),
            "development_final_sidecar": _binding(_sidecar_path(final_path)),
            "development_condition_results": _binding(
                development_output / "development_condition_results.csv"
            ),
            "development_candidate_summaries": _binding(
                development_output / "development_candidate_summaries.json"
            ),
        },
        "prohibitions": [
            "No route or seed change after this freeze.",
            "The existing 40-condition run is not independent confirmation.",
        ],
    }
    write_json_atomic(selection_path, payload)
    digest = write_hash_sidecar(selection_path)
    return {"state": "PASS", "selection_sha256": digest, **payload}


def load_selection(path: Path, contract_path: Path) -> tuple[dict[str, Any], str]:
    digest = require_sidecar(path)
    payload = read_json(path)
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise ProtocolError("Unexpected selection-freeze schema")
    if payload.get("state") != "ROUTE_FROZEN_AFTER_DEVELOPMENT_PASS_BEFORE_EXISTING40_CONFIRMATION":
        raise ProtocolError("Selection-freeze state changed")
    contract, contract_hash = load_contract(contract_path)
    if payload["bindings"]["contract"]["sha256"] != contract_hash:
        raise ProtocolError("Selection freeze points to a different contract")
    for label, record in payload["bindings"].items():
        path_value = resolve_path(record["path"])
        if sha256_file(path_value) != record["sha256"]:
            raise ProtocolError(f"Selection binding mismatch: {label}")
    candidate = AdapterCandidate.from_dict(payload["selected_candidate"])
    if _candidate_map(contract).get(candidate.route_id) != candidate or candidate.alpha == 0.0:
        raise ProtocolError("Frozen selected candidate is invalid")
    return payload, digest


def confirm_existing_40(
    contract_path: Path,
    selection_path: Path,
    output: Path,
    *,
    stage: str,
) -> dict[str, Any] | None:
    selection, selection_hash = load_selection(selection_path, contract_path)
    candidate = AdapterCandidate.from_dict(selection["selected_candidate"])
    if stage == "inference":
        _run_frames(
            contract_path=contract_path,
            output=output,
            study="confirmation_existing40",
            cohort_selector=_confirmation_cohort,
            candidates=[candidate],
            selection_hash=selection_hash,
        )
        return None
    if stage != "aggregate":
        raise ValueError(stage)
    final_path = output / "confirmation_final.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    contract, contract_hash = load_contract(contract_path)
    module, _, _, cohort, c057_hash = load_c057_environment()
    cohort = _confirmation_cohort(cohort)
    roi, diagnostics, manifest = _collect_rows(
        module,
        cohort,
        output,
        contract_hash=contract_hash,
        candidates=[candidate],
        c057_freeze_hash=c057_hash,
        study="confirmation_existing40",
        selection_hash=selection_hash,
    )
    expected_run_keys = ["raw_none"] + [
        f"{candidate.route_id}_{seed}" for seed in (0, 1, 2)
    ]
    complete, coverage = build_common_complete_case(
        roi,
        expected_run_keys,
        minimum_overall_fraction=float(
            contract["safety"]["minimum_overall_common_complete_case_fraction"]
        ),
        minimum_each_distance_fraction=float(
            contract["safety"][
                "minimum_each_distance_common_complete_case_fraction"
            ]
        ),
    )
    development_final_path = resolve_path(selection["bindings"]["development_final"]["path"])
    development_final = read_json(development_final_path)
    reference_values = {
        float(key): float(value)
        for key, value in development_final["reference_mtf50_by_distance"].items()
    }
    reference_exact = {(candidate.route_id, seed): True for seed in (0, 1, 2)}
    conditions, _ = _condition_table(
        complete,
        diagnostics,
        [candidate],
        endpoint_filter=lambda frame: frame["iso"].isin([200, 400, 800, 1600]),
        coverage=coverage,
        reference_values=reference_values,
        reference_exact=reference_exact,
    )
    summaries = summarize_candidate_metrics(
        conditions,
        bootstrap_count=int(contract["bootstrap"]["count"]),
        bootstrap_seed=int(contract["bootstrap"]["seed"]) + 40,
        safety=contract["safety"],
        expected_conditions=40,
        minimum_improved_conditions=int(
            contract["confirmation_existing_40"]["minimum_improved_conditions"]
        ),
        enforce_incremental_contribution=False,
    )
    summary = summaries[0]
    roi_path = output / "confirmation_roi_results.csv"
    complete_path = output / "confirmation_complete_case_roi_results.csv"
    diagnostics_path = output / "confirmation_pixel_diagnostics.csv"
    conditions_path = output / "confirmation_condition_results.csv"
    manifest_path = output / "confirmation_progress_manifest.json"
    coverage_path = output / "confirmation_complete_case_report.json"
    write_csv_atomic(roi_path, roi)
    write_csv_atomic(complete_path, complete)
    write_csv_atomic(diagnostics_path, diagnostics)
    write_csv_atomic(conditions_path, conditions)
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(coverage_path, coverage)
    artifact_paths = {
        "roi_results": roi_path,
        "common_complete_case_roi_results": complete_path,
        "pixel_diagnostics": diagnostics_path,
        "condition_results": conditions_path,
        "common_complete_case_report": coverage_path,
        "progress_manifest": manifest_path,
    }
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "state": "EXISTING40_CONFIRMATION_COMPLETE",
        "evidence_label": CONFIRMATION_LABEL,
        "gate": summary["gate"],
        "selected_candidate": candidate.to_dict(),
        "summary": summary,
        "common_complete_case": coverage,
        "counts": {
            "frames": int(cohort["frame_id"].nunique()),
            "roi_identities": int(len(cohort)),
            "condition_cells": int(cohort["condition"].nunique()),
            "seeds": 3,
        },
        "inputs": {
            "contract_sha256": contract_hash,
            "selection_freeze_sha256": selection_hash,
            "c057_restoration_freeze_sha256": c057_hash,
        },
        "artifacts": {label: _binding(path) for label, path in artifact_paths.items()},
        "completed_utc": utc_now(),
    }
    write_json_atomic(final_path, payload)
    write_hash_sidecar(final_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "create-contract",
            "check-contract",
            "smoke",
            "run-development",
            "aggregate-development",
            "freeze-selection",
            "confirm-existing-40",
        ),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_FREEZE)
    parser.add_argument(
        "--stage", choices=("inference", "aggregate"), default="inference"
    )
    return parser.parse_args()


def resolve_output_path(command: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    if command == "confirm-existing-40":
        return DEFAULT_CONFIRMATION_OUTPUT
    return DEFAULT_DEVELOPMENT_OUTPUT


def main() -> None:
    args = parse_args()
    output = resolve_output_path(args.command, args.output)
    if args.command == "create-contract":
        result = create_contract(args.contract)
    elif args.command == "check-contract":
        contract, digest = load_contract(args.contract)
        result = {
            "state": "PASS",
            "contract_sha256": digest,
            "candidate_count": len(contract["candidate_grid"]),
        }
    elif args.command == "smoke":
        result = run_smoke(args.contract)
    elif args.command == "run-development":
        run_development(args.contract, output)
        return
    elif args.command == "aggregate-development":
        result = aggregate_development(args.contract, output)
    elif args.command == "freeze-selection":
        result = freeze_selection(args.contract, output, args.selection)
    else:
        result = confirm_existing_40(
            args.contract,
            args.selection,
            output,
            stage=args.stage,
        )
        if result is None:
            return
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
