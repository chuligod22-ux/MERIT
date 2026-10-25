"""Tolerance-safe release verification and selection freeze for C058 v2a.

This addendum never changes the frozen v2 code, contract, progress, or result
artifacts.  Pixel/ROI reconstruction products are compared to their released
CSV serialisations with the frozen numerical tolerance.  Statistical summaries
and route selection are then recomputed from the released condition CSV, which
is the exact representation used to create the frozen v2 summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = BASE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import c058_development_v2 as v2  # noqa: E402


ADDENDUM_SCHEMA = "MERIT-C058-RELEASE-VERIFICATION-ADDENDUM-v2a"
RESULT_SCHEMA = "MERIT-C058-RELEASE-VERIFICATION-RESULT-v2a"
ADDENDUM_STATE = "FROZEN_AFTER_V2_RELEASE_BEFORE_V2A_SELECTION"
RESULT_STATE = "PASS_PROGRESS_TO_SELECTION_RECOMPUTATION"
DEFAULT_CONTRACT = BASE / "protocol" / "development_contract_v2.json"
DEFAULT_DEVELOPMENT_OUTPUT = BASE / "artifacts" / "development_v2"
DEFAULT_ADDENDUM = BASE / "protocol" / "verification_addendum_v2a.json"
DEFAULT_VERIFICATION = (
    DEFAULT_DEVELOPMENT_OUTPUT / "release_verification_v2a.json"
)
DEFAULT_SELECTION = BASE / "protocol" / "selected_route_v2a.json"
ORIGINAL_V2_SELECTION = BASE / "protocol" / "selected_route_v2.json"
FLOAT_RTOL = 1e-13
FLOAT_ATOL = 1e-15
MAX_ALLOWED_ABS_DIFF = 1e-13


class AddendumError(RuntimeError):
    """Raised when a v2a release binding or recomputation check fails."""


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "roi_results": output / "development_roi_results.csv",
        "common_complete_case_roi_results": output
        / "development_complete_case_roi_results.csv",
        "pixel_diagnostics": output / "development_pixel_diagnostics.csv",
        "condition_results": output / "development_condition_results.csv",
        "candidate_summaries": output / "development_candidate_summaries.json",
        "common_complete_case_report": output
        / "development_complete_case_report.json",
        "progress_manifest": output / "development_progress_manifest.json",
    }


def _binding(path: Path) -> dict[str, str]:
    return {"path": v2.relative_path(path), "sha256": v2.sha256_file(path)}


def _write_json(path: Path, payload: Any) -> None:
    v2.write_json_atomic(path, payload, overwrite=False)
    v2.write_hash_sidecar(path)


def create_addendum(
    path: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    output: Path = DEFAULT_DEVELOPMENT_OUTPUT,
) -> dict[str, Any]:
    if path.exists() or _sidecar(path).exists():
        raise FileExistsError(path)
    contract, contract_hash = v2.load_contract(contract_path)
    final_path = output / "development_final.json"
    final_hash = v2.require_sidecar(final_path)
    final = v2.read_json(final_path)
    artifacts = _artifact_paths(output)
    if set(final.get("artifacts", {})) != set(artifacts):
        raise AddendumError("Frozen v2 final artifact labels changed")
    bindings = {
        "v2_contract": {"path": v2.relative_path(contract_path), "sha256": contract_hash},
        "v2_contract_sidecar": _binding(_sidecar(contract_path)),
        "v2_development_code": _binding(BASE / "scripts" / "c058_development_v2.py"),
        "v2_adapter_code": _binding(BASE / "scripts" / "c058_adapter.py"),
        "v2a_addendum_code": _binding(Path(__file__)),
        "v2a_addendum_tests": _binding(
            BASE / "tests" / "test_c058_release_addendum_v2a.py"
        ),
        "v2_final": {"path": v2.relative_path(final_path), "sha256": final_hash},
        "v2_final_sidecar": _binding(_sidecar(final_path)),
    }
    for label, artifact_path in artifacts.items():
        bindings[f"v2_artifact:{label}"] = _binding(artifact_path)
        record = final["artifacts"][label]
        if (
            v2.resolve_path(str(record.get("path", ""))).resolve()
            != artifact_path.resolve()
            or str(record.get("sha256", "")).lower()
            != bindings[f"v2_artifact:{label}"]["sha256"]
        ):
            raise AddendumError(f"Frozen v2 final artifact binding mismatch: {label}")
    payload = {
        "schema_version": ADDENDUM_SCHEMA,
        "state": ADDENDUM_STATE,
        "purpose": (
            "Correct the release verifier only: compare progress-derived floating "
            "tables to released CSV with frozen tolerance, then exactly recompute "
            "summaries and selection from the released condition CSV."
        ),
        "frozen_tolerance": {
            "relative": FLOAT_RTOL,
            "absolute": FLOAT_ATOL,
            "maximum_allowed_absolute_difference": MAX_ALLOWED_ABS_DIFF,
        },
        "verification_order": [
            "verify all v2/addendum file hashes",
            "reconstruct ROI, diagnostics, and manifest from all progress files",
            "reconstruct common complete-case and condition tables",
            "compare reconstructed numeric tables to released CSV with tolerance",
            "recompute summaries and route selection exactly from released condition CSV",
            "require unchanged selected route and PASS gate",
        ],
        "bindings": bindings,
        "original_v2": {
            "contract_schema": contract["schema_version"],
            "contract_sha256": contract_hash,
            "final_sha256": final_hash,
            "selected_route_id": final["selection"]["selected_route_id"],
            "gate": final["gate"],
        },
        "prohibitions": [
            "Do not modify or overwrite any v2 code, contract, progress, final, or artifact.",
            "Do not relax the frozen 1e-13/1e-15 numerical tolerance.",
            "Do not change the released condition CSV, summary, route ranking, or gate.",
            "Do not overwrite an existing v2 or v2a selection file.",
        ],
    }
    _write_json(path, payload)
    return {
        "state": "PASS",
        "addendum_sha256": v2.sha256_file(path),
        "bound_files": len(bindings),
    }


def load_addendum(path: Path) -> tuple[dict[str, Any], str]:
    digest = v2.require_sidecar(path)
    payload = v2.read_json(path)
    if payload.get("schema_version") != ADDENDUM_SCHEMA:
        raise AddendumError("Unexpected v2a addendum schema")
    if payload.get("state") != ADDENDUM_STATE:
        raise AddendumError("Unexpected v2a addendum state")
    expected_tolerance = {
        "relative": FLOAT_RTOL,
        "absolute": FLOAT_ATOL,
        "maximum_allowed_absolute_difference": MAX_ALLOWED_ABS_DIFF,
    }
    if payload.get("frozen_tolerance") != expected_tolerance:
        raise AddendumError("Frozen v2a tolerance changed")
    for label, record in payload.get("bindings", {}).items():
        bound = v2.resolve_path(str(record.get("path", "")))
        if not bound.is_file() or v2.sha256_file(bound) != str(
            record.get("sha256", "")
        ).lower():
            raise AddendumError(f"v2a binding mismatch: {label}")
    return payload, digest


IDENTITY_AS_TEXT = {
    "seed",
    "run_key",
    "route_id",
    "frame_id",
    "condition_cell",
    "orientation",
}


def compare_frames_with_tolerance(
    recomputed: pd.DataFrame,
    released: pd.DataFrame,
    *,
    label: str,
    rtol: float = FLOAT_RTOL,
    atol: float = FLOAT_ATOL,
    maximum_allowed_absolute_difference: float = MAX_ALLOWED_ABS_DIFF,
) -> dict[str, Any]:
    """Compare identity columns exactly and numeric columns with frozen tolerance."""

    if list(recomputed.columns) != list(released.columns):
        raise AddendumError(f"{label}: column names/order differ")
    if len(recomputed) != len(released):
        raise AddendumError(f"{label}: row count differs")
    left = recomputed.reset_index(drop=True).copy()
    right = released.reset_index(drop=True).copy()
    maximum = 0.0
    numeric_values = 0
    for column in left.columns:
        if column in IDENTITY_AS_TEXT:
            left_values = left[column].astype(str).to_numpy()
            right_values = right[column].astype(str).to_numpy()
            if not np.array_equal(left_values, right_values):
                raise AddendumError(f"{label}: identity column differs: {column}")
            continue
        if pd.api.types.is_bool_dtype(left[column]) or pd.api.types.is_bool_dtype(
            right[column]
        ):
            if not np.array_equal(
                left[column].astype(bool).to_numpy(),
                right[column].astype(bool).to_numpy(),
            ):
                raise AddendumError(f"{label}: boolean column differs: {column}")
            continue
        left_numeric = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_numeric = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        numeric_mask = np.isfinite(left_numeric) | np.isfinite(right_numeric)
        if bool(np.any(numeric_mask)) or (
            bool(np.all(pd.isna(left[column]))) and bool(np.all(pd.isna(right[column])))
        ):
            if not np.array_equal(np.isnan(left_numeric), np.isnan(right_numeric)):
                raise AddendumError(f"{label}: numeric missingness differs: {column}")
            finite = np.isfinite(left_numeric) & np.isfinite(right_numeric)
            if bool(np.any(finite)):
                differences = np.abs(left_numeric[finite] - right_numeric[finite])
                column_maximum = float(np.max(differences))
                maximum = max(maximum, column_maximum)
                numeric_values += int(np.sum(finite))
                if not bool(
                    np.allclose(
                        left_numeric[finite],
                        right_numeric[finite],
                        rtol=rtol,
                        atol=atol,
                    )
                ):
                    raise AddendumError(
                        f"{label}: numeric tolerance exceeded in {column}: {column_maximum}"
                    )
            continue
        left_values = left[column].fillna("<NA>").astype(str).to_numpy()
        right_values = right[column].fillna("<NA>").astype(str).to_numpy()
        if not np.array_equal(left_values, right_values):
            raise AddendumError(f"{label}: text column differs: {column}")
    if not math.isfinite(maximum) or maximum > maximum_allowed_absolute_difference:
        raise AddendumError(
            f"{label}: maximum absolute difference {maximum} exceeds "
            f"{maximum_allowed_absolute_difference}"
        )
    return {
        "label": label,
        "rows": int(len(left)),
        "columns": int(len(left.columns)),
        "numeric_values_compared": numeric_values,
        "maximum_absolute_difference": maximum,
        "rtol": rtol,
        "atol": atol,
        "maximum_allowed_absolute_difference": maximum_allowed_absolute_difference,
        "pass": True,
    }


def _require_exact(left: Any, right: Any, label: str) -> None:
    if v2.canonical_sha256(left) != v2.canonical_sha256(right):
        raise AddendumError(f"Exact recomputation mismatch: {label}")


def verify_release(
    addendum_path: Path = DEFAULT_ADDENDUM,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    output: Path = DEFAULT_DEVELOPMENT_OUTPUT,
) -> dict[str, Any]:
    addendum, addendum_hash = load_addendum(addendum_path)
    contract, contract_hash = v2.load_contract(contract_path)
    final_path = output / "development_final.json"
    final_hash = v2.require_sidecar(final_path)
    final = v2.read_json(final_path)
    if final.get("gate") != "PASS":
        raise AddendumError("Frozen v2 development gate is not PASS")
    if final.get("inputs", {}).get("contract_sha256") != contract_hash:
        raise AddendumError("Frozen v2 final contract binding changed")
    module, _, _, cohort, c057_hash = v2.load_c057_environment()
    cohort = v2._development_cohort(cohort)
    if final.get("inputs", {}).get("c057_restoration_freeze_sha256") != c057_hash:
        raise AddendumError("Frozen v2 final C057 binding changed")
    candidates = list(v2._candidate_map(contract).values())
    artifact_paths = _artifact_paths(output)
    for label, path in artifact_paths.items():
        record = final["artifacts"][label]
        if (
            v2.resolve_path(str(record["path"])).resolve() != path.resolve()
            or str(record["sha256"]).lower() != v2.sha256_file(path)
        ):
            raise AddendumError(f"Frozen v2 artifact binding changed: {label}")

    recomputed_roi, recomputed_diagnostics, recomputed_manifest = v2._collect_rows(
        module,
        cohort,
        output,
        contract_hash=contract_hash,
        candidates=candidates,
        c057_freeze_hash=c057_hash,
        study="development_iso100",
        selection_hash=None,
    )
    released_roi = pd.read_csv(artifact_paths["roi_results"])
    released_diagnostics = pd.read_csv(artifact_paths["pixel_diagnostics"])
    released_manifest = v2.read_json(artifact_paths["progress_manifest"])
    comparisons = {
        "progress_to_roi_csv": compare_frames_with_tolerance(
            recomputed_roi, released_roi, label="progress_to_roi_csv"
        ),
        "progress_to_diagnostics_csv": compare_frames_with_tolerance(
            recomputed_diagnostics,
            released_diagnostics,
            label="progress_to_diagnostics_csv",
        ),
    }
    _require_exact(
        recomputed_manifest, released_manifest, "progress_to_manifest_json"
    )

    expected_run_keys = ["raw_none"] + [
        f"{candidate.route_id}_{seed}"
        for seed in (0, 1, 2)
        for candidate in candidates
    ]
    recomputed_complete, recomputed_coverage = v2.build_common_complete_case(
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
    released_complete = pd.read_csv(artifact_paths["common_complete_case_roi_results"])
    comparisons["progress_to_complete_case_csv"] = compare_frames_with_tolerance(
        recomputed_complete,
        released_complete,
        label="progress_to_complete_case_csv",
    )
    released_coverage = v2.read_json(artifact_paths["common_complete_case_report"])
    _require_exact(recomputed_coverage, released_coverage, "complete_case_report")
    _require_exact(recomputed_coverage, final["common_complete_case"], "final_complete_case")

    recomputed_conditions, reference_values = v2._condition_table(
        recomputed_complete,
        recomputed_diagnostics,
        candidates,
        endpoint_filter=lambda frame: (frame["speed_kmh"] == 80)
        & (frame["iso"] == 100),
        coverage=recomputed_coverage,
    )
    released_conditions = pd.read_csv(artifact_paths["condition_results"])
    comparisons["progress_to_condition_csv"] = compare_frames_with_tolerance(
        recomputed_conditions,
        released_conditions,
        label="progress_to_condition_csv",
    )

    # Exact claims are intentionally recomputed from the released CSV, matching
    # the frozen v2 aggregate path and avoiding harmless binary-to-decimal drift.
    csv_summaries = v2.summarize_candidate_metrics(
        released_conditions,
        bootstrap_count=int(contract["bootstrap"]["count"]),
        bootstrap_seed=int(contract["bootstrap"]["seed"]),
        safety=contract["safety"],
    )
    frozen_summaries = v2.read_json(artifact_paths["candidate_summaries"])
    _require_exact(csv_summaries, frozen_summaries, "condition_csv_to_summaries")
    csv_selection = v2.rank_and_select_candidates(csv_summaries)
    _require_exact(csv_selection, final["selection"], "summary_to_selection")
    if csv_selection["gate"] != "PASS" or csv_selection["gate"] != final["gate"]:
        raise AddendumError("Recomputed selection/final gate is not the same PASS")
    route_id = str(csv_selection["selected_route_id"])
    selected_summary = next(
        row for row in csv_summaries if row["route_id"] == route_id
    )
    _require_exact(
        selected_summary, final["selected_candidate_summary"], "selected_summary"
    )
    selected_candidate = v2._candidate_map(contract).get(route_id)
    if selected_candidate is None or selected_candidate.alpha == 0.0:
        raise AddendumError("Recomputed selected route is not MERIT-dependent")
    recomputed_reference = {
        f"{distance:.1f}": value for distance, value in sorted(reference_values.items())
    }
    reference_max_diff = max(
        abs(float(recomputed_reference[key]) - float(final["reference_mtf50_by_distance"][key]))
        for key in recomputed_reference
    )
    if reference_max_diff > MAX_ALLOWED_ABS_DIFF:
        raise AddendumError("Recomputed distance references exceed v2a tolerance")
    maximum_difference = max(
        report["maximum_absolute_difference"] for report in comparisons.values()
    )
    if maximum_difference > MAX_ALLOWED_ABS_DIFF:
        raise AddendumError("Maximum table difference exceeds frozen v2a criterion")
    checks = {
        "all_addendum_bindings_match": True,
        "all_v2_artifact_bindings_match": True,
        "progress_manifest_exact": True,
        "complete_case_report_exact": True,
        "condition_csv_to_summary_exact": True,
        "summary_to_selection_exact": True,
        "selected_route_unchanged": route_id
        == str(addendum["original_v2"]["selected_route_id"]),
        "selection_gate_same_pass": csv_selection["gate"]
        == addendum["original_v2"]["gate"]
        == "PASS",
        "maximum_difference_within_frozen_tolerance": maximum_difference
        <= MAX_ALLOWED_ABS_DIFF,
    }
    if not all(checks.values()):
        raise AddendumError(f"v2a release checks failed: {checks}")
    return {
        "schema_version": RESULT_SCHEMA,
        "state": RESULT_STATE,
        "gate": "PASS",
        "selected_route_id": route_id,
        "selected_candidate": selected_candidate.to_dict(),
        "comparisons": comparisons,
        "maximum_absolute_difference": maximum_difference,
        "maximum_allowed_absolute_difference": MAX_ALLOWED_ABS_DIFF,
        "reference_maximum_absolute_difference": reference_max_diff,
        "exact_checks": checks,
        "inputs": {
            "addendum_sha256": addendum_hash,
            "v2_contract_sha256": contract_hash,
            "v2_final_sha256": final_hash,
            "c057_restoration_freeze_sha256": c057_hash,
        },
    }


def write_verification(
    path: Path,
    *,
    addendum_path: Path = DEFAULT_ADDENDUM,
    contract_path: Path = DEFAULT_CONTRACT,
    output: Path = DEFAULT_DEVELOPMENT_OUTPUT,
) -> dict[str, Any]:
    if path.exists() or _sidecar(path).exists():
        raise FileExistsError(path)
    result = verify_release(
        addendum_path, contract_path=contract_path, output=output
    )
    _write_json(path, result)
    return {
        "state": "PASS",
        "verification_sha256": v2.sha256_file(path),
        "selected_route_id": result["selected_route_id"],
        "maximum_absolute_difference": result["maximum_absolute_difference"],
    }


def freeze_selection(
    selection_path: Path = DEFAULT_SELECTION,
    *,
    addendum_path: Path = DEFAULT_ADDENDUM,
    verification_path: Path = DEFAULT_VERIFICATION,
    contract_path: Path = DEFAULT_CONTRACT,
    output: Path = DEFAULT_DEVELOPMENT_OUTPUT,
) -> dict[str, Any]:
    if ORIGINAL_V2_SELECTION.exists():
        raise FileExistsError(
            f"Existing original v2 selection will not be overwritten: {ORIGINAL_V2_SELECTION}"
        )
    if selection_path.exists() or _sidecar(selection_path).exists():
        raise FileExistsError(selection_path)
    verification_hash = v2.require_sidecar(verification_path)
    frozen_verification = v2.read_json(verification_path)
    recomputed = verify_release(
        addendum_path, contract_path=contract_path, output=output
    )
    _require_exact(recomputed, frozen_verification, "frozen_v2a_verification")
    if recomputed.get("gate") != "PASS":
        raise AddendumError("v2a verification gate is not PASS")
    addendum_hash = v2.require_sidecar(addendum_path)
    contract, contract_hash = v2.load_contract(contract_path)
    final_path = output / "development_final.json"
    final = v2.read_json(final_path)
    candidate = v2._candidate_map(contract)[recomputed["selected_route_id"]]
    payload = {
        # These values intentionally match the original v2 loader contract.
        "schema_version": v2.SELECTION_SCHEMA,
        "state": "ROUTE_FROZEN_AFTER_DEVELOPMENT_PASS_BEFORE_EXISTING40_CONFIRMATION",
        "created_utc": v2.utc_now(),
        "selected_candidate": candidate.to_dict(),
        "development_gate": "PASS",
        "verification_addendum": ADDENDUM_SCHEMA,
        "evidence_label_for_confirmation": v2.CONFIRMATION_LABEL,
        "bindings": {
            "contract": {"path": v2.relative_path(contract_path), "sha256": contract_hash},
            "development_final": _binding(final_path),
            "development_final_sidecar": _binding(_sidecar(final_path)),
            "development_condition_results": _binding(
                output / "development_condition_results.csv"
            ),
            "development_candidate_summaries": _binding(
                output / "development_candidate_summaries.json"
            ),
            "verification_addendum_v2a": {
                "path": v2.relative_path(addendum_path),
                "sha256": addendum_hash,
            },
            "verification_addendum_v2a_sidecar": _binding(_sidecar(addendum_path)),
            "release_verification_v2a": {
                "path": v2.relative_path(verification_path),
                "sha256": verification_hash,
            },
            "release_verification_v2a_sidecar": _binding(
                _sidecar(verification_path)
            ),
        },
        "prohibitions": [
            "No route or seed change after this freeze.",
            "Do not modify the frozen v2 or v2a verification artifacts.",
            "The existing 40-condition run is not independent confirmation.",
        ],
    }
    if final["selection"]["selected_route_id"] != candidate.route_id:
        raise AddendumError("v2a selected route differs from frozen v2 route")
    _write_json(selection_path, payload)
    # Acceptance by the unmodified original v2 loader is part of the release gate.
    loaded, selection_hash = v2.load_selection(selection_path, contract_path)
    if loaded["selected_candidate"] != candidate.to_dict():
        raise AddendumError("Original v2 loader returned a different candidate")
    return {
        "state": "PASS",
        "selection_sha256": selection_hash,
        "selected_route_id": candidate.route_id,
        "accepted_by_original_v2_load_selection": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("create-addendum", "check-addendum", "verify-release", "freeze-selection"),
    )
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DEVELOPMENT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create-addendum":
        result = create_addendum(
            args.addendum, contract_path=args.contract, output=args.output
        )
    elif args.command == "check-addendum":
        _, digest = load_addendum(args.addendum)
        result = {"state": "PASS", "addendum_sha256": digest}
    elif args.command == "verify-release":
        result = write_verification(
            args.verification,
            addendum_path=args.addendum,
            contract_path=args.contract,
            output=args.output,
        )
    else:
        result = freeze_selection(
            args.selection,
            addendum_path=args.addendum,
            verification_path=args.verification,
            contract_path=args.contract,
            output=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
