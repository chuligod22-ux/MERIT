"""RAW-only development/holdout runner for the frozen C057 estimator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mtf50_estimator import estimate_mtf50_v1, estimate_mtf50_v2


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = Path(__file__).parent / "protocol" / "mtf50_v2_development_contract_v2.json"
DEFAULT_FREEZE = Path(__file__).parent / "protocol" / "mtf50_v2_preoutcome_freeze_v2.json"
DEFAULT_EXECUTION_FREEZE = (
    Path(__file__).parent / "protocol" / "mtf50_v2_execution_freeze.json"
)
DEFAULT_OUTPUT = Path(__file__).parent / "artifacts" / "development_v2"


class ProtocolError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_contract_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def validate_partitions(
    cohort: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, object]:
    """Fail closed unless the frozen 50-cell ISO partition is exact."""

    required = {
        "frame_id", "condition", "speed_kmh", "distance_m", "iso", "roi_id"
    }
    missing = required - set(cohort.columns)
    if missing:
        raise ProtocolError(f"Cohort partition columns missing: {sorted(missing)}")
    if len(cohort) != 1737 or cohort["condition"].nunique() != 50:
        raise ProtocolError("Full cohort must contain exactly 1,737 ROIs and 50 cells")
    iso_set = set(cohort["iso"].astype(int).unique())
    if iso_set != {100, 200, 400, 800, 1600}:
        raise ProtocolError(f"Unexpected full-cohort ISO set: {sorted(iso_set)}")
    key_columns = ["frame_id", "roi_id"]
    if cohort.duplicated(key_columns).any():
        raise ProtocolError("Cohort contains duplicate frame_id/roi_id keys")

    development = cohort[cohort["iso"] == 100]
    holdout = cohort[cohort["iso"].isin([200, 400, 800, 1600])]
    partition = contract["partition"]
    expectations = {
        "development": (
            development,
            {100},
            int(partition["development_roi_records"]),
            int(partition["development_condition_cells"]),
        ),
        "holdout": (
            holdout,
            {200, 400, 800, 1600},
            int(partition["holdout_roi_records"]),
            int(partition["holdout_condition_cells"]),
        ),
    }
    expected_speeds = {60, 80}
    expected_distances = {2.5, 3.5, 4.5, 5.5, 6.5}
    report: dict[str, object] = {}
    for name, (subset, expected_iso, expected_rois, expected_cells) in expectations.items():
        observed_iso = set(subset["iso"].astype(int).unique())
        if observed_iso != expected_iso:
            raise ProtocolError(f"{name} ISO set changed: {sorted(observed_iso)}")
        if len(subset) != expected_rois:
            raise ProtocolError(f"{name} ROI count {len(subset)} != {expected_rois}")
        if subset["condition"].nunique() != expected_cells:
            raise ProtocolError(
                f"{name} condition count {subset['condition'].nunique()} != {expected_cells}"
            )
        if set(subset["speed_kmh"].astype(int).unique()) != expected_speeds:
            raise ProtocolError(f"{name} speed grid changed")
        if set(subset["distance_m"].astype(float).unique()) != expected_distances:
            raise ProtocolError(f"{name} distance grid changed")
        observed_grid = set(
            zip(
                subset["speed_kmh"].astype(int),
                subset["distance_m"].astype(float),
                subset["iso"].astype(int),
            )
        )
        expected_grid = {
            (speed, distance, iso)
            for speed in expected_speeds
            for distance in expected_distances
            for iso in expected_iso
        }
        if observed_grid != expected_grid:
            raise ProtocolError(f"{name} speed x distance x ISO grid changed")
        condition_metadata = subset.groupby("condition")[
            ["speed_kmh", "distance_m", "iso"]
        ].nunique()
        if (condition_metadata != 1).any().any():
            raise ProtocolError(f"{name} condition labels map to inconsistent metadata")
        report[name] = {
            "iso_set": sorted(expected_iso),
            "roi_records": int(len(subset)),
            "condition_cells": int(subset["condition"].nunique()),
            "speeds_kmh": sorted(expected_speeds),
            "distances_m": sorted(expected_distances),
        }

    development_keys = set(map(tuple, development[key_columns].to_numpy()))
    holdout_keys = set(map(tuple, holdout[key_columns].to_numpy()))
    if development_keys & holdout_keys:
        raise ProtocolError("Development and holdout ROI keys overlap")
    union_keys = development_keys | holdout_keys
    if len(union_keys) != 1737:
        raise ProtocolError("Development/holdout ROI union is not the full cohort")
    condition_union = set(development["condition"]) | set(holdout["condition"])
    if len(condition_union) != 50:
        raise ProtocolError("Development/holdout condition union is not 50 cells")
    report["union"] = {
        "roi_records": len(union_keys),
        "condition_cells": len(condition_union),
        "intersection_roi_records": 0,
    }
    return report


def load_contract_cohort(
    contract_path: Path, freeze_path: Path
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, object]]:
    contract = read_json(contract_path)
    freeze = read_json(freeze_path)
    if contract.get("schema_version") != "MERIT-C057-MTF50-V2-DEVELOPMENT-CONTRACT-v2":
        raise ProtocolError("Unexpected development contract schema")
    if freeze.get("schema_version") != "MERIT-C057-MTF50-V2-PREOUTCOME-FREEZE-v2":
        raise ProtocolError("Unexpected pre-outcome freeze schema")
    expected_contract_hash = freeze["development_contract"]["sha256"]
    if sha256_file(contract_path) != expected_contract_hash:
        raise ProtocolError("Development contract hash differs from freeze")
    cohort_path = resolve_contract_path(contract["inputs"]["cohort_path"])
    if sha256_file(cohort_path) != contract["inputs"]["cohort_sha256"]:
        raise ProtocolError("Cohort hash differs from contract")
    cohort = pd.read_csv(cohort_path)
    if len(cohort) != int(contract["inputs"]["roi_records"]):
        raise ProtocolError("Cohort ROI count differs from contract")
    partition_report = validate_partitions(cohort, contract)
    return contract, freeze, cohort, partition_report


def _read_execution_freeze(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ProtocolError("Execution freeze JSON or SHA-256 sidecar is missing")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ProtocolError("Execution freeze SHA-256 sidecar is malformed")
    actual = sha256_file(path)
    if fields[0].lower() != actual:
        raise ProtocolError("Execution freeze SHA-256 sidecar mismatch")
    payload = read_json(path)
    if payload.get("schema_version") != "MERIT-C057-MTF50-V2-EXECUTION-FREEZE-v1":
        raise ProtocolError("Unexpected execution freeze schema")
    if payload.get("state") != "FROZEN_BEFORE_OFFICIAL_HOLDOUT_AND_RESTORATION_OUTPUTS":
        raise ProtocolError("Execution freeze state is invalid")
    return payload, actual


def verify_execution_freeze(
    execution_freeze_path: Path,
    contract_path: Path,
    preoutcome_freeze_path: Path,
    contract: dict[str, Any],
    preoutcome_freeze: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    payload, freeze_hash = _read_execution_freeze(execution_freeze_path)
    cohort_path = resolve_contract_path(contract["inputs"]["cohort_path"])
    expected_paths = {
        "estimator_source": Path(__file__).with_name("mtf50_estimator.py"),
        "runner_source": Path(__file__),
        "test_source": Path(__file__).with_name("tests") / "test_mtf50_estimator.py",
        "development_contract": contract_path,
        "cohort": cohort_path,
        "preoutcome_freeze": preoutcome_freeze_path,
    }
    records = payload.get("files", {})
    for label, expected_path in expected_paths.items():
        record = records.get(label)
        if not isinstance(record, dict):
            raise ProtocolError(f"Execution freeze file record missing: {label}")
        recorded_path = resolve_contract_path(str(record.get("path", ""))).resolve()
        if recorded_path != expected_path.resolve():
            raise ProtocolError(f"Execution freeze path changed: {label}")
        if str(record.get("sha256", "")).lower() != sha256_file(expected_path):
            raise ProtocolError(f"Execution freeze hash mismatch: {label}")
    selected_sigma = float(
        preoutcome_freeze["selected_candidate"]["localization_gaussian_sigma_px"]
    )
    if float(payload["selected_candidate"]["localization_gaussian_sigma_px"]) != selected_sigma:
        raise ProtocolError("Execution freeze selected sigma differs from pre-outcome freeze")
    if selected_sigma != 0.75:
        raise ProtocolError("Frozen localization sigma must remain 0.75 px")
    return payload, freeze_hash


def load_frozen_inputs(
    contract_path: Path,
    freeze_path: Path,
    execution_freeze_path: Path = DEFAULT_EXECUTION_FREEZE,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, object], str]:
    contract, freeze, cohort, partition_report = load_contract_cohort(
        contract_path, freeze_path
    )
    _, execution_freeze_hash = verify_execution_freeze(
        execution_freeze_path,
        contract_path,
        freeze_path,
        contract,
        freeze,
    )
    return contract, freeze, cohort, partition_report, execution_freeze_hash


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ProtocolError(f"Could not decode image: {path}")
    return image


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 3:
        return math.nan
    return float(spearmanr(left[keep], right[keep]).statistic)


def _partition(cohort: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "development":
        subset = cohort[cohort["iso"] == 100].copy()
        expected = 335
    elif name == "holdout":
        subset = cohort[cohort["iso"].isin([200, 400, 800, 1600])].copy()
        expected = 1402
    else:  # pragma: no cover - argparse enforces this
        raise ProtocolError(f"Unknown partition: {name}")
    if len(subset) != expected:
        raise ProtocolError(f"{name} ROI count {len(subset)} != {expected}")
    return subset.sort_values(["condition", "frame", "roi_id"]).reset_index(drop=True)


def evaluate_partition(cohort: pd.DataFrame, sigma: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    image_cache: dict[str, np.ndarray] = {}
    for source in cohort.itertuples(index=False):
        image_path = str(source.image_path)
        image = image_cache.get(image_path)
        if image is None:
            path = Path(image_path)
            if sha256_file(path) != str(source.image_sha256):
                raise ProtocolError(f"Image hash mismatch: {path}")
            image = read_gray(path)
            image_cache[image_path] = image
        roi = image[int(source.y1) : int(source.y2), int(source.x1) : int(source.x2)]
        v1 = estimate_mtf50_v1(roi, str(source.orientation))
        v2 = estimate_mtf50_v2(
            roi,
            str(source.orientation),
            localization_sigma_px=sigma,
        )
        rows.append(
            {
                "frame_id": source.frame_id,
                "condition": source.condition,
                "speed_kmh": int(source.speed_kmh),
                "distance_m": float(source.distance_m),
                "iso": int(source.iso),
                "frame": source.frame,
                "roi_id": int(source.roi_id),
                "orientation": source.orientation,
                "imatest_roi_mtf50": float(source.imatest_roi_mtf50),
                "preexisting_frame_mtf50": float(source.preexisting_frame_mtf50),
                "v1_mtf50": v1.mtf50 if v1.valid else np.nan,
                "v1_valid": v1.valid,
                "v1_reason": v1.reason,
                "v2_mtf50": v2.mtf50 if v2.valid else np.nan,
                "v2_valid": v2.valid,
                "v2_reason": v2.reason,
                "v2_polarity": v2.polarity,
                "v2_line_support_rows": v2.first_stage_support_rows,
                "v2_total_rows": v2.total_rows,
                "v2_slope": v2.slope,
                "v2_intercept": v2.intercept,
                "localization_sigma_px": sigma,
                "esf_uses_original_pixels": v2.esf_uses_original_pixels,
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, contract: dict[str, Any], partition: str) -> dict[str, Any]:
    valid = results["v2_valid"].to_numpy(bool)
    v2 = results["v2_mtf50"].to_numpy(float)
    v1 = results["v1_mtf50"].to_numpy(float)
    reference = results["imatest_roi_mtf50"].to_numpy(float)
    common = np.isfinite(v1) & np.isfinite(v2)
    frame = (
        results.groupby("frame_id", as_index=False)
        .agg(
            v2_mtf50=("v2_mtf50", "mean"),
            preexisting_frame_mtf50=("preexisting_frame_mtf50", "first"),
        )
    )
    per_distance = []
    for distance, group in results.groupby("distance_m", sort=True):
        per_distance.append(
            {
                "distance_m": float(distance),
                "roi_records": int(len(group)),
                "valid_roi_count": int(group["v2_valid"].sum()),
                "valid_roi_fraction": float(group["v2_valid"].mean()),
            }
        )
    roi_spearman = finite_spearman(v2, reference)
    frame_spearman = finite_spearman(
        frame["v2_mtf50"].to_numpy(float),
        frame["preexisting_frame_mtf50"].to_numpy(float),
    )
    roi_mae = float(np.mean(np.abs(v2[valid] - reference[valid])))
    v1_common_mae = float(np.mean(np.abs(v1[common] - reference[common])))
    v2_common_mae = float(np.mean(np.abs(v2[common] - reference[common])))
    v1_common_spearman = finite_spearman(v1[common], reference[common])
    v2_common_spearman = finite_spearman(v2[common], reference[common])
    metrics = {
        "roi_records": int(len(results)),
        "valid_roi_count": int(valid.sum()),
        "valid_roi_fraction": float(valid.mean()),
        "minimum_distance_valid_fraction": float(
            min(row["valid_roi_fraction"] for row in per_distance)
        ),
        "per_distance": per_distance,
        "roi_spearman_vs_imatest": roi_spearman,
        "frame_spearman_vs_imatest_mean": frame_spearman,
        "roi_mean_absolute_error_cy_px": roi_mae,
        "common_roi_count": int(common.sum()),
        "common_roi_v1_mae_cy_px": v1_common_mae,
        "common_roi_v2_mae_cy_px": v2_common_mae,
        "common_roi_mae_change_vs_v1_cy_px": v2_common_mae - v1_common_mae,
        "common_roi_v1_spearman": v1_common_spearman,
        "common_roi_v2_spearman": v2_common_spearman,
        "common_roi_spearman_change_vs_v1": v2_common_spearman - v1_common_spearman,
        "invalid_reason_counts": {
            str(key): int(value)
            for key, value in results.loc[~valid, "v2_reason"].value_counts().sort_index().items()
        },
    }
    gates = contract[f"{partition}_gates"]
    checks = {
        "overall_valid_fraction": metrics["valid_roi_fraction"]
        >= float(gates["minimum_overall_valid_fraction"]),
        "each_distance_valid_fraction": metrics["minimum_distance_valid_fraction"]
        >= float(gates["minimum_each_distance_valid_fraction"]),
        "roi_spearman_vs_imatest": roi_spearman
        >= float(gates["minimum_roi_spearman_vs_imatest"]),
        "frame_spearman_vs_imatest_mean": frame_spearman
        >= float(gates["minimum_frame_spearman_vs_imatest_mean"]),
        "roi_mean_absolute_error_cy_px": roi_mae
        <= float(gates["maximum_roi_mean_absolute_error_cy_px"]),
        "common_roi_mae_increase_vs_v1_cy_px": metrics[
            "common_roi_mae_change_vs_v1_cy_px"
        ]
        <= float(gates["maximum_common_roi_mae_increase_vs_v1_cy_px"]),
        "common_roi_spearman_decrease_vs_v1": metrics[
            "common_roi_spearman_change_vs_v1"
        ]
        >= -float(gates["maximum_common_roi_spearman_decrease_vs_v1"]),
    }
    return {
        "schema_version": "MERIT-C057-MTF50-V2-RAW-PARTITION-v1",
        "created_utc": utc_now(),
        "partition": partition,
        "estimator": {
            "localization_gaussian_sigma_px": float(results["localization_sigma_px"].iloc[0]),
            "polarity": "signed-gradient polarity with greater two-stage OLS line support",
            "line_fit": "two-stage ordinary least squares; first-stage absolute residual < 6 px",
            "esf_input": "unaltered RAW pixels",
        },
        "metrics": metrics,
        "gate_checks": checks,
        "gate": "PASS" if all(checks.values()) else "FAIL",
    }


def write_outputs(output: Path, results: pd.DataFrame, summary: dict[str, Any], *, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{summary['partition']}_roi_results.csv"
    json_path = output / f"{summary['partition']}_summary.json"
    for path in (csv_path, json_path):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    results.to_csv(temporary_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary_csv, csv_path)
    summary["artifacts"] = {
        "roi_results_csv": str(csv_path.resolve()),
        "roi_results_sha256": sha256_file(csv_path),
    }
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--execution-freeze", type=Path, default=DEFAULT_EXECUTION_FREEZE
    )
    subparsers = parser.add_subparsers(dest="partition", required=True)
    for partition, default_output in (
        ("development", DEFAULT_OUTPUT),
        ("holdout", Path(__file__).parent / "artifacts" / "holdout_v2"),
    ):
        subparser = subparsers.add_parser(partition)
        subparser.add_argument("--output", type=Path, default=default_output)
        subparser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract, frozen, cohort, partition_report, execution_freeze_hash = load_frozen_inputs(
        args.contract, args.freeze, args.execution_freeze
    )
    selected = frozen["selected_candidate"]
    sigma = float(selected["localization_gaussian_sigma_px"])
    subset = _partition(cohort, args.partition)
    results = evaluate_partition(subset, sigma)
    summary = summarize(results, contract, args.partition)
    summary["inputs"] = {
        "contract_sha256": sha256_file(args.contract),
        "freeze_sha256": sha256_file(args.freeze),
        "cohort_sha256": contract["inputs"]["cohort_sha256"],
        "estimator_code_sha256": sha256_file(Path(__file__).with_name("mtf50_estimator.py")),
        "runner_code_sha256": sha256_file(Path(__file__)),
        "execution_freeze_sha256": execution_freeze_hash,
    }
    summary["partition_validation"] = partition_report
    write_outputs(args.output, results, summary, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
