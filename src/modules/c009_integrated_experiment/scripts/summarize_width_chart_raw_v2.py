"""Audit the orientation-aware v2 RAW printed-width measurements."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import summarize_width_chart_raw as s1


MERIT_ROOT = Path(__file__).resolve().parents[3]
BASE_ARTIFACT_ROOT = (
    MERIT_ROOT / "40_수정본" / "c009_integrated_experiment" / "artifacts"
)
DEFAULT_ARTIFACT_DIR = BASE_ARTIFACT_ROOT / "width_raw_measurement_v2"
V1_INPUT = BASE_ARTIFACT_ROOT / "width_raw_measurement_v1" / "width_chart_raw_measurements_v1.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--v1-input", type=Path, default=V1_INPUT)
    args = parser.parse_args()
    input_path = args.artifact_dir / "width_chart_raw_measurements_v2.csv"
    frame_path = args.artifact_dir / "width_chart_raw_condition_summary_v2.csv"
    width_path = args.artifact_dir / "width_chart_raw_per_width_summary_v2.csv"
    summary_path = args.artifact_dir / "width_chart_raw_audit_v2.json"

    data = pd.read_csv(input_path)
    if len(data) != 500 or data["source_id"].nunique() != 50:
        raise RuntimeError("v2 ledger must contain 500 rows and 50 source frames")
    if not (data.groupby("source_id")["nominal_width_mm"].nunique() == 10).all():
        raise RuntimeError("Every source frame must contain ten unique nominal widths")
    if not (data.groupby("source_id")["trace_orientation"].nunique() == 1).all():
        raise RuntimeError("Orientation must be consistent across all ten widths in each frame")

    width_rows: list[dict[str, object]] = []
    for width, group in data.groupby("nominal_width_mm", sort=True):
        estimable = group[group["status"] == "ESTIMABLE"]
        counts = group["status"].value_counts().to_dict()
        width_rows.append(
            {
                "nominal_width_mm": float(width),
                "n_total": len(group),
                "n_estimable": len(estimable),
                "n_predeclared_subpixel_non_estimable": counts.get(
                    "PREDECLARED_SUBPIXEL_NON_ESTIMABLE", 0
                ),
                "n_qc_non_estimable": counts.get("QC_NON_ESTIMABLE", 0),
                "n_uncertainty_non_estimable": counts.get("UNCERTAINTY_NON_ESTIMABLE", 0),
                "estimable_fraction": len(estimable) / len(group),
                "apparent_width_median_mm": s1.finite_quantile(
                    estimable["apparent_width_mm_design_gsd"], 0.5
                ),
                "apparent_width_q1_mm": s1.finite_quantile(
                    estimable["apparent_width_mm_design_gsd"], 0.25
                ),
                "apparent_width_q3_mm": s1.finite_quantile(
                    estimable["apparent_width_mm_design_gsd"], 0.75
                ),
                "absolute_error_median_mm": s1.finite_quantile(
                    estimable["absolute_error_mm"], 0.5
                ),
                "relative_error_median": s1.finite_quantile(estimable["relative_error"], 0.5),
            }
        )
    pd.DataFrame(width_rows).to_csv(width_path, index=False, encoding="utf-8-sig")

    frame_rows: list[dict[str, object]] = []
    per_frame_rho: list[float] = []
    per_frame_slope: list[float] = []
    for source_id, group in data.groupby("source_id", sort=True):
        filename = str(group["filename"].iloc[0])
        speed, iso, distance = s1.parse_condition(filename)
        eligible = group[group["nominal_width_mm"] >= 0.3]
        estimable = eligible[eligible["status"] == "ESTIMABLE"].copy()
        counts = eligible["status"].value_counts().to_dict()
        rho: float | None = None
        slope: float | None = None
        if len(estimable) >= 6:
            rho = float(
                spearmanr(
                    estimable["nominal_width_mm"],
                    estimable["apparent_width_mm_design_gsd"],
                ).statistic
            )
            slope = float(
                np.polyfit(
                    estimable["nominal_width_mm"],
                    estimable["apparent_width_mm_design_gsd"],
                    1,
                )[0]
            )
            per_frame_rho.append(rho)
            per_frame_slope.append(slope)
        frame_rows.append(
            {
                "source_id": source_id,
                "filename": filename,
                "speed_kmh": speed,
                "iso": iso,
                "distance_m": distance,
                "trace_orientation": str(group["trace_orientation"].iloc[0]),
                "eligible_width_count": len(eligible),
                "estimable_count": len(estimable),
                "qc_non_estimable_count": counts.get("QC_NON_ESTIMABLE", 0),
                "uncertainty_non_estimable_count": counts.get(
                    "UNCERTAINTY_NON_ESTIMABLE", 0
                ),
                "all_eight_estimable": len(estimable) == 8,
                "within_frame_spearman_rho": rho,
                "within_frame_linear_slope": slope,
                "median_absolute_error_mm": s1.finite_quantile(
                    estimable["absolute_error_mm"], 0.5
                ),
                "median_profile_snr": s1.finite_quantile(estimable["median_profile_snr"], 0.5),
            }
        )
    frame_summary = pd.DataFrame(frame_rows)
    frame_summary.to_csv(frame_path, index=False, encoding="utf-8-sig")

    eligible = data[data["nominal_width_mm"] >= 0.3]
    estimable = eligible[eligible["status"] == "ESTIMABLE"]
    pooled = spearmanr(
        estimable["nominal_width_mm"], estimable["apparent_width_mm_design_gsd"]
    )
    vertical_frames = frame_summary[frame_summary["trace_orientation"] == "VERTICAL"][
        ["source_id", "filename", "speed_kmh", "iso", "distance_m"]
    ].to_dict(orient="records")
    non_estimable_rows = eligible[eligible["status"] != "ESTIMABLE"][
        ["source_id", "filename", "nominal_width_mm", "status"]
    ].to_dict(orient="records")

    v1_data = pd.read_csv(args.v1_input)
    comparison = v1_data.merge(
        data,
        on=["source_id", "filename", "nominal_width_mm"],
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )
    shared = comparison[
        (comparison["status_v1"] == "ESTIMABLE")
        & (comparison["status_v2"] == "ESTIMABLE")
        & (comparison["trace_orientation"] == "HORIZONTAL")
    ].copy()
    shared_delta = (
        pd.to_numeric(shared["apparent_width_mm_design_gsd_v2"])
        - pd.to_numeric(shared["apparent_width_mm_design_gsd_v1"])
    ).abs()
    transitions = (
        comparison.groupby(["status_v1", "status_v2"]).size().reset_index(name="n")
    ).to_dict(orient="records")

    audit = {
        "schema_version": "width-raw-audit-v2",
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "scope": "Descriptive audit of orientation-aware frozen RAW apparent-width measurements",
        "invariants": {
            "rows": len(data),
            "source_frames": int(data["source_id"].nunique()),
            "status_counts": {
                str(key): int(value) for key, value in data["status"].value_counts().items()
            },
            "orientation_counts_roi": {
                str(key): int(value)
                for key, value in data["trace_orientation"].value_counts().items()
            },
            "orientation_counts_frame": {
                str(key): int(value)
                for key, value in frame_summary["trace_orientation"].value_counts().items()
            },
            "orientation_consistent_within_frame": True,
        },
        "eligible_endpoint": {
            "definition": "nominal widths 0.3 to 1.0 mm",
            "rows": len(eligible),
            "estimable": len(estimable),
            "estimable_fraction": len(estimable) / len(eligible),
            "frames_with_any_estimable": int((frame_summary["estimable_count"] > 0).sum()),
            "frames_with_all_eight_estimable": int(frame_summary["all_eight_estimable"].sum()),
            "frames_with_no_estimable": int((frame_summary["estimable_count"] == 0).sum()),
        },
        "ordering_diagnostic": {
            "pooled_spearman_rho": float(pooled.statistic),
            "pooled_spearman_p": float(pooled.pvalue),
            "frames_with_at_least_six_estimable": len(per_frame_rho),
            "within_frame_rho_median": float(np.median(per_frame_rho)),
            "within_frame_rho_q1": float(np.quantile(per_frame_rho, 0.25)),
            "within_frame_rho_q3": float(np.quantile(per_frame_rho, 0.75)),
            "within_frame_positive_rho_count": int(sum(value > 0 for value in per_frame_rho)),
            "within_frame_slope_median": float(np.median(per_frame_slope)),
            "within_frame_slope_q1": float(np.quantile(per_frame_slope, 0.25)),
            "within_frame_slope_q3": float(np.quantile(per_frame_slope, 0.75)),
            "within_frame_positive_slope_count": int(sum(value > 0 for value in per_frame_slope)),
            "interpretation": "Nominal ordering is usually retained within a condition, but FWHM is compressed and offset by the acquisition PSF; this is a relative preservation endpoint, not direct physical-width metrology.",
        },
        "orientation_audit": {
            "vertical_frames": vertical_frames,
            "visual_contact_sheets": str((args.artifact_dir / "trace_qc").resolve()),
        },
        "remaining_non_estimable_rows": non_estimable_rows,
        "v1_to_v2_audit": {
            "status_transitions": transitions,
            "shared_horizontal_estimable_rows": len(shared),
            "shared_horizontal_max_absolute_difference_mm": float(shared_delta.max()),
            "interpretation": "Unchanged horizontal cases reproduce exactly; changes are confined to orientation handling and its downstream estimability.",
        },
        "claim_boundary": "RAW feasibility only; no restoration-effect conclusion",
        "inputs": {
            "measurements_csv": str(input_path.resolve()),
            "measurements_sha256": s1.sha256(input_path),
            "v1_measurements_csv": str(args.v1_input.resolve()),
            "v1_measurements_sha256": s1.sha256(args.v1_input),
        },
        "outputs": {
            "condition_summary_csv": str(frame_path.resolve()),
            "condition_summary_sha256": s1.sha256(frame_path),
            "per_width_summary_csv": str(width_path.resolve()),
            "per_width_summary_sha256": s1.sha256(width_path),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": s1.sha256(Path(__file__))},
    }
    summary_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["eligible_endpoint"], ensure_ascii=False, indent=2))
    print(json.dumps(audit["ordering_diagnostic"], ensure_ascii=False, indent=2))
    print(json.dumps(audit["v1_to_v2_audit"], ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
