"""Audit and summarize the frozen RAW printed-width measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
MERIT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = (
    MERIT_ROOT
    / "40_수정본"
    / "c009_integrated_experiment"
    / "artifacts"
    / "width_raw_measurement_v1"
)
FILENAME_PATTERN = re.compile(
    r"^cam1_v(?P<speed>60|80)_iso(?P<iso>100|200|400|800|1600)_d(?P<distance>25|35|45|55|65)(?:_x)?\.png$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_condition(filename: str) -> tuple[int, int, float]:
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Unrecognized filename: {filename}")
    return (
        int(match.group("speed")),
        int(match.group("iso")),
        int(match.group("distance")) / 10.0,
    )


def finite_quantile(values: pd.Series, probability: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(probability)) if len(clean) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    input_path = args.artifact_dir / "width_chart_raw_measurements_v1.csv"
    frame_path = args.artifact_dir / "width_chart_raw_condition_summary_v1.csv"
    width_path = args.artifact_dir / "width_chart_raw_per_width_summary_v1.csv"
    summary_path = args.artifact_dir / "width_chart_raw_audit_v1.json"

    data = pd.read_csv(input_path)
    if len(data) != 500 or data["source_id"].nunique() != 50:
        raise RuntimeError("RAW measurement ledger must contain 500 rows and 50 source frames")
    if not (data.groupby("source_id")["nominal_width_mm"].nunique() == 10).all():
        raise RuntimeError("Every source frame must contain ten unique nominal widths")

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
                "apparent_width_median_mm": finite_quantile(
                    estimable["apparent_width_mm_design_gsd"], 0.5
                ),
                "apparent_width_q1_mm": finite_quantile(
                    estimable["apparent_width_mm_design_gsd"], 0.25
                ),
                "apparent_width_q3_mm": finite_quantile(
                    estimable["apparent_width_mm_design_gsd"], 0.75
                ),
                "absolute_error_median_mm": finite_quantile(
                    estimable["absolute_error_mm"], 0.5
                ),
                "relative_error_median": finite_quantile(estimable["relative_error"], 0.5),
            }
        )
    width_summary = pd.DataFrame(width_rows)
    width_summary.to_csv(width_path, index=False, encoding="utf-8-sig")

    frame_rows: list[dict[str, object]] = []
    per_frame_rho: list[float] = []
    per_frame_slope: list[float] = []
    for source_id, group in data.groupby("source_id", sort=True):
        filename = str(group["filename"].iloc[0])
        speed, iso, distance = parse_condition(filename)
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
                "eligible_width_count": len(eligible),
                "estimable_count": len(estimable),
                "qc_non_estimable_count": counts.get("QC_NON_ESTIMABLE", 0),
                "uncertainty_non_estimable_count": counts.get(
                    "UNCERTAINTY_NON_ESTIMABLE", 0
                ),
                "all_eight_estimable": len(estimable) == 8,
                "within_frame_spearman_rho": rho,
                "within_frame_linear_slope": slope,
                "median_absolute_error_mm": finite_quantile(estimable["absolute_error_mm"], 0.5),
                "median_profile_snr": finite_quantile(estimable["median_profile_snr"], 0.5),
            }
        )
    frame_summary = pd.DataFrame(frame_rows)
    frame_summary.to_csv(frame_path, index=False, encoding="utf-8-sig")

    estimable = data[data["status"] == "ESTIMABLE"]
    eligible = data[data["nominal_width_mm"] >= 0.3]
    pooled = spearmanr(
        estimable["nominal_width_mm"], estimable["apparent_width_mm_design_gsd"]
    )
    failed_frames = frame_summary[frame_summary["estimable_count"] == 0][
        ["source_id", "filename", "speed_kmh", "iso", "distance_m"]
    ].to_dict(orient="records")
    partial_frames = frame_summary[
        (frame_summary["estimable_count"] > 0) & (frame_summary["estimable_count"] < 8)
    ][["source_id", "filename", "estimable_count"]].to_dict(orient="records")

    status_counts = {str(key): int(value) for key, value in data["status"].value_counts().items()}
    audit = {
        "schema_version": "width-raw-audit-v1",
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "scope": "Descriptive audit of frozen RAW apparent-width measurements",
        "invariants": {
            "rows": len(data),
            "source_frames": int(data["source_id"].nunique()),
            "nominal_widths": sorted(float(value) for value in data["nominal_width_mm"].unique()),
            "status_counts": status_counts,
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
            "within_frame_positive_slope_count": int(
                sum(value > 0 for value in per_frame_slope)
            ),
            "interpretation": "Nominal ordering is usually retained within a condition, but FWHM is compressed and offset by the acquisition PSF; this is a relative preservation endpoint, not direct physical-width metrology.",
        },
        "failed_frames": failed_frames,
        "partial_frames": partial_frames,
        "claim_boundary": "RAW feasibility only; no restoration-effect conclusion",
        "inputs": {"measurements_csv": str(input_path.resolve()), "sha256": sha256(input_path)},
        "outputs": {
            "condition_summary_csv": str(frame_path.resolve()),
            "condition_summary_sha256": sha256(frame_path),
            "per_width_summary_csv": str(width_path.resolve()),
            "per_width_summary_sha256": sha256(width_path),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
    }
    summary_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["eligible_endpoint"], ensure_ascii=False, indent=2))
    print(json.dumps(audit["ordering_diagnostic"], ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
