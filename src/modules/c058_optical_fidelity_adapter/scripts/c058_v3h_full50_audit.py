"""Reaggregate the frozen v3h route over the complete exposed 50-condition grid.

This is a development-data audit.  It does not tune the already selected route
and it does not relabel exposed data as independent confirmation.  Active cells
reuse the frozen candidate measurements; every other cell is an exact RAW
bypass and therefore has a structural zero change in MTF50 and pixels.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SELECTION = BASE / "protocol/selected_safety_router_v3h_development.json"
AUDIT = BASE / "artifacts/safety_router_v3h_development/locator_v2_audit/exposed/frame_results.csv"
GRID = BASE / "artifacts/safety_router_v3h_development/candidate_grid_v1"
OUTPUT = BASE / "artifacts/safety_router_v3h_development/full50_condition_audit"
EXPECTED_ROUTE = "safe_s1_am0.75"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def bootstrap_ci(values: np.ndarray, count: int = 10000, seed: int = 20261021) -> list[float]:
    values = np.sort(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def active_condition_metrics(route: str) -> pd.DataFrame:
    by_source = pd.read_csv(GRID / "condition_by_source.csv")
    rows = by_source[
        (by_source["run_key"] == route) & (by_source["source_cohort"] == "exposed")
    ].copy()
    if rows["condition_cell"].duplicated().any():
        raise RuntimeError("duplicate exposed active-condition result")

    roi = pd.read_csv(GRID / "roi_results.csv")
    keys = ["frame_id", "condition_cell", "source_cohort", "distance_m", "iso", "roi_id"]
    raw = roi[
        (roi["source_cohort"] == "exposed")
        & (roi["run_key"] == "raw_none")
        & (roi["valid"] == True)  # noqa: E712
        & (roi["is_reference"] != True)  # noqa: E712
    ][keys].drop_duplicates()
    routed = roi[
        (roi["source_cohort"] == "exposed")
        & (roi["run_key"] == route)
        & (roi["valid"] == True)  # noqa: E712
    ][keys].drop_duplicates()
    common = raw.merge(routed, on=keys, how="inner")
    roi_counts = common.groupby("condition_cell").size().rename("common_valid_rois")
    frame_counts = common.groupby("condition_cell")["frame_id"].nunique().rename("evaluated_frames")

    diagnostics = pd.read_csv(GRID / "pixel_diagnostics.csv")
    diagnostics = diagnostics[
        (diagnostics["source_cohort"] == "exposed") & (diagnostics["run_key"] == route)
    ]
    diagnostic_summary = diagnostics.groupby("condition_cell").agg(
        maximum_introduced_clipping_fraction=("introduced_clipping_fraction", "max"),
        maximum_unprojected_overshoot_fraction=("unprojected_preclip_overshoot_fraction", "max"),
        maximum_headroom_limited_fraction=("headroom_limited_fraction", "max"),
        maximum_abs_change_dn=("max_abs_change_dn", "max"),
    )

    return (
        rows.set_index("condition_cell")
        .join(roi_counts)
        .join(frame_counts)
        .join(diagnostic_summary)
        .reset_index()
    )


def build_condition_table() -> tuple[pd.DataFrame, dict]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    route = str(selection["selected_route"]["run_key"])
    if route != EXPECTED_ROUTE:
        raise RuntimeError(f"unexpected selected route: {route}")

    frame_audit = pd.read_csv(AUDIT)
    frame_counts = frame_audit.groupby("condition").size().rename("available_frames")
    strict_counts = (
        frame_audit[frame_audit["effective_eligible"] == True]  # noqa: E712
        .groupby("condition")
        .size()
        .rename("strict_full_chart_frames")
    )
    table = pd.DataFrame(index=sorted(frame_counts.index)).join(frame_counts).join(strict_counts)
    table["strict_full_chart_frames"] = table["strict_full_chart_frames"].fillna(0).astype(int)
    table["speed_kmh"] = table.index.str.extract(r"^(60|80)km", expand=False).astype(int)
    table["distance_m"] = table.index.str.extract(r"_(2\.5|3\.5|4\.5|5\.5|6\.5)m_", expand=False).astype(float)
    table["iso"] = table.index.str.extract(r"ISO(100|200|400|800|1600)$", expand=False).astype(int)
    table["route_active"] = (
        (table["speed_kmh"] == 80)
        & table["distance_m"].isin([2.5, 3.5, 4.5])
        & table["iso"].isin([100, 200])
    )
    table["route"] = np.where(table["route_active"], route, "raw_exact_bypass")
    table["evaluated_frames"] = np.where(table["route_active"], np.nan, table["available_frames"])
    table["common_valid_rois"] = np.where(table["route_active"], np.nan, 0)
    table["delta_abs_error_cy_px"] = np.where(table["route_active"], np.nan, 0.0)
    table["maximum_unprojected_overshoot_fraction"] = 0.0
    table["maximum_introduced_clipping_fraction"] = 0.0
    table["maximum_headroom_limited_fraction"] = 0.0
    table["maximum_abs_change_dn"] = 0.0
    table["result"] = np.where(table["route_active"], "pending", "preserved_exact_raw_bypass")
    table["evidence_basis"] = np.where(
        table["route_active"],
        "matched_valid_chart_roi_measurement",
        "pixel-identical_router_bypass",
    )

    active = active_condition_metrics(route).set_index("condition_cell")
    if set(active.index) != set(table.index[table["route_active"]]):
        raise RuntimeError("active-condition measurements do not cover the frozen active grid")
    for condition, row in active.iterrows():
        table.loc[condition, "evaluated_frames"] = int(row["evaluated_frames"])
        table.loc[condition, "common_valid_rois"] = int(row["common_valid_rois"])
        table.loc[condition, "delta_abs_error_cy_px"] = float(row["delta_abs_error"])
        table.loc[condition, "maximum_unprojected_overshoot_fraction"] = float(
            row["maximum_unprojected_overshoot_fraction"]
        )
        table.loc[condition, "maximum_introduced_clipping_fraction"] = float(
            row["maximum_introduced_clipping_fraction"]
        )
        table.loc[condition, "maximum_headroom_limited_fraction"] = float(
            row["maximum_headroom_limited_fraction"]
        )
        table.loc[condition, "maximum_abs_change_dn"] = float(row["maximum_abs_change_dn"])
        table.loc[condition, "result"] = (
            "improved" if float(row["delta_abs_error"]) < 0 else "worsened_or_tied"
        )

    table["evaluated_frames"] = table["evaluated_frames"].astype(int)
    table["common_valid_rois"] = table["common_valid_rois"].astype(int)
    table.index.name = "condition_cell"
    table = table.reset_index()

    deltas = table["delta_abs_error_cy_px"].to_numpy(dtype=float)
    summary = {
        "schema_version": "MERIT-C058-V3H-FULL50-DEVELOPMENT-AUDIT-v1",
        "state": "COMPLETE_DEVELOPMENT_EVIDENCE",
        "evidence_label": "development_only_exposed_479",
        "selection_path": str(SELECTION.relative_to(BASE)).replace("\\", "/"),
        "selection_sha256": sha256(SELECTION),
        "selected_route": route,
        "condition_cells": int(len(table)),
        "available_frames": int(table["available_frames"].sum()),
        "active_condition_cells": int(table["route_active"].sum()),
        "bypass_condition_cells": int((~table["route_active"]).sum()),
        "active_available_frames": int(table.loc[table["route_active"], "available_frames"].sum()),
        "active_evaluated_chart_frames": int(table.loc[table["route_active"], "evaluated_frames"].sum()),
        "active_common_valid_rois": int(table.loc[table["route_active"], "common_valid_rois"].sum()),
        "bypass_available_frames": int(table.loc[~table["route_active"], "available_frames"].sum()),
        "improved_condition_cells": int((table["result"] == "improved").sum()),
        "preserved_exact_bypass_condition_cells": int(
            (table["result"] == "preserved_exact_raw_bypass").sum()
        ),
        "worsened_condition_cells": int((deltas > 0).sum()),
        "nonworse_condition_cells": int((deltas <= 0).sum()),
        "full50_mean_delta_abs_error_cy_px": float(deltas.mean()),
        "full50_condition_bootstrap_95pct_ci": bootstrap_ci(deltas),
        "active_mean_delta_abs_error_cy_px": float(
            table.loc[table["route_active"], "delta_abs_error_cy_px"].mean()
        ),
        "maximum_unprojected_overshoot_fraction": float(
            table["maximum_unprojected_overshoot_fraction"].max()
        ),
        "maximum_introduced_clipping_fraction": float(
            table["maximum_introduced_clipping_fraction"].max()
        ),
        "maximum_headroom_limited_fraction": float(
            table["maximum_headroom_limited_fraction"].max()
        ),
        "maximum_abs_change_dn": float(table["maximum_abs_change_dn"].max()),
        "interpretation": (
            "Six frozen active cells are measured on matched valid chart ROIs; "
            "the other 44 cells are exact RAW bypasses and therefore structural ties."
        ),
        "independent_confirmation": False,
    }
    summary["gate_checks"] = {
        "all_50_condition_cells_present": len(table) == 50,
        "all_479_exposed_frames_accounted": int(table["available_frames"].sum()) == 479,
        "six_active_condition_cells": int(table["route_active"].sum()) == 6,
        "forty_four_bypass_condition_cells": int((~table["route_active"]).sum()) == 44,
        "all_active_conditions_improved": bool(
            (table.loc[table["route_active"], "delta_abs_error_cy_px"] < 0).all()
        ),
        "all_bypass_conditions_exact_ties": bool(
            (table.loc[~table["route_active"], "delta_abs_error_cy_px"] == 0).all()
        ),
        "all_conditions_nonworse": bool((deltas <= 0).all()),
        "full50_bootstrap_ci_upper_negative": summary["full50_condition_bootstrap_95pct_ci"][1] < 0,
        "introduced_clipping_zero": summary["maximum_introduced_clipping_fraction"] == 0,
        "unprojected_overshoot_zero": summary["maximum_unprojected_overshoot_fraction"] == 0,
        "maximum_change_le_4dn": summary["maximum_abs_change_dn"] <= 4,
    }
    if not all(summary["gate_checks"].values()):
        raise RuntimeError(f"full-50 audit gate failed: {summary['gate_checks']}")
    return table, summary


def render_report(table: pd.DataFrame, summary: dict) -> str:
    active = table[table["route_active"]]
    active_rows = "\n".join(
        f"| {row.condition_cell} | {row.evaluated_frames} | {row.common_valid_rois} | "
        f"{row.delta_abs_error_cy_px:+.10f} | 개선 |"
        for row in active.itertuples(index=False)
    )
    ci = summary["full50_condition_bootstrap_95pct_ci"]
    return f"""# 1. C058 v3h 전체 50조건 개발 감사

## 1.1 판정

동결된 v3h를 공개 479장·50조건 전체에 조건 단위로 재집계했다. 활성 6조건은 모두 MTF50 절대오차가 감소했고, 비활성 44조건은 RAW exact bypass로 픽셀과 MTF50 변화가 0이다. 따라서 전체 조건 판정은 **6개 개선, 44개 동일, 0개 악화**다. 이 결과는 개발자료 감사이며 독립 확증은 아니다.

## 1.2 전체 결과

- 전체 조건: **50/50**
- 전체 프레임 회계: **479/479장**
- 활성 조건: **6개**, 유효 차트 프레임 **{summary['active_evaluated_chart_frames']}장**, 공통 유효 ROI **{summary['active_common_valid_rois']}개**
- RAW exact bypass: **44개 조건**, **{summary['bypass_available_frames']}장**
- 조건 판정: **6개 개선 + 44개 동일 + 0개 악화**
- 전체 50조건 평균 Δ절대오차: **{summary['full50_mean_delta_abs_error_cy_px']:+.10f} cy/px**
- 조건 부트스트랩 95% CI: **[{ci[0]:+.10f}, {ci[1]:+.10f}]**
- 활성 6조건 평균 Δ절대오차: **{summary['active_mean_delta_abs_error_cy_px']:+.10f} cy/px**
- 최대 preclip overshoot: **0%**
- 최대 신규 clipping: **0%**
- 최대 픽셀 변화: **{summary['maximum_abs_change_dn']:.0f} DN**

## 1.3 활성 조건 결과

| 조건 | 유효 프레임 | 공통 ROI | Δ 절대오차 (cy/px) | 판정 |
|---|---:|---:|---:|---|
{active_rows}

음수는 RAW보다 거리별 기준 MTF50에 가까워졌다는 뜻이다.

## 1.4 비활성 44조건의 해석

비활성 조건에서는 복원 출력을 혼합하지 않고 입력 RAW를 그대로 반환한다. 따라서 44조건은 성능 향상으로 계산하지 않고 모두 **정확한 동률**로 처리했다. 전체 조건을 포함하되, 우회 조건을 개선으로 부풀리지 않았다.

## 1.5 증거 경계

이 감사는 이미 개발에 노출된 479장으로 수행했으며 선택된 파라미터를 변경하지 않았다. 그러므로 R2-1의 전체 조건 개발 검증에는 사용할 수 있지만 독립 광학 확증으로 기술할 수 없다. 조건별 원자료 회계와 판정은 `condition_results.csv`에 보존한다.
"""


def main() -> None:
    table, summary = build_condition_table()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "condition_results.csv"
    json_path = OUTPUT / "summary.json"
    report_path = OUTPUT / "FULL50_CONDITION_REPORT.md"
    table.to_csv(csv_path, index=False)
    write_atomic(json_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_atomic(report_path, render_report(table, summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
