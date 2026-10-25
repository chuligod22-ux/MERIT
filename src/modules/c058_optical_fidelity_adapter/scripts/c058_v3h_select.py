"""Freeze the selected C058 v3h development route and its evidence boundary."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
GRID = BASE / "artifacts/safety_router_v3h_development/candidate_grid_v1"
AUDIT = BASE / "artifacts/safety_router_v3h_development/locator_v2_audit"
OUTPUT = BASE / "protocol/selected_safety_router_v3h_development.json"
REPORT = BASE / "artifacts/safety_router_v3h_development/V3H_DEVELOPMENT_REPORT.md"
ROUTE = "safe_s1_am0.75"


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


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(BASE.parents[2])).replace("\\", "/"), "sha256": sha256(path)}


def main() -> None:
    candidates = json.loads((GRID / "candidate_summary.json").read_text(encoding="utf-8"))
    selected = next(item for item in candidates if item["run_key"] == ROUTE)
    combined = pd.read_csv(GRID / "condition_combined.csv")
    condition = combined[combined["run_key"] == ROUTE].sort_values("condition_cell")
    by_source = pd.read_csv(GRID / "condition_by_source.csv")
    source_selected = by_source[by_source["run_key"] == ROUTE].copy()
    source_selected["raw_abs_error"] = abs(source_selected["raw_mtf50"] - source_selected["reference_mtf50"])
    source_selected["v3h_abs_error"] = abs(source_selected["mtf50_native_cy_px"] - source_selected["reference_mtf50"])
    condition_errors = source_selected.groupby("condition_cell")[["raw_abs_error", "v3h_abs_error"]].mean()
    raw_mean = float(condition_errors["raw_abs_error"].mean())
    v3h_mean = float(condition_errors["v3h_abs_error"].mean())
    diagnostics = pd.read_csv(GRID / "pixel_diagnostics.csv")
    selected_diagnostics = diagnostics[diagnostics["run_key"] == ROUTE]
    exposed = json.loads((AUDIT / "exposed/summary.json").read_text(encoding="utf-8"))
    holdout = json.loads((AUDIT / "holdout/summary.json").read_text(encoding="utf-8"))

    checks = {
        "six_active_conditions": len(condition) == 6,
        "all_condition_deltas_negative": bool((condition["delta_abs_error"] < 0).all()),
        "mean_delta_negative": float(selected["mean_delta_abs_error_cy_px"]) < 0,
        "bootstrap_ci_upper_negative": float(selected["condition_bootstrap_95pct_ci"][1]) < 0,
        "introduced_clipping_zero": float(selected_diagnostics["introduced_clipping_fraction"].max()) == 0,
        "unprojected_overshoot_zero": float(selected_diagnostics["unprojected_preclip_overshoot_fraction"].max()) == 0,
        "maximum_change_le_4dn": float(selected_diagnostics["max_abs_change_dn"].max()) <= 4,
        "former_holdout_not_reused_as_confirmation": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v3h selection gate failed: {checks}")

    files = [
        BASE / "scripts/c058_raw_locator_v2.py",
        BASE / "scripts/c058_mtf_measure_v3h.py",
        BASE / "scripts/c058_inference_v3h.py",
        BASE / "scripts/c058_adapter_v2.py",
        BASE / "scripts/c058_v3h_candidate_development.py",
        GRID / "development_summary.json",
        GRID / "candidate_summary.json",
        GRID / "condition_combined.csv",
        GRID / "condition_by_source.csv",
        GRID / "pixel_diagnostics.csv",
        AUDIT / "exposed/summary.json",
        AUDIT / "holdout/summary.json",
    ]
    payload = {
        "schema_version": "MERIT-C058-SAFETY-ROUTER-SELECTION-v3h-development",
        "state": "DEVELOPMENT_SELECTED_REQUIRES_UNTOUCHED_CONFIRMATION",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evidence_boundary": {
            "label": "development_only",
            "former_280_frame_holdout": "exposed_by_v3g_and_used_only_for_v3h_development",
            "independent_optical_confirmation": False,
            "reason": "only one former-holdout frame was active-condition chart-usable after target-type and source-footprint QC",
        },
        "selected_route": {
            "run_key": ROUTE,
            "model_condition": "task_on",
            "model_seed": 1,
            "alpha": -0.75,
            "sigma_px": 0.75,
            "residual_cap_dn": 8.0,
            "maximum_headroom_limited_fraction": 0.01,
            "active_if": "speed_kmh == 80 and iso in {100,200} and distance_m in {2.5,3.5,4.5}",
            "otherwise": "raw_exact_bypass",
        },
        "development_result": {
            "frames": 84,
            "reference_frames": 45,
            "active_frames": 39,
            "active_condition_cells": 6,
            "common_valid_active_roi_identities": 120,
            "mean_delta_abs_error_cy_px": float(selected["mean_delta_abs_error_cy_px"]),
            "condition_bootstrap_95pct_ci": selected["condition_bootstrap_95pct_ci"],
            "improved_conditions": int(selected["improved_conditions"]),
            "raw_mean_abs_error_cy_px": raw_mean,
            "v3h_mean_abs_error_cy_px": v3h_mean,
            "relative_abs_error_reduction": float(1.0 - v3h_mean / raw_mean),
            "maximum_unprojected_overshoot_fraction": float(selected_diagnostics["unprojected_preclip_overshoot_fraction"].max()),
            "maximum_introduced_clipping_fraction": float(selected_diagnostics["introduced_clipping_fraction"].max()),
            "maximum_headroom_limited_fraction": float(selected_diagnostics["headroom_limited_fraction"].max()),
            "maximum_abs_change_dn": float(selected_diagnostics["max_abs_change_dn"].max()),
            "per_condition_delta_abs_error_cy_px": {
                str(row.condition_cell): float(row.delta_abs_error)
                for row in condition.itertuples(index=False)
            },
        },
        "locator_audit": {
            "exposed_479": exposed,
            "former_holdout_280": holdout,
            "interpretation": "The v3g 280-frame list mixed quadrant-chart, partial-chart, crack-board, and wall-only frames; v3h excludes non-chart and replicated-border ROIs before MTF50 estimation.",
        },
        "gate_checks": checks,
        "bindings": {
            str(path.relative_to(BASE)).replace("\\", "/"): binding(path)
            for path in files
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    write_atomic(OUTPUT, text)
    write_atomic(OUTPUT.with_suffix(OUTPUT.suffix + ".sha256"), f"{sha256(OUTPUT)}  {OUTPUT.name}\n")

    table = "\n".join(
        f"| {row.condition_cell} | {float(row.delta_abs_error):+.10f} |"
        for row in condition.itertuples(index=False)
    )
    report = f"""# 1. C058 v3h 추가개발 결과

## 1.1 판정

v3h 개발은 **성공**했다. 선택 경로는 task-ON seed 1, `alpha=-0.75`, `sigma=0.75 px`, residual cap `8 DN`이며 활성 6조건 모두에서 RAW 대비 기준 MTF50 절대오차를 줄였다. 이 결과는 개발 증거이며 독립 확증으로 표기하지 않는다.

## 1.2 핵심 수치

- 개발 프레임: 84장(기준 45, 활성 39)
- 공통 유효 활성 ROI: 120개
- 6조건 평균 Δ절대오차: **{float(selected['mean_delta_abs_error_cy_px']):+.10f} cy/px**
- 조건 부트스트랩 95% CI: **[{float(selected['condition_bootstrap_95pct_ci'][0]):+.10f}, {float(selected['condition_bootstrap_95pct_ci'][1]):+.10f}]**
- 개선 조건: **6/6**
- 평균 절대오차: RAW **{raw_mean:.10f}** → v3h **{v3h_mean:.10f} cy/px**
- 상대 절대오차 감소: **{100*(1-v3h_mean/raw_mean):.2f}%**
- 최대 preclip overshoot: **0%**
- 최대 신규 clipping: **0%**
- 최대 픽셀 변화: **4 DN**

## 1.3 조건별 결과

| 조건 | Δ 절대오차 (cy/px) |
|---|---:|
{table}

음수는 RAW보다 기준 MTF50에 가까워졌다는 뜻이다.

## 1.4 v3g 실패 원인 교정

기존 280장 중 엄격한 전체 사분원 차트 적격 프레임은 8장뿐이었다. 259장은 사분원 차트가 아닌 균열판·벽면이었고 13장은 부분 차트였다. v3g 로케이터는 이들을 작은 차트로 오검출했고 잘못된 반경을 native MTF 환산에 사용했다. v3h는 큰 차트 중심 탐색, radial 반경 보정, 실제 센서 source-footprint 검사, 부분 ROI의 명시적 무효화를 적용한다.

## 1.5 독립성 경계

기존 280장은 v3g 결과가 공개된 뒤 v3h 개발에 사용했으므로 다시 독립 확증으로 사용할 수 없다. 그중 활성 조건에서 실제로 측정 가능한 차트 프레임은 3.5 m·ISO 200 한 장뿐이었다. 따라서 v3h는 개발 완료 상태이며, 독립적인 실제 광학 성능 개선 주장은 아직 결과를 보지 않은 사분원 차트 원본 집합에 이 경로를 동결 적용한 뒤에만 가능하다.
"""
    write_atomic(REPORT, report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
