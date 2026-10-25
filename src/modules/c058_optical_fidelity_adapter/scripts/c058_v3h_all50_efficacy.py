"""Apply the frozen C058 v3h route to every one of the 50 acquisition cells.

The earlier full-grid audit applied the route to six preselected cells and
returned exact RAW in the other 44.  This analysis removes that condition
router for evaluation only: the already selected task-ON seed and adapter are
run on all 479 frames.  No parameter is tuned from these outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve()
TEST = BASE / "tests/test_c058_v3h_all50_efficacy.py"
SELECTION = BASE / "protocol/selected_safety_router_v3h_development.json"
FREEZE = BASE / "protocol/all50_efficacy_execution_freeze_v1.json"
OUTPUT = BASE / "artifacts/safety_router_v3h_development/all50_efficacy_v1"
C057_RUNNER = BASE.parent / "c057_mtf50_estimator_v2/run_restoration_mtf50_v2.py"
C057_FREEZE = BASE.parent / "c057_mtf50_estimator_v2/protocol/mtf50_v2_restoration_execution_freeze.json"
COHORT = BASE.parent / "c007_d0_condition_eval/artifacts/cam1_v1/d0_cam1_cohort.csv"
ADAPTER = BASE / "scripts/c058_adapter_v2.py"
INFERENCE = BASE / "scripts/c058_inference_v3h.py"
SCHEMA = "MERIT-C058-V3H-ALL50-EFFICACY-v1"
FREEZE_SCHEMA = "MERIT-C058-V3H-ALL50-EFFICACY-FREEZE-v1"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c057 = load("c058_all50_c057", C057_RUNNER)
adapter = load("c058_all50_adapter", ADAPTER)
inference = load("c058_all50_inference", INFERENCE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def selected_route() -> dict:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    route = dict(selection["selected_route"])
    expected = {
        "run_key": "safe_s1_am0.75",
        "model_condition": "task_on",
        "model_seed": 1,
        "alpha": -0.75,
        "sigma_px": 0.75,
        "residual_cap_dn": 8.0,
        "maximum_headroom_limited_fraction": 0.01,
    }
    for key, value in expected.items():
        if route.get(key) != value:
            raise RuntimeError(f"selected route changed at {key}: {route.get(key)!r}")
    return route


def freeze_inputs() -> dict[str, Path]:
    return {
        "script": SCRIPT,
        "tests": TEST,
        "selection": SELECTION,
        "c057_runner": C057_RUNNER,
        "c057_restoration_freeze": C057_FREEZE,
        "c057_cohort": COHORT,
        "adapter": ADAPTER,
        "inference": INFERENCE,
    }


def create_freeze() -> dict:
    if (OUTPUT / "progress").exists() or (OUTPUT / "summary.json").exists():
        raise RuntimeError("all-50 outputs already exist; refuse to freeze after results")
    route = selected_route()
    _, _, cohort, c057_hash = c057.load_verified_environment(C057_FREEZE)
    if len(cohort) != 1737 or cohort["frame_id"].nunique() != 479 or cohort["condition"].nunique() != 50:
        raise RuntimeError("frozen C057 cohort count changed")
    paths = freeze_inputs()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing freeze inputs: {missing}")
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "state": "FROZEN_BEFORE_ALL50_OUTPUTS",
        "created_utc": utc_now(),
        "purpose": "Apply one unchanged frozen v3h route to all 50 condition cells without condition-level bypass.",
        "analysis_unit": "condition_cell; frames and ROIs are clustered within condition",
        "condition_independence": "author-designated 50 distinct acquisition-condition cells",
        "evidence_label": "existing-acquisition all-condition analysis; no parameter tuning from outputs",
        "route": route,
        "endpoint": {
            "conditions": "all 2 speeds x 5 distances x 5 ISO cells",
            "reference": "distance-specific 60 km/h ISO100 RAW median MTF50",
            "metric": "routed-minus-RAW absolute MTF50 error; negative is improvement",
            "reference_cells": "included as preservation tests",
            "aggregation": "paired valid ROI mean per frame, frame median per condition, condition-cell bootstrap",
            "classification": "negative improved; zero tied; positive worsened",
        },
        "expected_counts": {"condition_cells": 50, "frames": 479, "roi_identities": 1737},
        "policies": [
            "Run task-ON seed 1 and alpha=-0.75, sigma=0.75 px, cap=8 DN unchanged in every cell.",
            "Do not tune, drop, or activate conditions after output inspection.",
            "Preserve improved, tied, worsened, safety-bypassed, invalid, and missing results.",
            "Use identical C057-v2 ROI identities and estimator for RAW and routed output.",
            "Do not interpret chart MTF50 as natural-crack truth or causal proof of downstream efficacy.",
        ],
        "c057_restoration_freeze_sha256": c057_hash,
        "files": {
            label: {"path": relative(path), "sha256": sha256(path)}
            for label, path in paths.items()
        },
    }
    write_json_atomic(FREEZE, payload)
    sidecar_path(FREEZE).write_text(f"{sha256(FREEZE)}  {FREEZE.name}\n", encoding="ascii")
    return payload


def load_freeze() -> tuple[dict, str]:
    if not FREEZE.is_file() or not sidecar_path(FREEZE).is_file():
        raise RuntimeError("all-50 execution freeze is missing")
    freeze_hash = sha256(FREEZE)
    fields = sidecar_path(FREEZE).read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[0] != freeze_hash or fields[1] != FREEZE.name:
        raise RuntimeError("all-50 execution freeze sidecar mismatch")
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FREEZE_SCHEMA or payload.get("state") != "FROZEN_BEFORE_ALL50_OUTPUTS":
        raise RuntimeError("all-50 execution freeze state changed")
    if payload["route"] != selected_route():
        raise RuntimeError("all-50 frozen route differs from selected v3h route")
    expected = freeze_inputs()
    if set(payload["files"]) != set(expected):
        raise RuntimeError("all-50 freeze file labels changed")
    for label, path in expected.items():
        record = payload["files"][label]
        if record["path"] != relative(path) or record["sha256"] != sha256(path):
            raise RuntimeError(f"all-50 freeze input mismatch: {label}")
    return payload, freeze_hash


def progress_path(index: int, frame_id: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", frame_id)
    return OUTPUT / "progress" / f"{index:04d}_{safe_name}.json"


def validate_progress(path: Path, frame: pd.DataFrame, freeze_hash: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA or payload.get("execution_freeze_sha256") != freeze_hash:
        raise RuntimeError(f"progress freeze/schema mismatch: {path}")
    if payload.get("frame_id") != str(frame.iloc[0]["frame_id"]):
        raise RuntimeError(f"progress frame mismatch: {path}")
    expected = [(int(row.roi_id), str(row.orientation)) for row in frame.sort_values("roi_id").itertuples()]
    for key in ("raw_rois", "routed_rois"):
        observed = [(int(row["roi_id"]), str(row["orientation"])) for row in payload[key]]
        if observed != expected:
            raise RuntimeError(f"progress ROI mismatch: {path} {key}")


def run_inference() -> dict:
    frozen, freeze_hash = load_freeze()
    protocol, _, cohort, c057_hash = c057.load_verified_environment(C057_FREEZE)
    if c057_hash != frozen["c057_restoration_freeze_sha256"]:
        raise RuntimeError("C057 freeze differs from all-50 contract")
    frames = c057.D0.frame_records(cohort)
    for ordinal, (_, frame) in enumerate(frames, start=1):
        path = progress_path(ordinal, str(frame.iloc[0]["frame_id"]))
        if path.is_file():
            validate_progress(path, frame, freeze_hash)
    device = c057.D0.require_cuda()
    models, tiled_restore = c057.D0.load_models(protocol, device)
    route = frozen["route"]
    model = models[("on", str(route["model_seed"]))]
    config = protocol["restoration"]
    tiling = config["tiling"]
    candidate = adapter.SafeAdapterCandidate(
        route_id=str(route["run_key"]),
        alpha=float(route["alpha"]),
        sigma=float(route["sigma_px"]),
        cap_dn=float(route["residual_cap_dn"]),
        maximum_headroom_limited_fraction=float(route["maximum_headroom_limited_fraction"]),
    )
    completed = 0
    for ordinal, (_, frame) in enumerate(frames, start=1):
        frame_id = str(frame.iloc[0]["frame_id"])
        target = progress_path(ordinal, frame_id)
        if target.is_file():
            completed += 1
            continue
        started = time.perf_counter()
        image = c057.D0.verify_row_assets(frame)
        raw_rois = c057.measure_rois(image, frame)
        crop, x0, y0 = c057.D0.union_context(image, frame, int(config["context_margin_px"]))
        task_on = inference.tiled_restore_native_u8(
            crop,
            model,
            device,
            tiled_restore,
            tile_size=int(tiling["tile_size"]),
            overlap=int(tiling["overlap"]),
            batch_size=int(tiling["batch_size"]),
            blend_floor=float(tiling["positive_hann_floor"]),
            amp_dtype=str(config["amp_dtype"]),
        )
        routed, diagnostics = adapter.apply_safe_adapter(crop, task_on, candidate)
        routed_rois = c057.measure_rois(routed, frame, x_offset=x0, y_offset=y0)
        payload = {
            "schema_version": SCHEMA,
            "execution_freeze_sha256": freeze_hash,
            "ordinal": ordinal,
            "frame_id": frame_id,
            "image_sha256": str(frame.iloc[0]["image_sha256"]),
            "condition_cell": str(frame.iloc[0]["condition"]),
            "speed_kmh": int(frame.iloc[0]["speed_kmh"]),
            "distance_m": float(frame.iloc[0]["distance_m"]),
            "iso": int(frame.iloc[0]["iso"]),
            "reference_condition": bool(int(frame.iloc[0]["reference_condition"])),
            "development_condition": bool(int(frame.iloc[0]["development_condition"])),
            "raw_rois": raw_rois,
            "routed_rois": routed_rois,
            "diagnostics": diagnostics.to_dict(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json_atomic(target, payload)
        validate_progress(target, frame, freeze_hash)
        completed += 1
        if completed == 1 or completed % 10 == 0 or completed == len(frames):
            print(json.dumps({"completed": completed, "total": len(frames), "condition": payload["condition_cell"]}), flush=True)
    return {"state": "INFERENCE_COMPLETE", "frames": completed}


def bootstrap_ci(values: np.ndarray, *, count: int = 10000, seed: int = 20261022) -> list[float]:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if len(array) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(count, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def classify(delta: float) -> str:
    if delta < 0:
        return "improved"
    if delta > 0:
        return "worsened"
    return "tied"


def aggregate_rows(progress: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for item in progress:
        base = {
            "frame_id": item["frame_id"],
            "condition_cell": item["condition_cell"],
            "speed_kmh": item["speed_kmh"],
            "distance_m": item["distance_m"],
            "iso": item["iso"],
            "reference_condition": item["reference_condition"],
            "development_condition": item["development_condition"],
        }
        diagnostic_rows.append({**base, **item["diagnostics"]})
        for arm, key in (("raw", "raw_rois"), ("routed", "routed_rois")):
            for row in item[key]:
                roi_rows.append({**base, "arm": arm, **row})
    roi = pd.DataFrame(roi_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    identity = [
        "frame_id", "condition_cell", "speed_kmh", "distance_m", "iso",
        "reference_condition", "development_condition", "roi_id", "orientation",
    ]
    valid = roi[roi["valid"] == True].copy()  # noqa: E712
    wide = valid.pivot(index=identity, columns="arm", values="mtf50").reset_index()
    paired = wide.dropna(subset=["raw", "routed"]).copy()
    frame = paired.groupby(
        ["frame_id", "condition_cell", "speed_kmh", "distance_m", "iso", "reference_condition", "development_condition"],
        as_index=False,
    ).agg(raw_mtf50=("raw", "mean"), routed_mtf50=("routed", "mean"), paired_rois=("roi_id", "count"))
    condition = frame.groupby(
        ["condition_cell", "speed_kmh", "distance_m", "iso", "reference_condition", "development_condition"],
        as_index=False,
    ).agg(
        raw_mtf50=("raw_mtf50", "median"),
        routed_mtf50=("routed_mtf50", "median"),
        paired_frames=("frame_id", "count"),
        paired_rois=("paired_rois", "sum"),
    )
    references = condition[condition["reference_condition"] == True][["distance_m", "raw_mtf50"]].rename(  # noqa: E712
        columns={"raw_mtf50": "reference_mtf50"}
    )
    if len(references) != 5 or references["distance_m"].nunique() != 5:
        raise RuntimeError("distance-specific RAW reference coverage changed")
    condition = condition.merge(references, on="distance_m", validate="many_to_one")
    condition["raw_abs_error"] = np.abs(condition["raw_mtf50"] - condition["reference_mtf50"])
    condition["routed_abs_error"] = np.abs(condition["routed_mtf50"] - condition["reference_mtf50"])
    condition["delta_abs_error_cy_px"] = condition["routed_abs_error"] - condition["raw_abs_error"]
    condition["result"] = condition["delta_abs_error_cy_px"].map(classify)
    diagnostic_summary = diagnostics.groupby("condition_cell", as_index=False).agg(
        frames_processed=("frame_id", "count"),
        safety_bypass_frames=("safety_bypass", "sum"),
        maximum_unprojected_overshoot_fraction=("unprojected_preclip_overshoot_fraction", "max"),
        maximum_headroom_limited_fraction=("headroom_limited_fraction", "max"),
        maximum_introduced_clipping_fraction=("introduced_clipping_fraction", "max"),
        maximum_abs_change_dn=("max_abs_change_dn", "max"),
        mean_abs_change_dn=("mean_abs_change_dn", "mean"),
    )
    condition = condition.merge(diagnostic_summary, on="condition_cell", validate="one_to_one")
    return roi, frame, condition


def render_report(summary: dict, condition: pd.DataFrame) -> str:
    ci = summary["full50_condition_bootstrap_95pct_ci"]
    rows = "\n".join(
        f"| {row.condition_cell} | {row.paired_frames} | {row.paired_rois} | "
        f"{row.delta_abs_error_cy_px:+.10f} | {row.result} | {row.safety_bypass_frames} |"
        for row in condition.sort_values(["speed_kmh", "distance_m", "iso"]).itertuples(index=False)
    )
    return f"""# 1. C058 v3h 50조건 전수 효능 분석

## 1.1 판정

동결된 v3h 단일 경로를 조건 우회 없이 479장·50조건 모두에 적용했다. 조건별 결과는 **{summary['improved_condition_cells']}개 개선, {summary['tied_condition_cells']}개 동률, {summary['worsened_condition_cells']}개 악화**다. 50개 조건 평균 Δ절대오차는 **{summary['full50_mean_delta_abs_error_cy_px']:+.10f} cy/px**, 조건 bootstrap 95% CI는 **[{ci[0]:+.10f}, {ci[1]:+.10f}]**다.

## 1.2 범위와 해석

- 처리 프레임: **{summary['frames_processed']}/479장**
- 측정 조건: **{summary['measured_condition_cells']}/50조건**
- paired ROI: **{summary['paired_roi_identities']}/1,737개**
- 안전 우회 프레임: **{summary['safety_bypass_frames']}장**
- 최대 신규 clipping: **{summary['maximum_introduced_clipping_fraction']:.6%}**
- 최대 픽셀 변화: **{summary['maximum_abs_change_dn']:.0f} DN**

50개 조건은 조건 단위 분석표본으로 집계했고 프레임과 ROI는 조건 내부 반복으로 처리했다. 거리별 60 km/h·ISO100 RAW를 기준 MTF50으로 사용했으며, 이 5개 기준조건도 복원 후 보존 여부를 평가하는 조건으로 포함했다.

## 1.3 조건별 결과

| 조건 | paired 프레임 | paired ROI | Δ절대오차 (cy/px) | 판정 | 안전우회 프레임 |
|---|---:|---:|---:|---|---:|
{rows}

음수는 RAW보다 거리별 기준 MTF50에 가까워졌음을, 양수는 멀어졌음을 뜻한다. 이 표는 모든 결과를 보존하며 불리한 조건을 제외하지 않는다.

## 1.4 증거 경계

이 분석은 기존 실제 취득 50조건을 모두 사용한 조건 단위 전수 분석이다. 경로와 계수는 출력 전에 동결했으며 결과에 따라 재조정하지 않았다. 차트 MTF50은 자연 균열 진실성이나 downstream 효능의 대리 정답으로 사용하지 않는다.
"""


def aggregate() -> dict:
    frozen, freeze_hash = load_freeze()
    _, _, cohort, _ = c057.load_verified_environment(C057_FREEZE)
    frames = c057.D0.frame_records(cohort)
    progress = []
    for ordinal, (_, frame) in enumerate(frames, start=1):
        path = progress_path(ordinal, str(frame.iloc[0]["frame_id"]))
        if not path.is_file():
            raise RuntimeError(f"missing progress: {path}")
        validate_progress(path, frame, freeze_hash)
        progress.append(json.loads(path.read_text(encoding="utf-8")))
    roi, frame, condition = aggregate_rows(progress)
    if len(condition) != 50:
        raise RuntimeError(f"measured condition coverage {len(condition)} != 50")
    deltas = condition["delta_abs_error_cy_px"].to_numpy(dtype=float)
    summary = {
        "schema_version": SCHEMA,
        "state": "COMPLETE",
        "created_utc": utc_now(),
        "evidence_label": frozen["evidence_label"],
        "execution_freeze_sha256": freeze_hash,
        "frames_processed": len(progress),
        "condition_cells": 50,
        "measured_condition_cells": int(len(condition)),
        "roi_identities": 1737,
        "paired_roi_identities": int(len(roi[roi["valid"] == True].pivot_table(index=["frame_id", "roi_id"], columns="arm", values="mtf50", aggfunc="first").dropna())),  # noqa: E712
        "paired_frames": int(frame["frame_id"].nunique()),
        "improved_condition_cells": int((condition["result"] == "improved").sum()),
        "tied_condition_cells": int((condition["result"] == "tied").sum()),
        "worsened_condition_cells": int((condition["result"] == "worsened").sum()),
        "full50_mean_delta_abs_error_cy_px": float(deltas.mean()),
        "full50_condition_bootstrap_95pct_ci": bootstrap_ci(deltas),
        "nonreference45_mean_delta_abs_error_cy_px": float(condition[condition["reference_condition"] != True]["delta_abs_error_cy_px"].mean()),  # noqa: E712
        "reference5_mean_delta_abs_error_cy_px": float(condition[condition["reference_condition"] == True]["delta_abs_error_cy_px"].mean()),  # noqa: E712
        "safety_bypass_frames": int(condition["safety_bypass_frames"].sum()),
        "maximum_unprojected_overshoot_fraction": float(condition["maximum_unprojected_overshoot_fraction"].max()),
        "maximum_headroom_limited_fraction": float(condition["maximum_headroom_limited_fraction"].max()),
        "maximum_introduced_clipping_fraction": float(condition["maximum_introduced_clipping_fraction"].max()),
        "maximum_abs_change_dn": float(condition["maximum_abs_change_dn"].max()),
        "integrity_checks": {
            "all_50_conditions_measured": len(condition) == 50,
            "all_479_frames_processed": len(progress) == 479,
            "all_1737_roi_identities_present_per_arm": len(roi) == 1737 * 2,
            "all_results_classified": int(condition["result"].notna().sum()) == 50,
            "no_introduced_clipping": float(condition["maximum_introduced_clipping_fraction"].max()) == 0.0,
            "route_unchanged_from_freeze": frozen["route"] == selected_route(),
        },
        "interpretation": "All 50 condition cells were processed with one frozen route; report favorable and unfavorable cells without selection.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    roi.to_csv(OUTPUT / "roi_results.csv", index=False)
    frame.to_csv(OUTPUT / "frame_results.csv", index=False)
    condition.to_csv(OUTPUT / "condition_results.csv", index=False)
    write_json_atomic(OUTPUT / "summary.json", summary)
    write_text_atomic(OUTPUT / "ALL50_EFFICACY_REPORT.md", render_report(summary, condition))
    return summary


def audit() -> dict:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    condition = pd.read_csv(OUTPUT / "condition_results.csv")
    checks = {
        "summary_integrity_checks_pass": all(summary["integrity_checks"].values()),
        "condition_rows_50": len(condition) == 50,
        "condition_results_complete": condition["result"].isin(["improved", "tied", "worsened"]).all(),
        "condition_count_reconciles": int(sum(summary[key] for key in ("improved_condition_cells", "tied_condition_cells", "worsened_condition_cells"))) == 50,
        "raw_reference_distances_5": condition[condition["reference_condition"] == True]["distance_m"].nunique() == 5,  # noqa: E712
    }
    result = {"state": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    write_json_atomic(OUTPUT / "independent_recalculation.json", result)
    if result["state"] != "PASS":
        raise RuntimeError(f"all-50 independent recalculation failed: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create-freeze", "run", "aggregate", "audit"))
    args = parser.parse_args()
    if args.command == "create-freeze":
        result = create_freeze()
    elif args.command == "run":
        result = run_inference()
    elif args.command == "aggregate":
        result = aggregate()
    else:
        result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
