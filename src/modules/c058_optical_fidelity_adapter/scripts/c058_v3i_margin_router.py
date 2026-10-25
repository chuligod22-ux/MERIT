"""Develop and evaluate the conservative C058 v3i optical-margin router.

The v3h all-condition run showed that the fixed restoration route usually
raises MTF50.  Applying it when RAW is already at, above, or very close to the
distance-specific reference can therefore overshoot the target.  v3i adds a
RAW-only preflight gate: the frozen v3h route is active only when the RAW
condition median is more than 0.001 cy/px below its reference.  Every other
condition is returned as byte-exact RAW without invoking the model.

This is a development analysis.  The margin threshold was selected after the
v3h all-50 results were available, so this script does not relabel the same 50
conditions as an untouched confirmation set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve()
TEST = BASE / "tests/test_c058_v3i_margin_router.py"
RECALCULATOR = BASE / "scripts/c058_v3i_recalculate.py"
SOURCE = BASE / "artifacts/safety_router_v3h_development/all50_efficacy_v1"
SOURCE_FREEZE = BASE / "protocol/all50_efficacy_execution_freeze_v1.json"
SELECTION = BASE / "protocol/selected_safety_router_v3h_development.json"
ADAPTER = BASE / "scripts/c058_adapter_v2.py"
FREEZE = BASE / "protocol/v3i_margin_router_development_freeze_v1.json"
OUTPUT = BASE / "artifacts/safety_router_v3i_development/all50_margin_gated_v1"
MARGIN_THRESHOLD_CY_PX = 0.001
SCHEMA = "MERIT-C058-V3I-MARGIN-ROUTER-DEVELOPMENT-v1"
FREEZE_SCHEMA = "MERIT-C058-V3I-MARGIN-ROUTER-FREEZE-v1"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


all50 = load("c058_v3i_all50", BASE / "scripts/c058_v3h_all50_efficacy.py")
adapter = load("c058_v3i_adapter", ADAPTER)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def progress_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(directory.glob("*.json"), key=lambda path: path.name)
    if len(files) != 479:
        raise RuntimeError(f"source progress count {len(files)} != 479")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
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


def route_is_active(*, reference_condition: bool, optical_margin_cy_px: float) -> bool:
    """Return the frozen RAW-only preflight decision for one condition."""

    return bool(not reference_condition and optical_margin_cy_px > MARGIN_THRESHOLD_CY_PX)


def identity_diagnostics() -> dict[str, bool | float]:
    return {
        "exact_identity": True,
        "safety_bypass": True,
        "residual_cap_fraction": 0.0,
        "uncapped_residual_mean_abs_dn": 0.0,
        "capped_residual_mean_abs_dn": 0.0,
        "unprojected_preclip_overshoot_fraction": 0.0,
        "unprojected_below_zero_fraction": 0.0,
        "unprojected_above_255_fraction": 0.0,
        "headroom_limited_fraction": 0.0,
        "introduced_clipping_fraction": 0.0,
        "introduced_low_clipping_fraction": 0.0,
        "introduced_high_clipping_fraction": 0.0,
        "mean_abs_change_dn": 0.0,
        "max_abs_change_dn": 0.0,
    }


def route_native_crop(
    raw: np.ndarray,
    *,
    active: bool,
    model_callback: Callable[[np.ndarray], np.ndarray],
    candidate,
) -> tuple[np.ndarray, dict[str, bool | float]]:
    """Apply v3h only after the margin gate; inactive output is byte exact."""

    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError("v3i route requires a 2D uint8 RAW crop")
    if not active:
        output = raw.copy()
        if not np.array_equal(output, raw):
            raise RuntimeError("inactive v3i route is not byte-exact")
        return output, identity_diagnostics()
    task_on = model_callback(raw)
    output, diagnostics = adapter.apply_safe_adapter(raw, task_on, candidate)
    return output, diagnostics.to_dict()


def source_files() -> dict[str, Path]:
    return {
        "script": SCRIPT,
        "tests": TEST,
        "recalculator": RECALCULATOR,
        "source_summary": SOURCE / "summary.json",
        "source_condition_results": SOURCE / "condition_results.csv",
        "source_roi_results": SOURCE / "roi_results.csv",
        "source_freeze": SOURCE_FREEZE,
        "v3h_selection": SELECTION,
        "adapter": ADAPTER,
    }


def create_freeze() -> dict:
    if (OUTPUT / "progress").exists() or (OUTPUT / "summary.json").exists():
        raise RuntimeError("v3i outputs already exist; refuse to freeze after results")
    files = source_files()
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing v3i freeze inputs: {missing}")
    source_summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("condition_cells") != 50 or source_summary.get("frames_processed") != 479:
        raise RuntimeError("v3h all-50 source coverage changed")
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "state": "DEVELOPMENT_ROUTER_FROZEN_BEFORE_V3I_OUTPUTS",
        "created_utc": utc_now(),
        "evidence_label": "all-50 condition-aware router development; not untouched confirmation",
        "purpose": "Prevent target overshoot while preserving the positive v3h optical response.",
        "router": {
            "reference": "distance-specific 60 km/h ISO100 RAW condition median MTF50",
            "optical_margin": "reference MTF50 minus RAW condition median MTF50",
            "active_if": "non-reference condition and optical_margin_cy_px > 0.001",
            "margin_threshold_cy_px": MARGIN_THRESHOLD_CY_PX,
            "active_route": "frozen v3h task-ON seed 1, alpha=-0.75, sigma=0.75 px, cap=8 DN",
            "otherwise": "model-free byte-exact RAW bypass",
        },
        "required_gates": {
            "condition_cells": 50,
            "reference_conditions_exact": 5,
            "worsened_conditions": 0,
            "speed80_mean_delta_negative": True,
            "speed80_bootstrap_ci_upper_negative": True,
            "introduced_clipping_fraction": 0.0,
        },
        "development_disclosure": (
            "The 0.001 cy/px margin threshold was selected after inspecting the v3h all-50 development "
            "results. This rerouting analysis is development evidence and must not be described as a new "
            "untouched confirmation of optical efficacy."
        ),
        "source_progress_tree_sha256": progress_tree_sha256(SOURCE / "progress"),
        "files": {
            label: {"path": relative(path), "sha256": sha256(path)}
            for label, path in files.items()
        },
    }
    write_json_atomic(FREEZE, payload)
    sidecar_path(FREEZE).write_text(f"{sha256(FREEZE)}  {FREEZE.name}\n", encoding="ascii")
    return payload


def load_freeze() -> tuple[dict, str]:
    if not FREEZE.is_file() or not sidecar_path(FREEZE).is_file():
        raise RuntimeError("v3i development freeze is missing")
    freeze_hash = sha256(FREEZE)
    fields = sidecar_path(FREEZE).read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[0] != freeze_hash or fields[1] != FREEZE.name:
        raise RuntimeError("v3i freeze sidecar mismatch")
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FREEZE_SCHEMA:
        raise RuntimeError("v3i freeze schema changed")
    if payload.get("router", {}).get("margin_threshold_cy_px") != MARGIN_THRESHOLD_CY_PX:
        raise RuntimeError("v3i margin threshold changed")
    for label, path in source_files().items():
        record = payload["files"].get(label, {})
        if record.get("path") != relative(path) or record.get("sha256") != sha256(path):
            raise RuntimeError(f"v3i freeze input mismatch: {label}")
    if payload.get("source_progress_tree_sha256") != progress_tree_sha256(SOURCE / "progress"):
        raise RuntimeError("v3h source progress tree changed")
    return payload, freeze_hash


def load_source_progress() -> list[dict]:
    paths = sorted((SOURCE / "progress").glob("*.json"), key=lambda path: path.name)
    if len(paths) != 479:
        raise RuntimeError(f"source progress count {len(paths)} != 479")
    progress = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({row["frame_id"] for row in progress}) != 479:
        raise RuntimeError("source frame identities are not unique")
    return progress


def raw_gate_table(progress: list[dict]) -> pd.DataFrame:
    roi_rows: list[dict] = []
    for item in progress:
        base = {
            "frame_id": item["frame_id"],
            "condition_cell": item["condition_cell"],
            "speed_kmh": int(item["speed_kmh"]),
            "distance_m": float(item["distance_m"]),
            "iso": int(item["iso"]),
            "reference_condition": bool(item["reference_condition"]),
        }
        for row in item["raw_rois"]:
            if bool(row["valid"]):
                roi_rows.append({**base, "roi_id": int(row["roi_id"]), "mtf50": float(row["mtf50"])})
    roi = pd.DataFrame(roi_rows)
    frame = roi.groupby(
        ["frame_id", "condition_cell", "speed_kmh", "distance_m", "iso", "reference_condition"],
        as_index=False,
    ).agg(raw_mtf50=("mtf50", "mean"))
    condition = frame.groupby(
        ["condition_cell", "speed_kmh", "distance_m", "iso", "reference_condition"],
        as_index=False,
    ).agg(raw_mtf50=("raw_mtf50", "median"))
    references = condition[condition["reference_condition"] == True][  # noqa: E712
        ["distance_m", "raw_mtf50"]
    ].rename(columns={"raw_mtf50": "reference_mtf50"})
    if len(references) != 5 or references["distance_m"].nunique() != 5:
        raise RuntimeError("RAW reference coverage changed")
    condition = condition.merge(references, on="distance_m", validate="many_to_one")
    condition["optical_margin_cy_px"] = condition["reference_mtf50"] - condition["raw_mtf50"]
    condition["route_active"] = [
        route_is_active(
            reference_condition=bool(reference),
            optical_margin_cy_px=float(margin),
        )
        for reference, margin in zip(condition["reference_condition"], condition["optical_margin_cy_px"], strict=True)
    ]
    condition["route_decision"] = np.where(condition["route_active"], "v3h_active", "raw_exact_bypass")
    if len(condition) != 50 or condition["condition_cell"].nunique() != 50:
        raise RuntimeError("v3i gate table does not contain 50 unique conditions")
    return condition.sort_values(["speed_kmh", "distance_m", "iso"]).reset_index(drop=True)


def progress_path(index: int, frame_id: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in "_.-" else "_" for char in frame_id)
    return OUTPUT / "progress" / f"{index:04d}_{safe_name}.json"


def run_router() -> dict:
    frozen, freeze_hash = load_freeze()
    source = load_source_progress()
    gate = raw_gate_table(source)
    active_by_condition = dict(zip(gate["condition_cell"], gate["route_active"], strict=True))
    margin_by_condition = dict(zip(gate["condition_cell"], gate["optical_margin_cy_px"], strict=True))
    completed = 0
    for ordinal, item in enumerate(source, start=1):
        condition = str(item["condition_cell"])
        active = bool(active_by_condition[condition])
        payload = copy.deepcopy(item)
        payload["schema_version"] = SCHEMA
        payload["execution_freeze_sha256"] = freeze_hash
        payload["source_execution_freeze_sha256"] = item["execution_freeze_sha256"]
        payload["route_active"] = active
        payload["optical_margin_cy_px"] = float(margin_by_condition[condition])
        payload["route_decision"] = "v3h_active" if active else "raw_exact_bypass"
        if not active:
            payload["routed_rois"] = copy.deepcopy(payload["raw_rois"])
            payload["diagnostics"] = identity_diagnostics()
        target = progress_path(ordinal, str(item["frame_id"]))
        write_json_atomic(target, payload)
        completed += 1
    gate.to_csv(OUTPUT / "gate_table.csv", index=False)
    return {
        "state": "ROUTING_COMPLETE",
        "frames": completed,
        "condition_cells": int(len(gate)),
        "active_condition_cells": int(gate["route_active"].sum()),
        "exact_bypass_condition_cells": int((~gate["route_active"]).sum()),
        "execution_freeze_sha256": freeze_hash,
        "evidence_label": frozen["evidence_label"],
    }


def load_routed_progress(freeze_hash: str) -> list[dict]:
    paths = sorted((OUTPUT / "progress").glob("*.json"), key=lambda path: path.name)
    if len(paths) != 479:
        raise RuntimeError(f"v3i progress count {len(paths)} != 479")
    progress = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("schema_version") != SCHEMA for row in progress):
        raise RuntimeError("v3i progress schema mismatch")
    if any(row.get("execution_freeze_sha256") != freeze_hash for row in progress):
        raise RuntimeError("v3i progress freeze mismatch")
    return progress


def bootstrap_ci(values: np.ndarray, *, count: int = 10000, seed: int = 20261029) -> list[float]:
    array = np.sort(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(count, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def render_report(summary: dict, condition: pd.DataFrame) -> str:
    full_ci = summary["full50_condition_bootstrap_95pct_ci"]
    speed80_ci = summary["speed80_condition_bootstrap_95pct_ci"]
    rows = "\n".join(
        f"| {row.condition_cell} | {row.route_decision} | {row.optical_margin_cy_px:+.10f} | "
        f"{row.delta_abs_error_cy_px:+.10f} | {row.result} |"
        for row in condition.sort_values(["speed_kmh", "distance_m", "iso"]).itertuples(index=False)
    )
    return f"""# 1. C058 v3i 광학 여유 라우터 50조건 개발 결과

## 1.1 판정

v3i는 50조건 전부를 보존해 재집계했고 목표 게이트를 모두 통과했다. 조건별 결과는 **{summary['improved_condition_cells']}개 개선, {summary['tied_condition_cells']}개 동률, {summary['worsened_condition_cells']}개 악화**다. 기준 5조건은 모두 byte-exact RAW 경로이며, 80 km/h의 평균 Δ절대오차와 조건 bootstrap 95% CI는 각각 **{summary['speed80_mean_delta_abs_error_cy_px']:+.10f} cy/px**, **[{speed80_ci[0]:+.10f}, {speed80_ci[1]:+.10f}]**다.

## 1.2 라우팅 규칙

- 광학 여유: 거리별 60 km/h·ISO100 RAW 기준 MTF50 - 현재 조건 RAW MTF50
- 활성: 비기준 조건이면서 광학 여유가 `0.001 cy/px`보다 큰 경우
- 비활성: 모델을 호출하지 않고 byte-exact RAW 반환
- 활성 복원: 기존 v3h task-ON seed 1, `alpha=-0.75`, `sigma=0.75 px`, cap `8 DN`

전체 50조건 평균 Δ절대오차는 **{summary['full50_mean_delta_abs_error_cy_px']:+.10f} cy/px**, 95% CI는 **[{full_ci[0]:+.10f}, {full_ci[1]:+.10f}]**다. 최대 신규 clipping은 **{summary['maximum_introduced_clipping_fraction']:.6%}**다.

## 1.3 조건별 결과

| 조건 | 경로 | RAW 광학 여유 (cy/px) | Δ절대오차 (cy/px) | 판정 |
|---|---|---:|---:|---|
{rows}

## 1.4 증거 경계

이 결과는 v3h 50조건 결과를 확인한 뒤 `0.001 cy/px` 게이트를 정한 개발 분석이다. 50조건 전수 결과와 불리한 결과를 모두 보존하지만, 같은 자료의 재라우팅 결과를 새로운 미노출 확증으로 부르지 않는다.
"""


def aggregate() -> dict:
    frozen, freeze_hash = load_freeze()
    progress = load_routed_progress(freeze_hash)
    roi, frame, condition = all50.aggregate_rows(progress)
    gate = pd.read_csv(OUTPUT / "gate_table.csv")
    condition = condition.merge(
        gate[["condition_cell", "optical_margin_cy_px", "route_active", "route_decision"]],
        on="condition_cell",
        validate="one_to_one",
    )
    deltas = condition["delta_abs_error_cy_px"].to_numpy(dtype=float)
    speed80 = condition[condition["speed_kmh"] == 80]
    reference = condition[condition["reference_condition"] == True]  # noqa: E712
    full_ci = bootstrap_ci(deltas, seed=20261029)
    speed80_ci = bootstrap_ci(speed80["delta_abs_error_cy_px"].to_numpy(dtype=float), seed=20261030)
    checks = {
        "all_50_conditions_measured": len(condition) == 50 and condition["condition_cell"].nunique() == 50,
        "all_479_frames_processed": len(progress) == 479,
        "reference_5_exact_preservation": bool(
            len(reference) == 5
            and (reference["route_active"] == False).all()  # noqa: E712
            and (reference["delta_abs_error_cy_px"] == 0.0).all()
            and (reference["maximum_abs_change_dn"] == 0.0).all()
        ),
        "speed80_mean_delta_negative": float(speed80["delta_abs_error_cy_px"].mean()) < 0.0,
        "speed80_bootstrap_ci_upper_negative": speed80_ci[1] < 0.0,
        "worsened_conditions_zero": int((condition["result"] == "worsened").sum()) == 0,
        "no_introduced_clipping": float(condition["maximum_introduced_clipping_fraction"].max()) == 0.0,
    }
    summary = {
        "schema_version": SCHEMA,
        "state": "PASS" if all(checks.values()) else "FAIL",
        "created_utc": utc_now(),
        "evidence_label": frozen["evidence_label"],
        "execution_freeze_sha256": freeze_hash,
        "frames_processed": len(progress),
        "condition_cells": int(len(condition)),
        "paired_roi_identities": int(
            len(
                roi[roi["valid"] == True]  # noqa: E712
                .pivot_table(index=["frame_id", "roi_id"], columns="arm", values="mtf50", aggfunc="first")
                .dropna()
            )
        ),
        "active_condition_cells": int(condition["route_active"].sum()),
        "exact_bypass_condition_cells": int((condition["route_active"] == False).sum()),  # noqa: E712
        "improved_condition_cells": int((condition["result"] == "improved").sum()),
        "tied_condition_cells": int((condition["result"] == "tied").sum()),
        "worsened_condition_cells": int((condition["result"] == "worsened").sum()),
        "reference_exact_condition_cells": int(
            ((condition["reference_condition"] == True) & (condition["delta_abs_error_cy_px"] == 0.0)).sum()  # noqa: E712
        ),
        "full50_mean_delta_abs_error_cy_px": float(deltas.mean()),
        "full50_condition_bootstrap_95pct_ci": full_ci,
        "speed80_mean_delta_abs_error_cy_px": float(speed80["delta_abs_error_cy_px"].mean()),
        "speed80_condition_bootstrap_95pct_ci": speed80_ci,
        "maximum_introduced_clipping_fraction": float(condition["maximum_introduced_clipping_fraction"].max()),
        "maximum_abs_change_dn": float(condition["maximum_abs_change_dn"].max()),
        "gate_checks": checks,
        "interpretation": (
            "The RAW-only optical-margin gate prevents v3h target overshoot on this 50-condition "
            "development set while retaining the positive 80 km/h response."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    roi.to_csv(OUTPUT / "roi_results.csv", index=False)
    frame.to_csv(OUTPUT / "frame_results.csv", index=False)
    condition.to_csv(OUTPUT / "condition_results.csv", index=False)
    write_json_atomic(OUTPUT / "summary.json", summary)
    write_text_atomic(OUTPUT / "ALL50_V3I_DEVELOPMENT_REPORT.md", render_report(summary, condition))
    if summary["state"] != "PASS":
        raise RuntimeError(f"v3i development gates failed: {checks}")
    return summary


def audit() -> dict:
    recalculator = load("c058_v3i_recalculate_runtime", RECALCULATOR)
    return recalculator.recalculate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create-freeze", "run", "aggregate", "audit"))
    args = parser.parse_args()
    if args.command == "create-freeze":
        result = create_freeze()
    elif args.command == "run":
        result = run_router()
    elif args.command == "aggregate":
        result = aggregate()
    else:
        result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
