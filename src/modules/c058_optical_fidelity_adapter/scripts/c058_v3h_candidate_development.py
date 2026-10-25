"""Candidate development on chart-eligible C058 frames after the v3g audit.

Both the former 280-frame holdout and the earlier exposed cohort are explicitly
development data here.  The script never labels its output as independent
confirmation.  It reuses one RAW-derived v3h geometry for RAW and every model
variant, performs boundary-safe tiling, and applies quantization-safe headroom
projection before MTF50 measurement.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(r"<LOCAL_PATH_REDACTED>")
AUDIT = BASE / "artifacts/safety_router_v3h_development/locator_v2_audit"
OUTPUT = BASE / "artifacts/safety_router_v3h_development/candidate_grid_v1"
ALPHAS = (-1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5)
SEEDS = (0, 1, 2)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


locator = load("c058_v3h_grid_locator", BASE / "scripts/c058_raw_locator_v2.py")
v3 = load("c058_v3h_grid_v3", BASE / "scripts/c058_safety_router_v3.py")
safe = load("c058_v3h_grid_adapter", BASE / "scripts/c058_adapter_v2.py")
inference = load("c058_v3h_grid_inference", BASE / "scripts/c058_inference_v3h.py")
measure = load("c058_v3h_grid_measure", BASE / "scripts/c058_mtf_measure_v3h.py")
c057 = v3.c057_module()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_condition(condition: str):
    match = re.fullmatch(r"(60|80)km_(2\.5|3\.5|4\.5|5\.5|6\.5)m_ISO(100|200|400|800|1600)", condition)
    if match is None:
        raise ValueError(condition)
    return int(match.group(1)), float(match.group(2)), int(match.group(3))


def included(speed: int, distance: float, iso: int) -> tuple[bool, bool]:
    reference = speed == 60 and iso == 100
    active = speed == 80 and iso in {100, 200} and distance in {2.5, 3.5, 4.5}
    return reference, active


def audit_records():
    records = []
    for source in ("exposed", "holdout"):
        progress = AUDIT / source / "progress"
        if not progress.is_dir():
            continue
        for path in sorted(progress.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            speed, distance, iso = parse_condition(str(row["condition"]))
            reference, active = included(speed, distance, iso)
            located = rebuild_locator(row)
            usable_roi_count = sum(
                value >= measure.MIN_ROI_SOURCE_COVERAGE
                for value in locator.roi_source_coverages(located)
            )
            effective_eligible = (
                float(row["template_score"]) >= locator.MIN_TEMPLATE_SCORE
                and float(row["radial_coverage"]) >= locator.MIN_RADIAL_COVERAGE
                and usable_roi_count >= 2
            )
            if effective_eligible and (reference or active):
                row["eligible"] = True
                row["rejection_reason"] = "ok"
                row["usable_source_rois"] = usable_roi_count
                row.update(
                    {
                        "source_cohort": source,
                        "speed_kmh": speed,
                        "distance_m": distance,
                        "iso": iso,
                        "is_reference": reference,
                        "route_active": active,
                    }
                )
                records.append(row)
    records.sort(key=lambda row: (row["source_cohort"], row["condition"], row["frame_id"]))
    return records


def rebuild_locator(row: dict):
    names = {field.name for field in dataclasses.fields(locator.LocatorV2Result)}
    return locator.LocatorV2Result(**{name: row[name] for name in names})


def read_gray(path: Path):
    import cv2

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"unreadable image: {path}")
    return image


def route_key(seed: int, alpha: float) -> str:
    sign = "m" if alpha < 0 else "p"
    return f"safe_s{seed}_a{sign}{abs(alpha):.2f}"


def identity_diagnostics():
    return {
        "exact_identity": True,
        "safety_bypass": False,
        "introduced_clipping_fraction": 0.0,
        "headroom_limited_fraction": 0.0,
        "max_abs_change_dn": 0.0,
    }


def run() -> None:
    records = audit_records()
    if not records:
        raise RuntimeError("no chart-eligible audit records")
    protocol, _, _, freeze_hash = c057.load_verified_environment(c057.DEFAULT_FREEZE)
    device = c057.D0.require_cuda()
    models, tiled = c057.D0.load_models(protocol, device)
    models = {key: value for key, value in models.items() if key[0] == "on"}
    tiling = protocol["restoration"]["tiling"]
    for ordinal, row in enumerate(records, start=1):
        target = OUTPUT / "progress" / f"{ordinal:04d}.json"
        if target.is_file():
            continue
        started = time.perf_counter()
        image = read_gray(RAW_ROOT / row["frame_id"])
        located = rebuild_locator(row)
        raw_rois = measure.measure_native(image, located, locator, c057)
        events = [{"run_key": "raw_none", "diagnostics": identity_diagnostics(), "rois": raw_rois}]
        if row["route_active"]:
            x0, y0, x1, y1 = v3.native_context_bounds(image.shape, located)
            crop = image[y0:y1, x0:x1]
            local = dataclasses.replace(
                located,
                center_x=float(located.center_x) - x0,
                center_y=float(located.center_y) - y0,
                original_width=int(crop.shape[1]),
                original_height=int(crop.shape[0]),
            )
            for seed in SEEDS:
                task_on = inference.tiled_restore_native_u8(
                    crop,
                    models[("on", str(seed))],
                    device,
                    tiled,
                    tile_size=int(tiling["tile_size"]),
                    overlap=int(tiling["overlap"]),
                    batch_size=int(tiling["batch_size"]),
                    blend_floor=float(tiling["positive_hann_floor"]),
                    amp_dtype=str(protocol["restoration"]["amp_dtype"]),
                )
                for alpha in ALPHAS:
                    candidate = safe.SafeAdapterCandidate(
                        route_id=route_key(seed, alpha),
                        alpha=alpha,
                        sigma=0.75,
                        cap_dn=8.0,
                        maximum_headroom_limited_fraction=0.01,
                    )
                    output, diagnostics = safe.apply_safe_adapter(crop, task_on, candidate)
                    events.append(
                        {
                            "run_key": candidate.route_id,
                            "diagnostics": diagnostics.to_dict(),
                            "rois": measure.measure_native(output, local, locator, c057),
                        }
                    )
        payload = {
            "schema_version": "MERIT-C058-V3H-CANDIDATE-DEVELOPMENT-v1",
            "evidence_label": "development_only_former_holdout_exposed",
            "ordinal": ordinal,
            "frame_id": row["frame_id"],
            "condition_cell": row["condition"],
            "source_cohort": row["source_cohort"],
            "speed_kmh": row["speed_kmh"],
            "distance_m": row["distance_m"],
            "iso": row["iso"],
            "is_reference": row["is_reference"],
            "route_active": row["route_active"],
            "locator": located.to_dict(),
            "events": events,
            "c057_freeze_sha256": freeze_hash,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(target, payload)
        if ordinal == 1 or ordinal % 5 == 0 or ordinal == len(records):
            print(json.dumps({"completed": ordinal, "total": len(records)}), flush=True)


def bootstrap_ci(values, count=10000, seed=20261058):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(count, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def aggregate() -> dict:
    records = audit_records()
    progress = []
    for ordinal in range(1, len(records) + 1):
        path = OUTPUT / "progress" / f"{ordinal:04d}.json"
        if not path.is_file():
            raise RuntimeError(f"missing progress: {path}")
        progress.append(json.loads(path.read_text(encoding="utf-8")))
    roi_rows, diagnostic_rows = [], []
    for frame in progress:
        for event in frame["events"]:
            diagnostic_rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "condition_cell": frame["condition_cell"],
                    "source_cohort": frame["source_cohort"],
                    "distance_m": frame["distance_m"],
                    "iso": frame["iso"],
                    "is_reference": frame["is_reference"],
                    "run_key": event["run_key"],
                    **event["diagnostics"],
                }
            )
            for result in event["rois"]:
                roi_rows.append(
                    {
                        "frame_id": frame["frame_id"],
                        "condition_cell": frame["condition_cell"],
                        "source_cohort": frame["source_cohort"],
                        "distance_m": frame["distance_m"],
                        "iso": frame["iso"],
                        "is_reference": frame["is_reference"],
                        "run_key": event["run_key"],
                        **result,
                    }
                )
    roi = pd.DataFrame(roi_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    reference = roi[(roi["is_reference"] == True) & (roi["run_key"] == "raw_none") & (roi["valid"] == True)].copy()  # noqa: E712
    reference_frame = reference.groupby(["frame_id", "distance_m"], as_index=False)["mtf50_native_cy_px"].mean()
    references = reference_frame.groupby("distance_m")["mtf50_native_cy_px"].median().to_dict()
    active = roi[roi["is_reference"] != True].copy()  # noqa: E712
    raw = active[(active["run_key"] == "raw_none") & (active["valid"] == True)][
        ["frame_id", "condition_cell", "source_cohort", "distance_m", "iso", "roi_id", "mtf50_native_cy_px"]
    ].rename(columns={"mtf50_native_cy_px": "raw_mtf50"})
    routed = active[(active["run_key"] != "raw_none") & (active["valid"] == True)].copy()  # noqa: E712
    common = routed.merge(
        raw,
        on=["frame_id", "condition_cell", "source_cohort", "distance_m", "iso", "roi_id"],
        how="inner",
    )
    frame_values = common.groupby(
        ["frame_id", "condition_cell", "source_cohort", "distance_m", "iso", "run_key"], as_index=False
    )[["mtf50_native_cy_px", "raw_mtf50"]].mean()
    condition = frame_values.groupby(
        ["condition_cell", "source_cohort", "distance_m", "iso", "run_key"], as_index=False
    )[["mtf50_native_cy_px", "raw_mtf50"]].median()
    condition["reference_mtf50"] = [references[float(value)] for value in condition["distance_m"]]
    condition["delta_abs_error"] = (
        np.abs(condition["mtf50_native_cy_px"] - condition["reference_mtf50"])
        - np.abs(condition["raw_mtf50"] - condition["reference_mtf50"])
    )
    combined_condition = condition.groupby(
        ["condition_cell", "distance_m", "iso", "run_key"], as_index=False
    )["delta_abs_error"].mean()
    summaries = []
    for run_key, group in combined_condition.groupby("run_key"):
        values = group["delta_abs_error"].to_numpy(dtype=float)
        diag = diagnostics[diagnostics["run_key"] == run_key]
        summaries.append(
            {
                "run_key": run_key,
                "condition_cells": int(len(group)),
                "mean_delta_abs_error_cy_px": float(values.mean()),
                "condition_bootstrap_95pct_ci": bootstrap_ci(values),
                "improved_conditions": int((values < 0).sum()),
                "maximum_introduced_clipping_fraction": float(diag["introduced_clipping_fraction"].max()),
                "maximum_headroom_limited_fraction": float(diag["headroom_limited_fraction"].max()),
                "maximum_abs_change_dn": float(diag["max_abs_change_dn"].max()),
            }
        )
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["improved_conditions", "mean_delta_abs_error_cy_px"], ascending=[False, True]
    )
    roi.to_csv(OUTPUT / "roi_results.csv", index=False)
    diagnostics.to_csv(OUTPUT / "pixel_diagnostics.csv", index=False)
    condition.to_csv(OUTPUT / "condition_by_source.csv", index=False)
    combined_condition.to_csv(OUTPUT / "condition_combined.csv", index=False)
    summary_frame.to_json(OUTPUT / "candidate_summary.json", orient="records", indent=2)
    result = {
        "state": "DEVELOPMENT_COMPLETE",
        "evidence_label": "development_only_former_holdout_exposed",
        "frames": len(progress),
        "reference_frames": sum(bool(item["is_reference"]) for item in progress),
        "active_frames": sum(bool(item["route_active"]) for item in progress),
        "reference_mtf50_native_cy_px_by_distance": {str(k): float(v) for k, v in references.items()},
        "best_candidates": summary_frame.head(10).to_dict(orient="records"),
    }
    write_json(OUTPUT / "development_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "aggregate"))
    args = parser.parse_args()
    if args.command == "run":
        run()
        result = {"state": "INFERENCE_COMPLETE"}
    else:
        result = aggregate()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
