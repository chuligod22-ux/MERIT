"""Resumable geometry and chart-eligibility audit for the C058 v3h locator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(r"<LOCAL_PATH_REDACTED>")
REFERENCE = BASE / "artifacts/locator_development_v1/locator_frame_results_v2.csv"
INVENTORY = BASE / "protocol/holdout_inventory_v3g.json"
OUTPUT = BASE / "artifacts/safety_router_v3h_development/locator_v2_audit"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


locator = load("c058_v3h_locator_audit_locator", BASE / "scripts/c058_raw_locator_v2.py")
v3 = load("c058_v3h_locator_audit_v3", BASE / "scripts/c058_safety_router_v3.py")
measure = load("c058_v3h_locator_audit_measure", BASE / "scripts/c058_mtf_measure_v3h.py")
c057 = v3.c057_module()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def progress_path(mode: str, ordinal: int) -> Path:
    return OUTPUT / mode / "progress" / f"{ordinal:04d}.json"


def exposed_records():
    frame = pd.read_csv(REFERENCE).sort_values("frame_id").reset_index(drop=True)
    for ordinal, row in enumerate(frame.to_dict(orient="records"), start=1):
        yield ordinal, row


def holdout_records():
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    for row in payload["records"]:
        yield int(row["ordinal"]), row


def run(mode: str) -> None:
    records = list(exposed_records() if mode == "exposed" else holdout_records())
    for ordinal, row in records:
        target = progress_path(mode, ordinal)
        if target.is_file():
            continue
        frame_id = str(row["frame_id"])
        image = locator_read(RAW_ROOT / frame_id)
        result = locator.assess_chart(image)
        payload = {
            "ordinal": ordinal,
            "frame_id": frame_id,
            "condition": str(row["condition"]),
            **result.to_dict(),
        }
        if mode == "exposed":
            reference_x = float(row["crop_x"]) + float(row["reference_center_x_extract"])
            reference_y = float(row["crop_y"]) + float(row["reference_center_y_extract"])
            reference_radius = float(row["reference_radius"])
            payload.update(
                {
                    "reference_center_x_raw": reference_x,
                    "reference_center_y_raw": reference_y,
                    "reference_radius": reference_radius,
                    "center_error_px": float(
                        math.hypot(result.center_x - reference_x, result.center_y - reference_y)
                    ),
                    "center_error_radii": float(
                        math.hypot(result.center_x - reference_x, result.center_y - reference_y)
                        / reference_radius
                    ),
                    "radius_error_fraction": abs(result.radius - reference_radius)
                    / reference_radius,
                }
            )
        elif sum(value >= measure.MIN_ROI_SOURCE_COVERAGE for value in locator.roi_source_coverages(result)) >= 2:
            rois = measure.measure_native(image, result, locator, c057)
            payload["raw_rois"] = rois
            payload["raw_valid_rois"] = sum(bool(item["valid"]) for item in rois)
        write_json(target, payload)
        if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(records):
            print(json.dumps({"mode": mode, "completed": ordinal, "total": len(records)}), flush=True)


def locator_read(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"unreadable image: {path}")
    return image


def aggregate(mode: str) -> dict:
    records = list(exposed_records() if mode == "exposed" else holdout_records())
    rows = []
    for ordinal, _ in records:
        path = progress_path(mode, ordinal)
        if not path.is_file():
            raise RuntimeError(f"missing progress: {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame(rows)
    frame["effective_eligible"] = (
        (frame["template_score"] >= locator.MIN_TEMPLATE_SCORE)
        & (frame["minimum_roi_source_coverage"] >= locator.MIN_SOURCE_COVERAGE)
        & (frame["radial_coverage"] >= locator.MIN_RADIAL_COVERAGE)
    )
    frame["effective_rejection_reason"] = np.where(
        frame["effective_eligible"], "ok", frame["rejection_reason"]
    )
    reasons = {str(key): int(value) for key, value in frame["effective_rejection_reason"].value_counts().items()}
    summary = {
        "mode": mode,
        "frames": len(frame),
        "eligible_frames": int(frame["effective_eligible"].sum()),
        "eligible_fraction": float(frame["effective_eligible"].mean()),
        "rejection_reasons": reasons,
    }
    if mode == "exposed":
        eligible_geometry = frame[frame["effective_eligible"]].copy()
        summary.update(
            {
                "eligible_center_error_radii_median": float(eligible_geometry["center_error_radii"].median()),
                "eligible_center_error_radii_p95": float(eligible_geometry["center_error_radii"].quantile(0.95)),
                "eligible_radius_error_fraction_median": float(eligible_geometry["radius_error_fraction"].median()),
                "eligible_radius_error_fraction_p95": float(eligible_geometry["radius_error_fraction"].quantile(0.95)),
                "eligible_radius_error_le_10pct_fraction": float((eligible_geometry["radius_error_fraction"] <= 0.10).mean()),
                "all_frame_radius_error_fraction_median": float(frame["radius_error_fraction"].median()),
            }
        )
    else:
        eligible = frame[frame["effective_eligible"]].copy()
        valid_counts = eligible["raw_valid_rois"].fillna(0).astype(int) if len(eligible) else pd.Series(dtype=int)
        summary.update(
            {
                "eligible_raw_roi_attempts": int(len(eligible) * 4),
                "eligible_raw_valid_rois": int(valid_counts.sum()),
                "eligible_raw_valid_fraction": float(valid_counts.sum() / (len(eligible) * 4)) if len(eligible) else 0.0,
                "eligible_four_valid_roi_frames": int((valid_counts == 4).sum()),
                "eligible_four_valid_roi_fraction": float((valid_counts == 4).mean()) if len(eligible) else 0.0,
            }
        )
    output = OUTPUT / mode / "summary.json"
    write_json(output, summary)
    frame.to_csv(OUTPUT / mode / "frame_results.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--mode", choices=("exposed", "holdout"), required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.mode)
        result = {"state": "complete", "mode": args.mode}
    else:
        result = aggregate(args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
