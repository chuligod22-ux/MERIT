#!/usr/bin/env python3
"""Audit and summarize the completed MERIT D3 threshold-sweep addendum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd_across_seeds": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summary_index(summaries: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["condition"], row["seed"], row["subset"], row["group"]): row
        for row in summaries.values()
    }


def run_metric_rows(
    index: dict[tuple[str, str, str, str], dict[str, Any]],
    threshold: float,
    subset: str,
    group: str = "overall",
) -> list[dict[str, Any]]:
    rows = []
    for condition, seed in (
        ("raw", "none"),
        ("off", "0"),
        ("off", "1"),
        ("off", "2"),
        ("on", "0"),
        ("on", "1"),
        ("on", "2"),
    ):
        value = index[(condition, seed, subset, group)]
        rows.append(
            {
                "threshold": threshold,
                "subset": subset,
                "group": group,
                "condition": condition,
                "seed": seed,
                "positive_frames": value["positive_frames"],
                "positive_rois": value["positive_rois"],
                "frame_equal_roi_hit_fraction": value[
                    "frame_equal_roi_hit_fraction"
                ],
                "pooled_roi_hit_fraction": value[
                    "pooled_roi_hit_fraction_secondary"
                ],
                "no_visible_frames": value["no_visible_frames"],
                "components_per_no_visible_frame": value[
                    "components_per_no_visible_frame"
                ],
                "any_component_no_visible_frame_proportion": value[
                    "any_component_no_visible_frame_proportion"
                ],
            }
        )
    return rows


def condition_mean(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    if condition == "raw":
        if len(selected) != 1:
            raise ValueError("Raw condition must have one deterministic run")
    elif len(selected) != 3:
        raise ValueError(f"Condition {condition} must have three seeds")
    hit = [float(row["frame_equal_roi_hit_fraction"]) for row in selected]
    burden = [float(row["components_per_no_visible_frame"]) for row in selected]
    any_rate = [
        float(row["any_component_no_visible_frame_proportion"]) for row in selected
    ]
    return {
        "condition": condition,
        "n_runs": len(selected),
        "hit": mean_sd(hit),
        "burden": mean_sd(burden),
        "any_component_rate": mean_sd(any_rate),
    }


def fixed_threshold_summary(final: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold_key = "0.55"
    summaries = final["run_summaries_by_threshold"][threshold_key]
    index = summary_index(summaries)
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for subset in ("all", "calibration_nonR", "evaluation_R"):
        subset_rows = run_metric_rows(index, 0.55, subset)
        rows.extend(subset_rows)
        aggregate[subset] = {
            condition: condition_mean(subset_rows, condition)
            for condition in ("raw", "off", "on")
        }
    return rows, aggregate


def selected_operating_points(final: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_key, modes in final["development_selected_operating_points"].items():
        condition, seed = run_key.split("|")
        for mode, values in modes.items():
            calibration = values["calibration"]
            evaluation = values["evaluation"]
            rows.append(
                {
                    "run": run_key,
                    "condition": condition,
                    "seed": seed,
                    "mode": mode,
                    "feasible_on_calibration": calibration["feasible"],
                    "threshold": calibration["threshold"],
                    "target": calibration["target"],
                    "calibration_hit": calibration["hit"],
                    "calibration_burden": calibration["burden"],
                    "evaluation_hit": evaluation["hit"],
                    "evaluation_burden": evaluation["burden"],
                    "evaluation_any_component_rate": evaluation[
                        "any_component_rate"
                    ],
                }
            )
    aggregate: dict[str, Any] = {}
    for mode in ("sensitivity_matched", "burden_matched"):
        aggregate[mode] = {}
        for condition in ("raw", "off", "on"):
            selected = [
                row
                for row in rows
                if row["mode"] == mode and row["condition"] == condition
            ]
            aggregate[mode][condition] = {
                "runs": len(selected),
                "feasible_runs": sum(
                    bool(row["feasible_on_calibration"]) for row in selected
                ),
                "thresholds": [row["threshold"] for row in selected],
                "evaluation_hit": mean_sd(
                    [float(row["evaluation_hit"]) for row in selected]
                ),
                "evaluation_burden": mean_sd(
                    [float(row["evaluation_burden"]) for row in selected]
                ),
            }
    return rows, aggregate


def curve_ranges(final: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run_key, splits in final["curves"].items():
        for split in ("calibration_nonR", "evaluation_R"):
            curve = splits[split]
            rows.append(
                {
                    "run": run_key,
                    "split": split,
                    "hit_min": min(point["hit"] for point in curve),
                    "hit_max": max(point["hit"] for point in curve),
                    "burden_min": min(point["burden"] for point in curve),
                    "burden_max": max(point["burden"] for point in curve),
                }
            )
    return rows


def strata_summary(final: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = final["run_summaries_by_threshold"]["0.55"]
    index = summary_index(summaries)
    rows = []
    groups = sorted(
        {
            key[3]
            for key in index
            if key[2] == "evaluation_R" and key[3].startswith("region_iso:")
        }
    )
    for group in groups:
        run_rows = run_metric_rows(index, 0.55, "evaluation_R", group)
        for condition in ("raw", "off", "on"):
            value = condition_mean(run_rows, condition)
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "positive_frames": next(
                        row["positive_frames"]
                        for row in run_rows
                        if row["condition"] == condition
                    ),
                    "no_visible_frames": next(
                        row["no_visible_frames"]
                        for row in run_rows
                        if row["condition"] == condition
                    ),
                    "hit_mean": value["hit"]["mean"],
                    "hit_sd_across_seeds": value["hit"]["sd_across_seeds"],
                    "burden_mean": value["burden"]["mean"],
                    "burden_sd_across_seeds": value["burden"]["sd_across_seeds"],
                }
            )
    return rows


def progress_fixed_055_summary(
    sweep_root: Path,
    *,
    require_stride50: bool,
    group_by_camera_pass: bool,
) -> list[dict[str, Any]]:
    """Re-aggregate held-out R records for correlation-aware sensitivities."""
    accumulators: dict[tuple[str, str, str], dict[str, float]] = {}
    for path in sorted((sweep_root / "progress").glob("*.json")):
        record = load_json(path)
        if record["analysis_split"] != "evaluation_R":
            continue
        first_event = record["events"][0]
        if require_stride50 and "stride50" not in first_event["subsets"]:
            continue
        if group_by_camera_pass:
            camera = next(
                value.removeprefix("camera:")
                for value in first_event["groups"]
                if value.startswith("camera:")
            )
            traversal = next(
                value.removeprefix("pass:")
                for value in first_event["groups"]
                if value.startswith("pass:")
            )
            group = f"{camera}|{traversal}"
        else:
            group = "overall"
        for event in record["events"]:
            key = (group, event["condition"], event["seed"])
            bucket = accumulators.setdefault(
                key,
                {
                    "positive_frames": 0.0,
                    "positive_rois": 0.0,
                    "frame_equal_hit_sum": 0.0,
                    "no_visible_frames": 0.0,
                    "component_count": 0.0,
                },
            )
            result = event["threshold_results"]["0.55"]
            if result["label"] == "crack_visible":
                bucket["positive_frames"] += 1
                bucket["positive_rois"] += result["roi_count"]
                bucket["frame_equal_hit_sum"] += result["frame_roi_hit_fraction"]
            elif result["label"] == "no_visible_crack":
                bucket["no_visible_frames"] += 1
                bucket["component_count"] += result["eligible_component_count"]
            else:
                raise ValueError(f"Unexpected label in {path}: {result['label']}")

    run_metrics: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, value in accumulators.items():
        if not value["positive_frames"] or not value["no_visible_frames"]:
            raise ValueError(f"Insufficient positive/no-visible support for {key}")
        run_metrics[key] = {
            "positive_frames": value["positive_frames"],
            "positive_rois": value["positive_rois"],
            "no_visible_frames": value["no_visible_frames"],
            "hit": value["frame_equal_hit_sum"] / value["positive_frames"],
            "burden": value["component_count"] / value["no_visible_frames"],
        }

    rows: list[dict[str, Any]] = []
    for group in sorted({key[0] for key in run_metrics}):
        raw = run_metrics[(group, "raw", "none")]
        for condition, seeds in (
            ("raw", ("none",)),
            ("off", ("0", "1", "2")),
            ("on", ("0", "1", "2")),
        ):
            metrics = [run_metrics[(group, condition, seed)] for seed in seeds]
            hits = [value["hit"] for value in metrics]
            burdens = [value["burden"] for value in metrics]
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "positive_frames": int(metrics[0]["positive_frames"]),
                    "positive_rois": int(metrics[0]["positive_rois"]),
                    "no_visible_frames": int(metrics[0]["no_visible_frames"]),
                    "hit_mean": statistics.fmean(hits),
                    "hit_sd_across_seeds": (
                        statistics.stdev(hits) if len(hits) > 1 else 0.0
                    ),
                    "hit_seed_min": min(hits),
                    "hit_seed_max": max(hits),
                    "hit_delta_from_raw": statistics.fmean(hits) - raw["hit"],
                    "burden_mean": statistics.fmean(burdens),
                    "burden_sd_across_seeds": (
                        statistics.stdev(burdens) if len(burdens) > 1 else 0.0
                    ),
                    "burden_seed_min": min(burdens),
                    "burden_seed_max": max(burdens),
                    "burden_delta_from_raw": (
                        statistics.fmean(burdens) - raw["burden"]
                    ),
                }
            )
    return rows


def integrity_audit(
    sweep_root: Path, original_root: Path, final: dict[str, Any]
) -> dict[str, Any]:
    progress_files = sorted((sweep_root / "progress").glob("*.json"))
    original_files = sorted((original_root / "progress").glob("*.json"))
    if len(progress_files) != 974 or len(original_files) != 974:
        raise ValueError("Expected 974 progress records in each run")
    original_by_name = {path.name: path for path in original_files}
    endpoint_mismatches = []
    probability_hash_mismatches = []
    restoration_hash_mismatches = []
    run_coverage_mismatches = []
    threshold_coverage_mismatches = []
    expected_thresholds = [f"{value:.2f}" for value in final["thresholds"]]
    expected_runs = {
        ("raw", "none"),
        ("off", "0"),
        ("off", "1"),
        ("off", "2"),
        ("on", "0"),
        ("on", "1"),
        ("on", "2"),
    }
    for path in progress_files:
        sweep = load_json(path)
        original = load_json(original_by_name[path.name])
        sweep_events = {
            (event["condition"], event["seed"]): event for event in sweep["events"]
        }
        original_events = {
            (event["condition"], event["seed"]): event
            for event in original["events"]
        }
        if set(sweep_events) != expected_runs or set(original_events) != expected_runs:
            run_coverage_mismatches.append(path.name)
            continue
        for run_key in sorted(expected_runs):
            current = sweep_events[run_key]
            prior = original_events[run_key]
            if list(current["threshold_results"]) != expected_thresholds:
                threshold_coverage_mismatches.append([path.name, *run_key])
            if (
                current["probability_sha256_float32_c"]
                != prior["probability_sha256_float32_c"]
            ):
                probability_hash_mismatches.append([path.name, *run_key])
            if (
                current["restored_uint8_bgr_sha256"]
                != prior["restored_uint8_bgr_sha256"]
            ):
                restoration_hash_mismatches.append([path.name, *run_key])
            if (
                current["threshold_results"]["0.55"]
                != prior["endpoint_by_minimum_area"]["32"]
            ):
                endpoint_mismatches.append([path.name, *run_key])
    original_final = load_json(original_root / "d3_endpoint_streaming_final.json")
    sweep_055 = final["run_summaries_by_threshold"]["0.55"]
    summary_mismatches = []
    for key, value in original_final["run_summaries"].items():
        if sweep_055.get(key) != value:
            summary_mismatches.append(key)
    mismatch_counts = {
        "run_coverage": len(run_coverage_mismatches),
        "threshold_coverage": len(threshold_coverage_mismatches),
        "probability_hash": len(probability_hash_mismatches),
        "restoration_hash": len(restoration_hash_mismatches),
        "endpoint_0.55": len(endpoint_mismatches),
        "aggregate_summary_0.55": len(summary_mismatches),
    }
    return {
        "pass": not any(mismatch_counts.values()),
        "sweep_progress_records": len(progress_files),
        "original_progress_records": len(original_files),
        "events_checked": len(progress_files) * len(expected_runs),
        "thresholds_per_event": len(expected_thresholds),
        "mismatch_counts": mismatch_counts,
        "sweep_final_sha256": sha256_file(
            sweep_root / "d3_threshold_sweep_final.json"
        ),
        "original_final_sha256": sha256_file(
            original_root / "d3_endpoint_streaming_final.json"
        ),
    }


def verdict(
    fixed: dict[str, Any], selected: dict[str, Any], ranges: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluation = fixed["evaluation_R"]
    raw_hit = evaluation["raw"]["hit"]["mean"]
    raw_burden = evaluation["raw"]["burden"]["mean"]
    off_hit = evaluation["off"]["hit"]["mean"]
    on_hit = evaluation["on"]["hit"]["mean"]
    off_burden = evaluation["off"]["burden"]["mean"]
    on_burden = evaluation["on"]["burden"]["mean"]
    evaluation_ranges = {
        row["run"]: row for row in ranges if row["split"] == "evaluation_R"
    }
    raw_min_hit = evaluation_ranges["raw|none"]["hit_min"]
    raw_min_burden = evaluation_ranges["raw|none"]["burden_min"]
    restored_max_hit = max(
        row["hit_max"]
        for run, row in evaluation_ranges.items()
        if not run.startswith("raw|")
    )
    restored_max_burden = max(
        row["burden_max"]
        for run, row in evaluation_ranges.items()
        if not run.startswith("raw|")
    )
    return {
        "fixed_0.55_evaluation": {
            "raw_hit": raw_hit,
            "off_hit_mean": off_hit,
            "on_hit_mean": on_hit,
            "off_minus_raw_hit": off_hit - raw_hit,
            "on_minus_raw_hit": on_hit - raw_hit,
            "raw_burden": raw_burden,
            "off_burden_mean": off_burden,
            "on_burden_mean": on_burden,
            "off_burden_reduction_fraction": 1.0 - off_burden / raw_burden,
            "on_burden_reduction_fraction": 1.0 - on_burden / raw_burden,
        },
        "sensitivity_matching": {
            "restored_feasible_runs": sum(
                selected["sensitivity_matched"][condition]["feasible_runs"]
                for condition in ("off", "on")
            ),
            "restored_total_runs": 6,
        },
        "tested_curve_separation_evaluation": {
            "raw_hit_min": raw_min_hit,
            "restored_hit_max": restored_max_hit,
            "raw_burden_min": raw_min_burden,
            "restored_burden_max": restored_max_burden,
            "interpretation": (
                "Across thresholds 0.05-0.95, raw retained higher crack-hit "
                "sensitivity while every restoration run retained a lower "
                "no-visible component burden; the tested curves express a "
                "non-overlapping trade-off, not a detection improvement."
            ),
        },
        "claim_disposition": {
            "crack_detection_improvement": "NOT_SUPPORTED",
            "threshold_recalibration_salvage": "NOT_SUPPORTED_FOR_MATCHED_SENSITIVITY",
            "no_visible_component_burden_reduction": "SUPPORTED_AS_PROXY_ONLY",
            "task_on_advantage": "NOT_SUPPORTED_CONSISTENTLY",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    final = load_json(args.sweep_root / "d3_threshold_sweep_final.json")
    fixed_rows, fixed = fixed_threshold_summary(final)
    selected_rows, selected = selected_operating_points(final)
    ranges = curve_ranges(final)
    strata = strata_summary(final)
    camera_traversal = progress_fixed_055_summary(
        args.sweep_root,
        require_stride50=False,
        group_by_camera_pass=True,
    )
    stride50 = progress_fixed_055_summary(
        args.sweep_root,
        require_stride50=True,
        group_by_camera_pass=False,
    )
    integrity = integrity_audit(args.sweep_root, args.original_root, final)
    decision = verdict(fixed, selected, ranges)
    output = {
        "schema_version": "MERIT-C006-D3-THRESHOLD-ANALYSIS-v1",
        "integrity": integrity,
        "fixed_threshold_0.55": fixed,
        "development_selected_operating_points": selected,
        "curve_ranges": ranges,
        "strata_fixed_0.55_evaluation": strata,
        "camera_traversal_fixed_0.55_evaluation": camera_traversal,
        "stride50_fixed_0.55_evaluation": stride50,
        "verdict": decision,
        "statistical_scope": {
            "independent_evaluation_directions": 4,
            "positive_frames": 74,
            "positive_rois": 107,
            "no_visible_frames": 404,
            "warning": (
                "Frames and cameras within an acquisition direction are correlated. "
                "Frame-level significance would be pseudo-replication; four held-out "
                "direction units are insufficient for a well-powered confirmatory test."
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "d3_threshold_analysis_summary.json", output)
    write_csv(args.output_dir / "fixed_0.55_run_metrics.csv", fixed_rows)
    write_csv(args.output_dir / "selected_operating_points.csv", selected_rows)
    write_csv(args.output_dir / "curve_ranges.csv", ranges)
    write_csv(args.output_dir / "strata_fixed_0.55_evaluation.csv", strata)
    write_csv(
        args.output_dir / "camera_traversal_fixed_0.55_evaluation.csv",
        camera_traversal,
    )
    write_csv(args.output_dir / "stride50_fixed_0.55_evaluation.csv", stride50)
    print(json.dumps(output["verdict"], ensure_ascii=False, sort_keys=True))
    if not integrity["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
