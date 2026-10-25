"""Result-neutral D3 schema repair for the frozen C009 v3 summarizer.

The frozen endpoint emits frame-level ROI hit fractions for ``crack_visible``
frames and component-burden fields for ``no_visible_crack`` frames.  The v3
summarizer incorrectly requested the latter fields from every label.  This
module replaces only ``summarize_d3`` and delegates every other calculation to
the unchanged, frozen v3 implementation.
"""

from __future__ import annotations

import math
from typing import Any

import c009_summarize_results as frozen


def d3_endpoint_values(
    endpoint: dict[str, Any], expected_label: str
) -> tuple[float, float, float]:
    """Return visible response, negative-frame hit, and negative burden.

    Non-applicable values are NaN so label-specific aggregation cannot mix the
    two endpoint meanings.
    """

    label = endpoint.get("label")
    if label != expected_label:
        raise frozen.common.EvaluationError(
            f"D3 endpoint label mismatch: expected {expected_label}, got {label}"
        )
    if label == "crack_visible":
        if "frame_roi_hit_fraction" not in endpoint:
            raise frozen.common.EvaluationError(
                "D3 crack-visible endpoint lacks frame_roi_hit_fraction"
            )
        value = float(endpoint["frame_roi_hit_fraction"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise frozen.common.EvaluationError(
                "D3 frame_roi_hit_fraction is outside [0, 1]"
            )
        return value, float("nan"), float("nan")
    if label == "no_visible_crack":
        if "any_eligible_component" not in endpoint or "components_per_megapixel" not in endpoint:
            raise frozen.common.EvaluationError(
                "D3 no-visible endpoint lacks component-burden fields"
            )
        burden = float(endpoint["components_per_megapixel"])
        if not math.isfinite(burden) or burden < 0.0:
            raise frozen.common.EvaluationError(
                "D3 components_per_megapixel is non-finite or negative"
            )
        return float("nan"), float(bool(endpoint["any_eligible_component"])), burden
    raise frozen.common.EvaluationError(f"Ineligible D3 endpoint label: {label}")


def summarize_d3(contract: dict[str, Any]) -> dict[str, Any]:
    legacy_root = (
        frozen.C009.parent
        / "c006_execution"
        / "artifacts"
        / "d3_stream_v1"
        / "progress"
    )
    expected_frames = int(
        contract["wp6_different_tunnel_same_system"]["eligible_rows"]
    )
    raw: dict[str, dict[str, Any]] = {}
    legacy_paths = sorted(legacy_root.glob("*.json"))
    if len(legacy_paths) != expected_frames:
        raise frozen.common.EvaluationError("D3 raw baseline coverage mismatch")
    for path in legacy_paths:
        value = frozen.common.load_json(path)
        cohort_id = value["cohort_id"]
        if cohort_id in raw:
            raise frozen.common.EvaluationError(
                f"Duplicate D3 raw cohort: {cohort_id}"
            )
        events = [event for event in value["events"] if event["condition"] == "raw"]
        if len(events) != 1:
            raise frozen.common.EvaluationError(
                f"D3 raw event count mismatch: {cohort_id}"
            )
        raw[cohort_id] = events[0]

    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    specifications = frozen.common.wp1_specifications(contract)
    if len(specifications) != 18:
        raise frozen.common.EvaluationError("D3 WP1 model count differs from 18")
    for spec in specifications:
        identity = frozen.common.model_id(spec)
        root = frozen.C009 / "evaluation" / "wp6_d3_fullframe" / identity / "progress"
        model_paths = sorted(root.glob("*.json"))
        if len(model_paths) != expected_frames:
            raise frozen.common.EvaluationError(
                f"D3 restored coverage mismatch for {identity}"
            )
        seen: set[str] = set()
        for path in model_paths:
            value = frozen.common.load_json(path)
            cohort_id = value["cohort_id"]
            if cohort_id in seen:
                raise frozen.common.EvaluationError(
                    f"Duplicate D3 restored cohort for {identity}: {cohort_id}"
                )
            seen.add(cohort_id)
            if cohort_id not in raw:
                raise frozen.common.EvaluationError(
                    f"D3 restored cohort lacks raw baseline: {cohort_id}"
                )
            label = value["consensus_label"]
            restored = value["endpoint"]["endpoint_by_minimum_area"]["32"]
            baseline = raw[cohort_id]["endpoint_by_minimum_area"]["32"]
            raw_visible, raw_negative_any, raw_negative_burden = d3_endpoint_values(
                baseline, label
            )
            (
                restored_visible,
                restored_negative_any,
                restored_negative_burden,
            ) = d3_endpoint_values(restored, label)
            rows.append(
                {
                    **spec,
                    "model_id": identity,
                    "cohort_id": cohort_id,
                    "label": label,
                    "raw_visible_response": raw_visible,
                    "restored_visible_response": restored_visible,
                    "raw_negative_any": raw_negative_any,
                    "restored_negative_any": restored_negative_any,
                    "raw_negative_components_per_megapixel": raw_negative_burden,
                    "restored_negative_components_per_megapixel": restored_negative_burden,
                }
            )
            quality_rows.append(
                {
                    **spec,
                    "model_id": identity,
                    "cohort_id": cohort_id,
                    **{
                        f"cell_{key}": val
                        for key, val in value.get("cell", {}).items()
                    },
                    **{
                        f"delta_{key}": float(value["restored_quality"][key])
                        - float(raw_value)
                        for key, raw_value in value["raw_quality"].items()
                    },
                }
            )
        if seen != set(raw):
            raise frozen.common.EvaluationError(
                f"D3 raw/restored cohort set mismatch for {identity}"
            )

    frame = frozen.pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    for identity, group in frame.groupby("model_id"):
        positive = group[group["label"] == "crack_visible"]
        negative = group[group["label"] == "no_visible_crack"]
        if len(positive) != 163 or len(negative) != 811:
            raise frozen.common.EvaluationError(
                f"D3 label coverage mismatch for {identity}"
            )
        summaries[identity] = {
            "visible_sensitivity_raw": float(positive["raw_visible_response"].mean()),
            "visible_sensitivity_restored": float(
                positive["restored_visible_response"].mean()
            ),
            "no_visible_positive_rate_raw": float(negative["raw_negative_any"].mean()),
            "no_visible_positive_rate_restored": float(
                negative["restored_negative_any"].mean()
            ),
            "no_visible_components_per_megapixel_delta": float(
                (
                    negative["restored_negative_components_per_megapixel"]
                    - negative["raw_negative_components_per_megapixel"]
                ).mean()
            ),
        }

    quality = frozen.pd.DataFrame(quality_rows)
    quality_metrics = [
        column for column in quality.columns if column.startswith("delta_")
    ]
    quality_summary = {
        metric: float(quality[metric].mean()) for metric in quality_metrics
    }
    return {
        "models": summaries,
        "quality_descriptor_mean_deltas_all_models": quality_summary,
        "frames_per_model": expected_frames,
        "visible_metric": "frame_equal_roi_hit_fraction",
        "no_visible_positive_metric": "any_component_frame_proportion",
        "no_visible_burden_metric": "components_per_megapixel",
        "claim_boundary": contract["wp6_different_tunnel_same_system"][
            "claim_boundary"
        ],
    }


def main() -> None:
    frozen.summarize_d3 = summarize_d3
    frozen.main()


if __name__ == "__main__":
    main()
