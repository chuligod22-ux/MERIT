"""Exploratory stability audit for MERIT v2.4 detection cycle 4 candidates."""

from __future__ import annotations

import json
import itertools

import c016_cycle4 as cycle4
import c016_pipeline as base
import c016_profiles as profiles


def score_subset(rows, detector, condition, tunnels):
    selected = [
        row
        for row in rows
        if row["detector"] == detector
        and row["condition"] == condition
        and row["tunnel"] in tunnels
    ]
    per_tunnel = {
        tunnel: {
            key: base.mean(row[key] for row in selected if row["tunnel"] == tunnel)
            for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
        }
        for tunnel in tunnels
    }
    return {
        "condition": condition,
        **{
            key: base.mean(value[key] for value in per_tunnel.values())
            for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
        },
        "positive_tunnels": sum(value["delta_auprc"] > 0.0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def tuple_score(scores, route_tuple, tunnels):
    detectors = tuple(base.field_common.DETECTORS)
    metrics = {
        detector: scores[detector][condition]
        for detector, condition in zip(detectors, route_tuple)
    }
    per_tunnel = {
        tunnel: base.mean(
            metrics[detector]["per_tunnel"][tunnel]["delta_auprc"]
            for detector in detectors
        )
        for tunnel in tunnels
    }
    return {
        "routes": dict(zip(detectors, route_tuple)),
        "detectors": metrics,
        "panel_delta_auprc": base.mean(value["delta_auprc"] for value in metrics.values()),
        "panel_delta_matched_fpr": base.mean(
            value["delta_matched_fpr"] for value in metrics.values()
        ),
        "panel_per_tunnel_delta_auprc": per_tunnel,
        "magnitude": sum(
            float(cycle4.FUSIONS[condition]["weight"]) for condition in route_tuple
        ),
    }


def crossfit(rows):
    result = {}
    detectors = tuple(base.field_common.DETECTORS)
    for held in base.TUNNELS:
        training = tuple(tunnel for tunnel in base.TUNNELS if tunnel != held)
        scores = {
            detector: {
                condition: score_subset(rows, detector, condition, training)
                for condition in cycle4.FUSIONS
            }
            for detector in detectors
        }
        reference = tuple_score(scores, tuple("full" for _ in detectors), training)
        eligible = []
        for route_tuple in itertools.product(cycle4.FUSIONS, repeat=len(detectors)):
            candidate = tuple_score(scores, route_tuple, training)
            paired = {
                tunnel: candidate["panel_per_tunnel_delta_auprc"][tunnel]
                - reference["panel_per_tunnel_delta_auprc"][tunnel]
                for tunnel in training
            }
            if (
                candidate["panel_delta_matched_fpr"] <= 0.0
                and candidate["panel_delta_auprc"] > reference["panel_delta_auprc"]
                and sum(value > 0.0 for value in paired.values()) >= 2
                and all(
                    value > 0.0
                    for value in candidate["panel_per_tunnel_delta_auprc"].values()
                )
                and all(value["delta_auprc"] >= 0.0 for value in candidate["detectors"].values())
            ):
                eligible.append(candidate)
        selected = max(
            eligible,
            key=lambda item: (
                item["panel_delta_auprc"],
                -item["magnitude"],
                json.dumps(item["routes"], sort_keys=True),
            ),
        )
        held_scores = {
            detector: {
                condition: score_subset(rows, detector, condition, (held,))
                for condition in cycle4.FUSIONS
            }
            for detector in detectors
        }
        held_selected = tuple_score(
            held_scores,
            tuple(selected["routes"][detector] for detector in detectors),
            (held,),
        )
        held_reference = tuple_score(held_scores, tuple("full" for _ in detectors), (held,))
        result[held] = {
            "routes": selected["routes"],
            "held_delta_auprc": held_selected["panel_delta_auprc"],
            "held_delta_fpr": held_selected["panel_delta_matched_fpr"],
            "held_minus_cycle3": held_selected["panel_delta_auprc"]
            - held_reference["panel_delta_auprc"],
        }
    return result


def main() -> None:
    output = {}
    screen = base.load_json(base.RUNS / "screen_v1" / "detection_screen.json")
    screen_rows = screen["parent_rows"]
    output["seed0_profile_screen"] = {
        detector: sorted(
            (
                {
                    "condition": condition,
                    "delta_auprc": score_subset(
                        screen_rows, detector, condition, tuple(base.TUNNELS)
                    )["delta_auprc"],
                    "positive_tunnels": score_subset(
                        screen_rows, detector, condition, tuple(base.TUNNELS)
                    )["positive_tunnels"],
                }
                for condition in (profiles.IDENTITY, *profiles.CANDIDATES)
            ),
            key=lambda item: item["delta_auprc"],
            reverse=True,
        )
        for detector in base.field_common.DETECTORS
    }
    for label, seeds in (("all", (0, 1, 2)), ("held", (1, 2))):
        rows = cycle4._load_rows(seeds)
        scores = {
            detector: {
                condition: cycle4._score(rows, detector, condition)
                for condition in cycle4.FUSIONS
            }
            for detector in base.field_common.DETECTORS
        }
        output[label] = {
            detector: sorted(
                (
                    {
                        "condition": condition,
                        "delta_auprc": values["delta_auprc"],
                        "delta_fpr": values["delta_matched_fpr"],
                        "positive_tunnels": values["positive_tunnels"],
                        "per_tunnel": {
                            tunnel: metrics["delta_auprc"]
                            for tunnel, metrics in values["per_tunnel"].items()
                        },
                    }
                    for condition, values in conditions.items()
                ),
                key=lambda item: item["delta_auprc"],
                reverse=True,
            )
            for detector, conditions in scores.items()
        }
        output[label]["crossfit"] = crossfit(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
