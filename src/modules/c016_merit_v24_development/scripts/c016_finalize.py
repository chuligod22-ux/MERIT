"""Finalize the endpoint-wise MERIT v2.4 release evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c016_pipeline as base  # noqa: E402
import c016_release as release  # noqa: E402


OUTPUT = base.RUNS / "c016_release_v24_summary.json"
C012_HOLDOUT = base.MODIFIED / "c012_field_model" / "runs" / "holdout_v1" / "summary.json"
CYCLE1 = base.RUNS / "c016_final_summary.json"
CYCLE2 = base.RUNS / "detection_cycle2" / "summary.json"
CYCLE3 = base.RUNS / "detection_cycle3" / "summary.json"
CYCLE4 = base.RUNS / "detection_cycle4" / "summary.json"
CYCLE5 = base.RUNS / "detection_cycle5" / "summary.json"
CYCLE5_PROTOCOL = base.C016 / "protocol" / "v24_detection_cycle5_protocol_v1.json"
CYCLE5_SELECTION = base.C016 / "protocol" / "v24_detection_cycle5_selection_v1.json"


def finalize() -> dict[str, Any]:
    base_protocol = base.verify_protocol()
    holdout = base.load_json(C012_HOLDOUT)
    c014 = base.load_json(base.C014_SUMMARY)
    cycle1 = base.load_json(CYCLE1)
    cycle2 = base.load_json(CYCLE2)
    cycle3 = base.load_json(CYCLE3)
    cycle4 = base.load_json(CYCLE4)
    cycle5 = base.load_json(CYCLE5)
    if cycle5.get("status") != "PASS_COMPLETE" or not cycle5.get("efficacy_gate_pass"):
        raise base.C016Error("Cannot release v2.4: cycle-5 detection gate is not PASS")
    width = cycle1["natural_crack_apparent_width"]
    if not bool(width["gate"]["pass"]):
        raise base.C016Error("Cannot release v2.4: C016 natural-width gate is not PASS")

    v23_width = c014["natural_crack_apparent_width"]
    v23_detection = float(cycle5["frozen_v23_four_tunnel_mean_delta_auprc"])
    all_seeds = cycle5["all_seeds"]
    held_seeds = cycle5["held_replication_seeds_1_2"]
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-RELEASE-v2.4",
        "status": "SUPPORTED_WITH_LIMITS",
        "version": release.VERSION,
        "release_decision": (
            "Adopt the cycle-5 detector-conditioned profile/probability-fusion component and "
            "the v2.4 natural apparent-width component."
        ),
        "components": {
            "crack_detection": {
                "version": release.DETECTION_VERSION,
                "profiles": release.DETECTION_PROFILES,
                "probability_fusion": {
                    "family": release.DETECTION_FUSION_FAMILY,
                    "weights": release.DETECTION_FUSION_WEIGHTS,
                },
                "adoption": "ADOPTED",
                "gate_pass": True,
                "coverage": {
                    "tiles": 1703,
                    "physical_parents": 14,
                    "tunnels": 4,
                    "seeds": 3,
                },
                "all_seeds": {
                    "panel": all_seeds["panel"],
                    "detectors": all_seeds["detectors"],
                    "selected_minus_cycle3_delta_auprc": all_seeds[
                        "selected_minus_cycle3_delta_auprc"
                    ],
                    "panel_gate_pass": all_seeds["panel_gate_pass"],
                    "superiority_gate_pass": all_seeds["superiority_gate_pass"],
                },
                "held_replication_seeds_1_2": {
                    "panel": held_seeds["panel"],
                    "detectors": held_seeds["detectors"],
                    "selected_minus_cycle3_delta_auprc": held_seeds[
                        "selected_minus_cycle3_delta_auprc"
                    ],
                    "panel_gate_pass": held_seeds["panel_gate_pass"],
                    "superiority_gate_pass": held_seeds["superiority_gate_pass"],
                },
                "change_from_v23_four_tunnel_mean_delta_auprc": (
                    float(all_seeds["panel"]["delta_auprc"]["mean"]) - v23_detection
                ),
                "v23_four_tunnel_mean_delta_auprc": v23_detection,
                "legacy_v23_sealed_internal_holdout_C_D": {
                    "delta_auprc": holdout["d2"]["panel"]["delta_auprc"],
                    "delta_f1": holdout["d2"]["panel"]["delta_f1"],
                    "delta_matched_fpr": holdout["d2"]["panel"]["delta_matched_fpr"],
                    "gate_pass": bool(holdout["d2"]["gate"]["pass"]),
                },
            },
            "natural_crack_apparent_width": {
                "version": release.VERSION,
                "profile": release.WIDTH_PROFILE,
                "safe_fallback": release.WIDTH_SAFE_FALLBACK,
                "adoption": "ADOPTED",
                "gate_pass": True,
                "coverage": width["coverage"],
                "delta_absolute_error_px": width["delta_absolute_error_px"],
                "raw_nonestimable_rate": width["raw_nonestimable_rate"],
                "effective_nonestimable_rate": width["effective_nonestimable_rate"],
                "paired_estimable_rate": width["paired_estimable_rate"],
                "safe_fallback_rois": width["safe_fallback_rois"],
                "change_from_v23": {
                    "delta_absolute_error_px": (
                        float(width["delta_absolute_error_px"]["mean"])
                        - float(v23_width["delta_absolute_error_px"]["mean"])
                    ),
                    "paired_estimable_rate": (
                        float(width["paired_estimable_rate"])
                        - float(v23_width["paired_estimable_rate"])
                    ),
                    "v23_gate_pass": bool(v23_width["gate"]["pass"]),
                    "v24_gate_pass": True,
                },
            },
        },
        "experimental_detection_cycles": {
            "cycle1": {
                "adopted": False,
                "gate_pass": cycle1["detection"]["gate"]["pass"],
            },
            "cycle2": {
                "adopted": False,
                "efficacy_gate_pass": cycle2["detection"]["efficacy_gate_pass"],
            },
            "cycle3": {
                "adopted": False,
                "superseded_by": "cycle5",
                "efficacy_gate_pass": cycle3["efficacy_gate_pass"],
            },
            "cycle4": {
                "adopted": False,
                "efficacy_gate_pass": cycle4["efficacy_gate_pass"],
            },
            "cycle5": {
                "adopted": True,
                "routes": cycle5["routes"],
                "efficacy_gate_pass": cycle5["efficacy_gate_pass"],
            },
        },
        "claim_boundary": (
            "v2.4 supports crack-detection assistance and expert-mask-compatible natural crack "
            "apparent-width measurement in pixels on the supplied four-tunnel data. Seed 1/2 "
            "are held stochastic replications, not independent sites. The evidence does not "
            "establish traceable physical millimetre width or prospective new-site efficacy."
        ),
        "assets": {
            "c012_holdout_summary": base.asset_record(C012_HOLDOUT),
            "c014_summary": base.asset_record(base.C014_SUMMARY),
            "c016_cycle1_summary": base.asset_record(CYCLE1),
            "c016_cycle2_summary": base.asset_record(CYCLE2),
            "c016_cycle3_summary": base.asset_record(CYCLE3),
            "c016_cycle4_summary": base.asset_record(CYCLE4),
            "c016_cycle5_summary": base.asset_record(CYCLE5),
            "c016_cycle5_protocol": base.asset_record(CYCLE5_PROTOCOL),
            "c016_cycle5_selection": base.asset_record(CYCLE5_SELECTION),
            "release_code": base.asset_record(HERE / "c016_release.py"),
            "base_protocol": base.asset_record(base.PROTOCOL),
        },
        "base_protocol_payload_sha256": base_protocol["payload_sha256"],
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    base.write_json_atomic(OUTPUT, payload)
    return payload


def main() -> None:
    print(json.dumps(finalize(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
