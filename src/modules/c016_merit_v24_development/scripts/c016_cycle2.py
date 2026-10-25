"""Second v2.4 detection iteration: conservative global seed-held route.

Cycle 1 showed that aggressive tunnel-conditioned coefficients did not
transfer to the excluded tunnel.  This iteration selects one profile per
detector on seed 0 using all four supplied tunnels, with a four-tunnel panel
positivity constraint.  Seeds 1 and 2 are then evaluated independently.  The
natural apparent-width route and results from cycle 1 remain frozen.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c016_pipeline as base
import c016_profiles as profiles


PROTOCOL = base.C016 / "protocol" / "v24_detection_cycle2_protocol_v1.json"
RUN_ROOT = base.RUNS / "detection_cycle2"
SCREEN = base.RUNS / "screen_v1" / "detection_screen.json"


def tunnel_metric(
    rows: list[dict[str, Any]],
    tunnel: str,
    detector: str,
    condition: str,
    key: str,
) -> float:
    selected = [
        row
        for row in rows
        if row["tunnel"] == tunnel
        and row["detector"] == detector
        and row["condition"] == condition
    ]
    return base.mean(float(row[key]) for row in selected)


def select_joint_routes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best seed-0 joint panel under frozen robustness gates."""

    conditions = (profiles.IDENTITY, *profiles.CANDIDATES)
    detectors = tuple(base.field_common.DETECTORS)
    candidates = []
    for route_tuple in itertools.product(conditions, repeat=len(detectors)):
        detector_metrics = {}
        for detector, condition in zip(detectors, route_tuple):
            per_tunnel_auprc = {
                tunnel: tunnel_metric(
                    rows, tunnel, detector, condition, "delta_auprc"
                )
                for tunnel in base.TUNNELS
            }
            per_tunnel_fpr = {
                tunnel: tunnel_metric(
                    rows, tunnel, detector, condition, "delta_matched_fpr"
                )
                for tunnel in base.TUNNELS
            }
            detector_metrics[detector] = {
                "condition": condition,
                "delta_auprc": base.mean(per_tunnel_auprc.values()),
                "delta_matched_fpr": base.mean(per_tunnel_fpr.values()),
                "positive_tunnels": sum(value > 0.0 for value in per_tunnel_auprc.values()),
                "per_tunnel_delta_auprc": per_tunnel_auprc,
                "per_tunnel_delta_matched_fpr": per_tunnel_fpr,
            }
        panel_per_tunnel = {
            tunnel: base.mean(
                detector_metrics[detector]["per_tunnel_delta_auprc"][tunnel]
                for detector in detectors
            )
            for tunnel in base.TUNNELS
        }
        panel_auprc = base.mean(
            detector_metrics[detector]["delta_auprc"] for detector in detectors
        )
        panel_fpr = base.mean(
            detector_metrics[detector]["delta_matched_fpr"] for detector in detectors
        )
        eligible = bool(
            panel_fpr <= 0.0
            and all(value > 0.0 for value in panel_per_tunnel.values())
            and all(
                value["delta_auprc"] >= 0.0
                and (
                    value["condition"] == profiles.IDENTITY
                    or value["positive_tunnels"] >= 2
                )
                for value in detector_metrics.values()
            )
        )
        if eligible:
            candidates.append(
                {
                    "routes": {
                        detector: detector_metrics[detector]["condition"]
                        for detector in detectors
                    },
                    "detectors": detector_metrics,
                    "panel_delta_auprc": panel_auprc,
                    "panel_delta_matched_fpr": panel_fpr,
                    "panel_per_tunnel_delta_auprc": panel_per_tunnel,
                    "magnitude": sum(profiles.profile_magnitude(condition) for condition in route_tuple),
                }
            )
    if not candidates:
        raise base.C016Error("Cycle-2 joint selector found no eligible route")
    return max(
        candidates,
        key=lambda item: (
            float(item["panel_delta_auprc"]),
            -float(item["magnitude"]),
            json.dumps(item["routes"], sort_keys=True),
        ),
    )


def freeze() -> dict[str, Any]:
    base_protocol = base.verify_protocol()
    screen = base.load_json(SCREEN)
    if base.payload_sha256(screen) != screen.get("payload_sha256"):
        raise base.C016Error("Cycle-2 source screen hash mismatch")
    selection = select_joint_routes(screen["parent_rows"])
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-DETECTION-CYCLE2-PROTOCOL-v1",
        "status": "FROZEN_BEFORE_CYCLE2_SEED1_SEED2_OUTPUTS",
        "development_reason": (
            "Cycle-1 outer-fold coefficient selection failed to transfer. Replace it with a "
            "single conservative detector-panel route selected on seed 0 under four-tunnel "
            "panel positivity and false-positive constraints."
        ),
        "scientific_role": (
            "All four sites are development data for route selection; seeds 1 and 2 are "
            "stochastic replication, not new-site external validation."
        ),
        "selection_data": "seed-0 all-four-tunnel screen only",
        "evaluation_data": "same four tunnels with restorer/detector seeds 1 and 2; seed 0 retained in final aggregate",
        "selection_rule": {
            "objective": "maximize equal-detector, equal-tunnel panel delta AUPRC",
            "constraints": (
                "seed-0 panel delta matched FPR <=0; every tunnel panel delta AUPRC >0; "
                "every detector mean delta AUPRC >=0; each active detector route positive "
                "on at least 2/4 tunnels"
            ),
            "tie_break": "smaller summed profile magnitude",
        },
        "selection": selection,
        "final_gate": (
            "three-seed panel delta AUPRC mean and 95% CI lower >0; all four tunnels positive; "
            "panel matched-sensitivity delta FPR mean <=0; every detector mean delta AUPRC >=0"
        ),
        "parent_protocol_payload_sha256": base_protocol["payload_sha256"],
        "assets": {
            "seed0_screen": base.asset_record(SCREEN),
            "v23_c014_summary": base.asset_record(base.C014_SUMMARY),
            "cycle1_final_summary": base.asset_record(base.RUNS / "c016_final_summary.json"),
        },
        "code": {
            "cycle2": base.asset_record(Path(__file__)),
            "profiles": base.asset_record(HERE / "c016_profiles.py"),
            "frozen_cycle1_pipeline": base_protocol["code"]["pipeline"],
        },
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    if PROTOCOL.exists():
        existing = base.load_json(PROTOCOL)
        if existing != payload:
            raise base.C016Error("Cycle-2 frozen protocol differs from recomputation")
        return existing
    base.write_json_exclusive(PROTOCOL, payload)
    return payload


def verify() -> dict[str, Any]:
    parent = base.verify_protocol()
    payload = base.load_json(PROTOCOL)
    if payload.get("schema_version") != "MERIT-C016-V24-DETECTION-CYCLE2-PROTOCOL-v1":
        raise base.C016Error("Unexpected cycle-2 protocol schema")
    if payload.get("status") != "FROZEN_BEFORE_CYCLE2_SEED1_SEED2_OUTPUTS":
        raise base.C016Error("Cycle-2 protocol is not frozen")
    if base.payload_sha256(payload) != payload.get("payload_sha256"):
        raise base.C016Error("Cycle-2 protocol payload hash mismatch")
    if payload["parent_protocol_payload_sha256"] != parent["payload_sha256"]:
        raise base.C016Error("Cycle-2 parent protocol mismatch")
    for group in ("assets", "code"):
        for label, record in payload[group].items():
            base.verify_asset(record, f"cycle2:{group}:{label}")
    return payload


def selection_adapter(protocol: dict[str, Any]) -> dict[str, Any]:
    routes = protocol["selection"]["routes"]
    return {
        "detection_routes": {
            tunnel: {
                detector: {"selected": routes[detector]}
                for detector in base.field_common.DETECTORS
            }
            for tunnel in base.TUNNELS
        }
    }


def derive_seed0(fold: str, protocol: dict[str, Any]) -> dict[str, Any]:
    screen = base.load_json(SCREEN)
    routes = protocol["selection"]["routes"]
    rows = []
    for detector, condition in routes.items():
        rows.extend(
            row
            for row in screen["parent_rows"]
            if row["tunnel"] == fold
            and row["detector"] == detector
            and row["condition"] == condition
        )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-EVAL-UNIT-v1",
        "status": "PASS",
        "fold": fold,
        "seed": 0,
        "source": "cycle-2 selected seed-0 screen rows",
        "routes": routes,
        "parent_rows": rows,
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    return payload


def evaluate(batch_size: int = 12) -> dict[str, Any]:
    protocol = verify()
    adapter = selection_adapter(protocol)
    progress = RUN_ROOT / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    for seed in base.SEEDS:
        for fold in base.TUNNELS:
            path = progress / f"fold_{fold}_seed{seed}.json"
            if not base._valid_payload(path, "MERIT-C016-DETECTION-EVAL-UNIT-v1"):
                payload = (
                    derive_seed0(fold, protocol)
                    if seed == 0
                    else base._detection_eval_unit(fold, seed, adapter, batch_size)
                )
                base.write_json_atomic(path, payload)
    files = sorted(progress.glob("fold_*_seed*.json"))
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-CYCLE2-COMPLETE-v1",
        "status": "PASS_COMPLETE",
        "units": len(files),
        "protocol_payload_sha256": protocol["payload_sha256"],
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    base.write_json_atomic(RUN_ROOT / "RUN_COMPLETE.json", payload)
    return payload


def summarize() -> dict[str, Any]:
    protocol = verify()
    files = sorted((RUN_ROOT / "progress").glob("fold_*_seed*.json"))
    if len(files) != len(base.TUNNELS) * len(base.SEEDS):
        raise base.C016Error("Cycle-2 evaluation is incomplete")
    seed_rows = []
    for path in files:
        payload = base.load_json(path)
        if base.payload_sha256(payload) != payload.get("payload_sha256"):
            raise base.C016Error(f"Cycle-2 unit hash mismatch: {path}")
        seed_rows.extend({**row, "seed": payload["seed"]} for row in payload["parent_rows"])
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in seed_rows:
        grouped[(row["tunnel"], row["detector"], row["parent"])].append(row)
    detector_parent_rows = [
        {
            "tunnel": tunnel,
            "detector": detector,
            "parent": parent,
            **{
                key: base.mean(row[key] for row in rows)
                for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
            },
        }
        for (tunnel, detector, parent), rows in sorted(grouped.items())
    ]
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in detector_parent_rows:
        by_parent[(row["tunnel"], row["parent"])].append(row)
    panel_rows = [
        {
            "tunnel": tunnel,
            "parent": parent,
            **{
                key: base.mean(row[key] for row in rows)
                for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
            },
        }
        for (tunnel, parent), rows in sorted(by_parent.items())
    ]
    panel = {
        "delta_auprc": base.stratified_bootstrap(panel_rows, "delta_auprc", base.BOOTSTRAP_SEED + 200),
        "delta_f1": base.stratified_bootstrap(panel_rows, "delta_f1", base.BOOTSTRAP_SEED + 201),
        "delta_matched_fpr": base.stratified_bootstrap(panel_rows, "delta_matched_fpr", base.BOOTSTRAP_SEED + 202),
    }
    detectors = {}
    for index, detector in enumerate(base.field_common.DETECTORS):
        rows = [row for row in detector_parent_rows if row["detector"] == detector]
        detectors[detector] = {
            "delta_auprc": base.stratified_bootstrap(rows, "delta_auprc", base.BOOTSTRAP_SEED + 210 + index),
            "delta_f1": base.stratified_bootstrap(rows, "delta_f1", base.BOOTSTRAP_SEED + 220 + index),
            "delta_matched_fpr": base.stratified_bootstrap(rows, "delta_matched_fpr", base.BOOTSTRAP_SEED + 230 + index),
        }
    efficacy_pass = bool(
        panel["delta_auprc"]["mean"] > 0.0
        and panel["delta_auprc"]["ci95_low"] > 0.0
        and panel["delta_auprc"]["positive_tunnels"] == 4
        and panel["delta_matched_fpr"]["mean"] <= 0.0
        and all(value["delta_auprc"]["mean"] >= 0.0 for value in detectors.values())
    )
    v23_mean = float(
        base.load_json(base.C014_SUMMARY)["detection"]["existing_four_tunnel_descriptive_delta_auprc"]["mean"]
    )
    width = base.load_json(base.RUNS / "c016_final_summary.json")["natural_crack_apparent_width"]
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-CYCLE2-SUMMARY-v1",
        "status": "PASS_COMPLETE" if efficacy_pass and width["gate"]["pass"] else "COMPLETE_GATE_FAIL",
        "scientific_status": "ALL_SITE_DEVELOPMENT_WITH_SEED_HELD_REPLICATION",
        "routes": protocol["selection"]["routes"],
        "detection": {
            "panel": panel,
            "detectors": detectors,
            "efficacy_gate_pass": efficacy_pass,
            "v23_four_tunnel_mean_delta_auprc": v23_mean,
            "delta_over_v23": panel["delta_auprc"]["mean"] - v23_mean,
            "superiority_over_v23": panel["delta_auprc"]["mean"] > v23_mean,
        },
        "natural_crack_apparent_width": width,
        "global_gate_pass": bool(efficacy_pass and width["gate"]["pass"]),
        "claim_boundary": base.load_json(base.PROTOCOL)["claim_boundary"],
        "protocol_payload_sha256": protocol["payload_sha256"],
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    base.write_json_atomic(RUN_ROOT / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "eval", "summarize", "run"))
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "eval":
        result = evaluate(args.batch_size)
    elif args.command == "summarize":
        result = summarize()
    else:
        freeze()
        evaluate(args.batch_size)
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
