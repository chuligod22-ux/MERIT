"""MERIT v2.4 detection cycle 3: minimal ResNet activation.

Cycle 3 retains the two active v2.3 detector routes and replaces only the
ResNet identity route with the weakest screened unsharp transform.  Seed 0 is
development evidence.  Seeds 1 and 2 are held stochastic replications; they
are not independent sites.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c016_pipeline as base  # noqa: E402


PROTOCOL = base.C016 / "protocol" / "v24_detection_cycle3_protocol_v1.json"
RUN_ROOT = base.RUNS / "detection_cycle3"
SCREEN = base.RUNS / "screen_v1" / "detection_screen.json"
ROUTES = {
    "d2_resnet34_unet": "sharp0025",
    "d2_vgg16_unet": "inv0250",
    "d2_deeplabv3_resnet50": "sharp0100",
}
REFERENCE_ROUTES = {
    "d2_resnet34_unet": base.profiles.IDENTITY,
    "d2_vgg16_unet": "inv0250",
    "d2_deeplabv3_resnet50": "sharp0100",
}


def _seed0_route_score(rows: list[dict[str, Any]], detector: str, condition: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["detector"] == detector and row["condition"] == condition
    ]
    per_tunnel = {
        tunnel: {
            key: base.mean(row[key] for row in selected if row["tunnel"] == tunnel)
            for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
        }
        for tunnel in base.TUNNELS
    }
    return {
        "condition": condition,
        "delta_auprc": base.mean(value["delta_auprc"] for value in per_tunnel.values()),
        "delta_f1": base.mean(value["delta_f1"] for value in per_tunnel.values()),
        "delta_matched_fpr": base.mean(
            value["delta_matched_fpr"] for value in per_tunnel.values()
        ),
        "positive_tunnels": sum(value["delta_auprc"] > 0.0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def freeze() -> dict[str, Any]:
    parent = base.verify_protocol()
    screen = base.load_json(SCREEN)
    if base.payload_sha256(screen) != screen.get("payload_sha256"):
        raise base.C016Error("Cycle-3 seed-0 screen hash mismatch")
    route_scores = {
        detector: _seed0_route_score(screen["parent_rows"], detector, condition)
        for detector, condition in ROUTES.items()
    }
    resnet = route_scores["d2_resnet34_unet"]
    if resnet["positive_tunnels"] != 4:
        raise base.C016Error("Cycle-3 ResNet route is not positive on all seed-0 tunnels")
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-DETECTION-CYCLE3-PROTOCOL-v1",
        "status": "FROZEN_BEFORE_CYCLE3_SEED1_SEED2_OUTPUTS",
        "development_reason": (
            "Retain the validated v2.3 VGG and DeepLab routes. Activate ResNet only with "
            "sharp0025, the smallest non-identity profile and the only screened ResNet route "
            "with positive seed-0 AUPRC deltas on all four tunnels."
        ),
        "scientific_role": (
            "Seed 0 is route-development evidence. Seeds 1 and 2 are held stochastic "
            "replications on the same four development tunnels, not prospective external sites."
        ),
        "routes": ROUTES,
        "matched_reference_routes": REFERENCE_ROUTES,
        "seed0_route_scores": route_scores,
        "evaluation": {
            "development_seed": 0,
            "held_replication_seeds": [1, 2],
            "tunnels": list(base.TUNNELS),
            "aggregation": "seed -> physical parent -> equal tunnel -> equal detector panel",
        },
        "final_gate": (
            "All-seed and held-seed panels must each have delta AUPRC mean and CI lower >0, "
            "all four tunnels positive, matched-sensitivity delta FPR mean <=0, every detector "
            "mean delta AUPRC >=0, and panel delta AUPRC above the matched v2.3 route. The "
            "all-seed panel must also exceed the frozen C014 v2.3 four-tunnel mean."
        ),
        "parent_protocol_payload_sha256": parent["payload_sha256"],
        "assets": {
            "seed0_screen": base.asset_record(SCREEN),
            "v23_c014_summary": base.asset_record(base.C014_SUMMARY),
            "cycle2_summary": base.asset_record(base.RUNS / "detection_cycle2" / "summary.json"),
        },
        "code": {
            "cycle3": base.asset_record(Path(__file__)),
            "profiles": parent["code"]["profiles"],
            "frozen_base_pipeline": parent["code"]["pipeline"],
        },
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    if PROTOCOL.exists():
        existing = base.load_json(PROTOCOL)
        if existing != payload:
            raise base.C016Error("Frozen cycle-3 protocol differs from recomputation")
        return existing
    base.write_json_exclusive(PROTOCOL, payload)
    return payload


def verify() -> dict[str, Any]:
    parent = base.verify_protocol()
    if not PROTOCOL.is_file():
        raise base.C016Error(f"Missing cycle-3 protocol: {PROTOCOL}")
    payload = base.load_json(PROTOCOL)
    if payload.get("schema_version") != "MERIT-C016-V24-DETECTION-CYCLE3-PROTOCOL-v1":
        raise base.C016Error("Unexpected cycle-3 protocol schema")
    if payload.get("status") != "FROZEN_BEFORE_CYCLE3_SEED1_SEED2_OUTPUTS":
        raise base.C016Error("Cycle-3 protocol is not frozen")
    if base.payload_sha256(payload) != payload.get("payload_sha256"):
        raise base.C016Error("Cycle-3 protocol payload hash mismatch")
    if payload["parent_protocol_payload_sha256"] != parent["payload_sha256"]:
        raise base.C016Error("Cycle-3 parent protocol mismatch")
    for group in ("assets", "code"):
        for label, record in payload[group].items():
            base.verify_asset(record, f"cycle3:{group}:{label}")
    if payload["routes"] != ROUTES or payload["matched_reference_routes"] != REFERENCE_ROUTES:
        raise base.C016Error("Cycle-3 routes changed after freeze")
    return payload


def _selection_adapter(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "detection_routes": {
            tunnel: {
                detector: {"selected": protocol["routes"][detector]}
                for detector in base.field_common.DETECTORS
            }
            for tunnel in base.TUNNELS
        }
    }


def _derive_seed0(fold: str, protocol: dict[str, Any]) -> dict[str, Any]:
    screen = base.load_json(SCREEN)
    rows = []
    for detector, condition in protocol["routes"].items():
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
        "source": "cycle-3 selected seed-0 screen rows",
        "routes": protocol["routes"],
        "parent_rows": rows,
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    return payload


def evaluate(batch_size: int = 12) -> dict[str, Any]:
    protocol = verify()
    adapter = _selection_adapter(protocol)
    progress = RUN_ROOT / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    for seed in base.SEEDS:
        for fold in base.TUNNELS:
            path = progress / f"fold_{fold}_seed{seed}.json"
            if not base._valid_payload(path, "MERIT-C016-DETECTION-EVAL-UNIT-v1"):
                result = (
                    _derive_seed0(fold, protocol)
                    if seed == 0
                    else base._detection_eval_unit(fold, seed, adapter, batch_size)
                )
                base.write_json_atomic(path, result)
    files = sorted(progress.glob("fold_*_seed*.json"))
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-CYCLE3-COMPLETE-v1",
        "status": "PASS_COMPLETE",
        "units": len(files),
        "protocol_payload_sha256": protocol["payload_sha256"],
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    base.write_json_atomic(RUN_ROOT / "RUN_COMPLETE.json", payload)
    return payload


def _load_seed_rows(seeds: Iterable[int]) -> list[dict[str, Any]]:
    selected_seeds = set(seeds)
    rows: list[dict[str, Any]] = []
    for path in sorted((RUN_ROOT / "progress").glob("fold_*_seed*.json")):
        payload = base.load_json(path)
        if base.payload_sha256(payload) != payload.get("payload_sha256"):
            raise base.C016Error(f"Cycle-3 unit hash mismatch: {path}")
        if int(payload["seed"]) in selected_seeds:
            rows.extend({**row, "seed": int(payload["seed"])} for row in payload["parent_rows"])
    expected = len(selected_seeds) * len(base.TUNNELS)
    files_for_seeds = {
        (row["tunnel"], row["seed"])
        for row in rows
    }
    if len(files_for_seeds) != expected:
        raise base.C016Error(f"Cycle-3 rows incomplete for seeds {sorted(selected_seeds)}")
    return rows


def _bootstrap_bundle(
    seed_rows: list[dict[str, Any]], seed_offset: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        "delta_auprc": base.stratified_bootstrap(
            panel_rows, "delta_auprc", base.BOOTSTRAP_SEED + seed_offset
        ),
        "delta_f1": base.stratified_bootstrap(
            panel_rows, "delta_f1", base.BOOTSTRAP_SEED + seed_offset + 1
        ),
        "delta_matched_fpr": base.stratified_bootstrap(
            panel_rows, "delta_matched_fpr", base.BOOTSTRAP_SEED + seed_offset + 2
        ),
    }
    detectors = {}
    for index, detector in enumerate(base.field_common.DETECTORS):
        rows = [row for row in detector_parent_rows if row["detector"] == detector]
        detectors[detector] = {
            "delta_auprc": base.stratified_bootstrap(
                rows, "delta_auprc", base.BOOTSTRAP_SEED + seed_offset + 10 + index
            ),
            "delta_f1": base.stratified_bootstrap(
                rows, "delta_f1", base.BOOTSTRAP_SEED + seed_offset + 20 + index
            ),
            "delta_matched_fpr": base.stratified_bootstrap(
                rows, "delta_matched_fpr", base.BOOTSTRAP_SEED + seed_offset + 30 + index
            ),
        }

    reference_detector_rows = []
    for row in detector_parent_rows:
        if row["detector"] == "d2_resnet34_unet":
            reference_detector_rows.append(
                {
                    **row,
                    "delta_auprc": 0.0,
                    "delta_f1": 0.0,
                    "delta_matched_fpr": 0.0,
                }
            )
        else:
            reference_detector_rows.append(row)
    reference_by_parent: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in reference_detector_rows:
        reference_by_parent[(row["tunnel"], row["parent"])].append(row)
    reference_panel_rows = [
        {
            "tunnel": tunnel,
            "parent": parent,
            **{
                key: base.mean(row[key] for row in rows)
                for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
            },
        }
        for (tunnel, parent), rows in sorted(reference_by_parent.items())
    ]
    reference_panel = {
        "delta_auprc": base.stratified_bootstrap(
            reference_panel_rows, "delta_auprc", base.BOOTSTRAP_SEED + seed_offset + 40
        ),
        "delta_f1": base.stratified_bootstrap(
            reference_panel_rows, "delta_f1", base.BOOTSTRAP_SEED + seed_offset + 41
        ),
        "delta_matched_fpr": base.stratified_bootstrap(
            reference_panel_rows, "delta_matched_fpr", base.BOOTSTRAP_SEED + seed_offset + 42
        ),
    }
    return panel, detectors, reference_panel


def _gate(panel: dict[str, Any], detectors: dict[str, Any], reference: dict[str, Any]) -> bool:
    return bool(
        panel["delta_auprc"]["mean"] > reference["delta_auprc"]["mean"]
        and panel["delta_auprc"]["ci95_low"] > 0.0
        and panel["delta_auprc"]["positive_tunnels"] == 4
        and panel["delta_f1"]["mean"] > 0.0
        and panel["delta_matched_fpr"]["mean"] <= 0.0
        and all(value["delta_auprc"]["mean"] >= 0.0 for value in detectors.values())
    )


def summarize() -> dict[str, Any]:
    protocol = verify()
    files = sorted((RUN_ROOT / "progress").glob("fold_*_seed*.json"))
    if len(files) != len(base.TUNNELS) * len(base.SEEDS):
        raise base.C016Error("Cycle-3 evaluation is incomplete")
    all_panel, all_detectors, all_reference = _bootstrap_bundle(
        _load_seed_rows(base.SEEDS), 300
    )
    held_panel, held_detectors, held_reference = _bootstrap_bundle(
        _load_seed_rows((1, 2)), 400
    )
    frozen_v23 = float(
        base.load_json(base.C014_SUMMARY)["detection"]
        ["existing_four_tunnel_descriptive_delta_auprc"]["mean"]
    )
    all_gate = _gate(all_panel, all_detectors, all_reference)
    held_gate = _gate(held_panel, held_detectors, held_reference)
    superiority = all_panel["delta_auprc"]["mean"] > frozen_v23
    efficacy_pass = bool(all_gate and held_gate and superiority)
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-CYCLE3-SUMMARY-v1",
        "status": "PASS_COMPLETE" if efficacy_pass else "COMPLETE_GATE_FAIL",
        "scientific_status": "SEED0_DEVELOPMENT_WITH_SEED1_SEED2_HELD_REPLICATION",
        "routes": protocol["routes"],
        "all_seeds": {
            "panel": all_panel,
            "detectors": all_detectors,
            "matched_v23_reference_panel": all_reference,
            "gate_pass": all_gate,
        },
        "held_replication_seeds_1_2": {
            "panel": held_panel,
            "detectors": held_detectors,
            "matched_v23_reference_panel": held_reference,
            "gate_pass": held_gate,
        },
        "frozen_v23_four_tunnel_mean_delta_auprc": frozen_v23,
        "delta_over_frozen_v23": all_panel["delta_auprc"]["mean"] - frozen_v23,
        "superiority_over_frozen_v23": superiority,
        "efficacy_gate_pass": efficacy_pass,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "claim_boundary": (
            "Cycle 3 is iterative development on the supplied four tunnels. Held seeds test "
            "stochastic replication, not new-site generalisation."
        ),
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
