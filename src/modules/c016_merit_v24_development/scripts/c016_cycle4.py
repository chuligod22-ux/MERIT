"""MERIT v2.4 detection cycle 4: raw/profile probability safety fusion."""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c016_pipeline as base  # noqa: E402


PROTOCOL = base.C016 / "protocol" / "v24_detection_cycle4_protocol_v1.json"
SELECTION = base.C016 / "protocol" / "v24_detection_cycle4_selection_v1.json"
RUN_ROOT = base.RUNS / "detection_cycle4"
PROFILES = {
    "d2_resnet34_unet": "sharp0025",
    "d2_vgg16_unet": "inv0250",
    "d2_deeplabv3_resnet50": "sharp0100",
}
FUSIONS: dict[str, dict[str, float | str]] = {
    "identity": {"family": "identity", "weight": 0.0},
    "linear025": {"family": "linear", "weight": 0.25},
    "linear050": {"family": "linear", "weight": 0.50},
    "linear075": {"family": "linear", "weight": 0.75},
    "full": {"family": "linear", "weight": 1.00},
    "dark025": {"family": "dark_bidir", "weight": 0.25},
    "dark050": {"family": "dark_bidir", "weight": 0.50},
    "dark075": {"family": "dark_bidir", "weight": 0.75},
    "dark100": {"family": "dark_bidir", "weight": 1.00},
    "darkpos050": {"family": "dark_positive", "weight": 0.50},
    "darkpos100": {"family": "dark_positive", "weight": 1.00},
}
DARK_SCALE = 0.04


def _dark_gate(raw: np.ndarray) -> np.ndarray:
    normalized = raw.astype(np.float32) / np.float32(255.0)
    blurred = np.stack(
        [
            cv2.GaussianBlur(
                image,
                (7, 7),
                sigmaX=1.0,
                sigmaY=1.0,
                borderType=cv2.BORDER_REFLECT_101,
            )
            for image in normalized
        ]
    )
    return np.clip((blurred - normalized) / np.float32(DARK_SCALE), 0.0, 1.0)


def fuse_probabilities(
    raw_probability: np.ndarray,
    profile_probability: np.ndarray,
    dark_gate: np.ndarray,
    condition: str,
) -> np.ndarray:
    specification = FUSIONS[condition]
    family = str(specification["family"])
    weight = np.float32(specification["weight"])
    difference = profile_probability - raw_probability
    if family == "identity":
        result = raw_probability
    elif family == "linear":
        result = raw_probability + weight * difference
    elif family == "dark_bidir":
        result = raw_probability + weight * dark_gate * difference
    elif family == "dark_positive":
        result = raw_probability + weight * dark_gate * np.maximum(difference, 0.0)
    else:
        raise base.C016Error(f"Unknown cycle-4 fusion family: {family}")
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def freeze_protocol() -> dict[str, Any]:
    parent = base.verify_protocol()
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-DETECTION-CYCLE4-PROTOCOL-v1",
        "status": "FROZEN_BEFORE_CYCLE4_PROBABILITY_OUTPUTS",
        "development_reason": (
            "Cycle 3 passed with a small ResNet gain. Test reference-blind probability fusion "
            "between raw and the cycle-3 profile so that detector evidence outside raw dark-line "
            "support can fall back toward raw."
        ),
        "scientific_role": (
            "All four tunnels remain iterative development data. Seed 0 selects one global "
            "detector-route tuple; seeds 1 and 2 are held stochastic replications."
        ),
        "profiles": PROFILES,
        "fusions": FUSIONS,
        "dark_gate": {
            "formula": "clip((gaussian_blur_7x7_sigma1(raw)-raw)/0.04, 0, 1)",
            "reference_or_gt_access": False,
        },
        "selection": {
            "data": "seed 0, all four development tunnels",
            "objective": "maximum equal-detector equal-tunnel panel delta AUPRC",
            "constraints": (
                "panel matched-sensitivity delta FPR <=0; every tunnel panel delta AUPRC >0; "
                "every detector mean delta AUPRC >=0; each active detector route positive on "
                "at least 2/4 tunnels; panel delta AUPRC greater than cycle-3 full-profile route"
            ),
            "tie_break": "smaller total fusion magnitude",
        },
        "final_gate": (
            "All-seed and held-seed selected panels must pass positive-CI, four-tunnel-positive, "
            "FPR-nonworsening and every-detector-nonnegative gates. Paired selected-minus-cycle3 "
            "panel delta AUPRC must have mean and CI lower >0 in both analyses and improve at "
            "least 3/4 tunnels. All-seed selected panel must exceed frozen C014 v2.3."
        ),
        "parent_protocol_payload_sha256": parent["payload_sha256"],
        "assets": {
            "cycle3_protocol": base.asset_record(
                base.C016 / "protocol" / "v24_detection_cycle3_protocol_v1.json"
            ),
            "cycle3_summary": base.asset_record(base.RUNS / "detection_cycle3" / "summary.json"),
            "v23_c014_summary": base.asset_record(base.C014_SUMMARY),
        },
        "code": {
            "cycle4": base.asset_record(Path(__file__)),
            "profiles": parent["code"]["profiles"],
            "frozen_base_pipeline": parent["code"]["pipeline"],
        },
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    if PROTOCOL.exists():
        existing = base.load_json(PROTOCOL)
        if existing != payload:
            raise base.C016Error("Frozen cycle-4 protocol differs from recomputation")
        return existing
    base.write_json_exclusive(PROTOCOL, payload)
    return payload


def verify_protocol() -> dict[str, Any]:
    parent = base.verify_protocol()
    if not PROTOCOL.is_file():
        raise base.C016Error(f"Missing cycle-4 protocol: {PROTOCOL}")
    payload = base.load_json(PROTOCOL)
    if payload.get("schema_version") != "MERIT-C016-V24-DETECTION-CYCLE4-PROTOCOL-v1":
        raise base.C016Error("Unexpected cycle-4 protocol schema")
    if payload.get("status") != "FROZEN_BEFORE_CYCLE4_PROBABILITY_OUTPUTS":
        raise base.C016Error("Cycle-4 protocol is not frozen")
    if base.payload_sha256(payload) != payload.get("payload_sha256"):
        raise base.C016Error("Cycle-4 protocol payload hash mismatch")
    if payload["parent_protocol_payload_sha256"] != parent["payload_sha256"]:
        raise base.C016Error("Cycle-4 parent protocol mismatch")
    for group in ("assets", "code"):
        for label, record in payload[group].items():
            base.verify_asset(record, f"cycle4:{group}:{label}")
    if payload["profiles"] != PROFILES or payload["fusions"] != FUSIONS:
        raise base.C016Error("Cycle-4 definitions changed after freeze")
    return payload


def _unit(fold: str, seed: int, batch_size: int) -> dict[str, Any]:
    c009_common, d2_eval, multi, detector_protocol, d2, cv, external, tail = (
        base._detection_modules()
    )
    legacy, external_protocol = tail
    c011_protocol = base.field_common.load_protocol()
    cohort = base.field_common.read_csv(base.D2_COHORT)
    fold_rows = [row for row in cohort if row["tunnel"] == fold]
    if not fold_rows:
        raise base.C016Error(f"Empty D2 fold: {fold}")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise base.C016Error("CUDA is required for cycle 4")
    model, provenance = base.c010.load_model(seed, device)
    detectors = d2_eval.load_detectors(
        fold,
        base.DETECTOR_SEEDS[seed],
        multi,
        detector_protocol,
        d2,
        legacy,
        external_protocol,
        device,
    )
    dataset = external.ExternalSyntheticDataset(fold_rows, external_protocol)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    bins = int(detector_protocol["detectors"]["histogram_bins"])
    threshold = float(detector_protocol["detectors"]["fixed_threshold"])
    accumulators = {
        detector: collections.defaultdict(
            lambda: {condition: external.Accumulator(bins) for condition in FUSIONS}
        )
        for detector in base.field_common.DETECTORS
    }
    started = time.perf_counter()
    for batch_index, (image_tensor, degraded_tensor, mask_tensor, parents, metadata) in enumerate(
        loader, 1
    ):
        del degraded_tensor, metadata
        raw = image_tensor.numpy().astype(np.uint8, copy=False)
        masks = mask_tensor.numpy().astype(np.uint8, copy=False)
        normalized = raw.astype(np.float32) / np.float32(255.0)
        zeros = np.zeros(len(raw), dtype=np.float32)
        tensor = torch.from_numpy(c009_common.prompt_tensor(normalized, zeros, zeros)).to(device)
        with base.c010.pilot.autocast(device):
            transformed = base.profiles.make_candidates(
                tensor, model.base, sorted(set(PROFILES.values()))
            )
        encoded = base.encode_candidates(transformed)
        gate = _dark_gate(raw)
        for detector_name, detector in detectors.items():
            profile_image = encoded[PROFILES[detector_name]]
            probabilities = base.detect_candidate_map(
                {"raw": raw, "profile": profile_image},
                detector,
                device,
                external,
                conditions_per_call=2,
            )
            fused = {
                condition: fuse_probabilities(
                    probabilities["raw"], probabilities["profile"], gate, condition
                )
                for condition in FUSIONS
            }
            for sample_index, parent in enumerate(parents):
                for condition, probability in fused.items():
                    candidate_image = raw if condition == "identity" else profile_image
                    external.update_accumulator(
                        accumulators[detector_name][parent][condition],
                        probability[sample_index : sample_index + 1],
                        masks[sample_index : sample_index + 1],
                        raw[sample_index : sample_index + 1],
                        candidate_image[sample_index : sample_index + 1],
                        cv,
                        threshold,
                        False,
                    )
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(
                json.dumps(
                    {
                        "stage": "cycle4",
                        "fold": fold,
                        "seed": seed,
                        "batch": batch_index,
                        "batches": len(loader),
                    }
                ),
                flush=True,
            )
    parent_rows: list[dict[str, Any]] = []
    for detector_name, by_parent in accumulators.items():
        for parent, by_condition in by_parent.items():
            metrics = {
                condition: external.finalize_accumulator(accumulator, d2, cv)
                for condition, accumulator in by_condition.items()
            }
            raw_metric = metrics["identity"]
            for condition in FUSIONS:
                burden = d2_eval.matched_sensitivity_burden(
                    by_condition[condition], raw_metric
                )
                parent_rows.append(
                    {
                        "tunnel": fold,
                        "detector": detector_name,
                        "parent": parent,
                        "condition": condition,
                        "delta_auprc": float(
                            metrics[condition]["pixel_auprc"] - raw_metric["pixel_auprc"]
                        ),
                        "delta_f1": float(
                            metrics[condition]["f1_at_0_5"] - raw_metric["f1_at_0_5"]
                        ),
                        "delta_matched_fpr": float(
                            burden["delta_false_positive_pixel_rate_restored_minus_degraded"]
                        ),
                    }
                )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-CYCLE4-UNIT-v1",
        "status": "PASS",
        "fold": fold,
        "seed": seed,
        "tile_count": len(dataset),
        "parents": len({row["parent"] for row in parent_rows}),
        "profiles": PROFILES,
        "parent_rows": parent_rows,
        "base_provenance": provenance,
        "c011_protocol_payload_sha256": c011_protocol["payload_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    del model, detectors, dataset, loader, accumulators
    torch.cuda.empty_cache()
    return payload


def evaluate_seed(seed: int, batch_size: int) -> dict[str, Any]:
    verify_protocol()
    progress = RUN_ROOT / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for fold in base.TUNNELS:
        path = progress / f"fold_{fold}_seed{seed}.json"
        if not base._valid_payload(path, "MERIT-C016-DETECTION-CYCLE4-UNIT-v1"):
            base.write_json_atomic(path, _unit(fold, seed, batch_size))
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-DETECTION-CYCLE4-SEED-COMPLETE-v1",
        "status": "PASS_COMPLETE",
        "seed": seed,
        "units": len(list(progress.glob(f"fold_*_seed{seed}.json"))),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    base.write_json_atomic(RUN_ROOT / f"SEED_{seed}_COMPLETE.json", payload)
    return payload


def _score(rows: list[dict[str, Any]], detector: str, condition: str) -> dict[str, Any]:
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
        **{
            key: base.mean(value[key] for value in per_tunnel.values())
            for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
        },
        "positive_tunnels": sum(value["delta_auprc"] > 0.0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def _tuple_score(scores: dict[str, dict[str, Any]], route_tuple: tuple[str, ...]) -> dict[str, Any]:
    detectors = tuple(base.field_common.DETECTORS)
    metrics = {detector: scores[detector][condition] for detector, condition in zip(detectors, route_tuple)}
    per_tunnel = {
        tunnel: base.mean(metrics[detector]["per_tunnel"][tunnel]["delta_auprc"] for detector in detectors)
        for tunnel in base.TUNNELS
    }
    return {
        "routes": dict(zip(detectors, route_tuple)),
        "detectors": metrics,
        "panel_delta_auprc": base.mean(value["delta_auprc"] for value in metrics.values()),
        "panel_delta_matched_fpr": base.mean(
            value["delta_matched_fpr"] for value in metrics.values()
        ),
        "panel_per_tunnel_delta_auprc": per_tunnel,
        "magnitude": sum(float(FUSIONS[condition]["weight"]) for condition in route_tuple),
    }


def select() -> dict[str, Any]:
    protocol = verify_protocol()
    files = sorted((RUN_ROOT / "progress").glob("fold_*_seed0.json"))
    if len(files) != len(base.TUNNELS):
        raise base.C016Error("Cycle-4 seed-0 screen is incomplete")
    rows: list[dict[str, Any]] = []
    hashes = {}
    for path in files:
        payload = base.load_json(path)
        if base.payload_sha256(payload) != payload.get("payload_sha256"):
            raise base.C016Error(f"Cycle-4 seed-0 unit hash mismatch: {path}")
        rows.extend(payload["parent_rows"])
        hashes[path.name] = base.sha256_file(path)
    scores = {
        detector: {
            condition: _score(rows, detector, condition) for condition in FUSIONS
        }
        for detector in base.field_common.DETECTORS
    }
    reference = _tuple_score(scores, tuple("full" for _ in base.field_common.DETECTORS))
    eligible = []
    for route_tuple in itertools.product(FUSIONS, repeat=len(base.field_common.DETECTORS)):
        candidate = _tuple_score(scores, route_tuple)
        if (
            candidate["panel_delta_matched_fpr"] <= 0.0
            and candidate["panel_delta_auprc"] > reference["panel_delta_auprc"]
            and all(value > 0.0 for value in candidate["panel_per_tunnel_delta_auprc"].values())
            and all(
                value["delta_auprc"] >= 0.0
                and (condition == "identity" or value["positive_tunnels"] >= 2)
                for condition, value in zip(route_tuple, candidate["detectors"].values())
            )
        ):
            eligible.append(candidate)
    selected = (
        max(
            eligible,
            key=lambda item: (
                float(item["panel_delta_auprc"]),
                -float(item["magnitude"]),
                json.dumps(item["routes"], sort_keys=True),
            ),
        )
        if eligible
        else reference
    )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-DETECTION-CYCLE4-SELECTION-v1",
        "status": "FROZEN_BEFORE_CYCLE4_SEED1_SEED2_OUTPUTS",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "seed0_unit_hashes": hashes,
        "eligible_route_tuples": len(eligible),
        "reference_cycle3": reference,
        "selection": selected,
        "all_detector_scores": scores,
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    if SELECTION.exists():
        existing = base.load_json(SELECTION)
        if existing != payload:
            raise base.C016Error("Frozen cycle-4 selection differs from recomputation")
        return existing
    base.write_json_exclusive(SELECTION, payload)
    return payload


def verify_selection() -> dict[str, Any]:
    protocol = verify_protocol()
    if not SELECTION.is_file():
        raise base.C016Error(f"Missing cycle-4 selection: {SELECTION}")
    payload = base.load_json(SELECTION)
    if payload.get("schema_version") != "MERIT-C016-V24-DETECTION-CYCLE4-SELECTION-v1":
        raise base.C016Error("Unexpected cycle-4 selection schema")
    if payload.get("status") != "FROZEN_BEFORE_CYCLE4_SEED1_SEED2_OUTPUTS":
        raise base.C016Error("Cycle-4 selection is not frozen")
    if base.payload_sha256(payload) != payload.get("payload_sha256"):
        raise base.C016Error("Cycle-4 selection payload hash mismatch")
    if payload["protocol_payload_sha256"] != protocol["payload_sha256"]:
        raise base.C016Error("Cycle-4 selection protocol mismatch")
    for name, expected in payload["seed0_unit_hashes"].items():
        path = RUN_ROOT / "progress" / name
        if base.sha256_file(path) != expected:
            raise base.C016Error(f"Cycle-4 seed-0 unit changed: {path}")
    return payload


def _load_rows(seeds: Iterable[int]) -> list[dict[str, Any]]:
    selected_seeds = set(seeds)
    rows: list[dict[str, Any]] = []
    units = set()
    for path in sorted((RUN_ROOT / "progress").glob("fold_*_seed*.json")):
        payload = base.load_json(path)
        if base.payload_sha256(payload) != payload.get("payload_sha256"):
            raise base.C016Error(f"Cycle-4 unit hash mismatch: {path}")
        seed = int(payload["seed"])
        if seed in selected_seeds:
            units.add((payload["fold"], seed))
            rows.extend({**row, "seed": seed} for row in payload["parent_rows"])
    if len(units) != len(selected_seeds) * len(base.TUNNELS):
        raise base.C016Error(f"Cycle-4 rows incomplete for seeds {sorted(selected_seeds)}")
    return rows


def _aggregate(
    rows: list[dict[str, Any]], routes: dict[str, str], seed_offset: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    selected = [row for row in rows if row["condition"] == routes[row["detector"]]]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in selected:
        grouped[(row["tunnel"], row["detector"], row["parent"])].append(row)
    detector_parent_rows = [
        {
            "tunnel": tunnel,
            "detector": detector,
            "parent": parent,
            **{
                key: base.mean(row[key] for row in group)
                for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
            },
        }
        for (tunnel, detector, parent), group in sorted(grouped.items())
    ]
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in detector_parent_rows:
        by_parent[(row["tunnel"], row["parent"])].append(row)
    panel_rows = [
        {
            "tunnel": tunnel,
            "parent": parent,
            **{
                key: base.mean(row[key] for row in group)
                for key in ("delta_auprc", "delta_f1", "delta_matched_fpr")
            },
        }
        for (tunnel, parent), group in sorted(by_parent.items())
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
        detector_rows = [row for row in detector_parent_rows if row["detector"] == detector]
        detectors[detector] = {
            "delta_auprc": base.stratified_bootstrap(
                detector_rows,
                "delta_auprc",
                base.BOOTSTRAP_SEED + seed_offset + 10 + index,
            ),
            "delta_f1": base.stratified_bootstrap(
                detector_rows, "delta_f1", base.BOOTSTRAP_SEED + seed_offset + 20 + index
            ),
            "delta_matched_fpr": base.stratified_bootstrap(
                detector_rows,
                "delta_matched_fpr",
                base.BOOTSTRAP_SEED + seed_offset + 30 + index,
            ),
        }
    return panel, detectors, panel_rows


def _panel_gate(panel: dict[str, Any], detectors: dict[str, Any]) -> bool:
    return bool(
        panel["delta_auprc"]["ci95_low"] > 0.0
        and panel["delta_auprc"]["positive_tunnels"] == 4
        and panel["delta_f1"]["mean"] > 0.0
        and panel["delta_matched_fpr"]["mean"] <= 0.0
        and all(value["delta_auprc"]["mean"] >= 0.0 for value in detectors.values())
    )


def _difference(
    selected_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], seed_offset: int
) -> dict[str, Any]:
    reference = {(row["tunnel"], row["parent"]): row for row in reference_rows}
    rows = [
        {
            "tunnel": row["tunnel"],
            "parent": row["parent"],
            "delta_auprc": row["delta_auprc"]
            - reference[(row["tunnel"], row["parent"])]["delta_auprc"],
        }
        for row in selected_rows
    ]
    return base.stratified_bootstrap(
        rows, "delta_auprc", base.BOOTSTRAP_SEED + seed_offset
    )


def summarize() -> dict[str, Any]:
    protocol = verify_protocol()
    selection = verify_selection()
    routes = selection["selection"]["routes"]
    reference_routes = {detector: "full" for detector in base.field_common.DETECTORS}
    analyses = {}
    for label, seeds, offset in (
        ("all_seeds", base.SEEDS, 500),
        ("held_replication_seeds_1_2", (1, 2), 600),
    ):
        rows = _load_rows(seeds)
        panel, detectors, panel_rows = _aggregate(rows, routes, offset)
        reference_panel, reference_detectors, reference_rows = _aggregate(
            rows, reference_routes, offset + 100
        )
        difference = _difference(panel_rows, reference_rows, offset + 200)
        superiority = bool(
            difference["mean"] > 0.0
            and difference["ci95_low"] > 0.0
            and difference["positive_tunnels"] >= 3
        )
        analyses[label] = {
            "panel": panel,
            "detectors": detectors,
            "cycle3_reference_panel": reference_panel,
            "cycle3_reference_detectors": reference_detectors,
            "selected_minus_cycle3_delta_auprc": difference,
            "panel_gate_pass": _panel_gate(panel, detectors),
            "superiority_gate_pass": superiority,
        }
    frozen_v23 = float(
        base.load_json(base.C014_SUMMARY)["detection"]
        ["existing_four_tunnel_descriptive_delta_auprc"]["mean"]
    )
    efficacy = bool(
        analyses["all_seeds"]["panel_gate_pass"]
        and analyses["all_seeds"]["superiority_gate_pass"]
        and analyses["held_replication_seeds_1_2"]["panel_gate_pass"]
        and analyses["held_replication_seeds_1_2"]["superiority_gate_pass"]
        and analyses["all_seeds"]["panel"]["delta_auprc"]["mean"] > frozen_v23
    )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C016-V24-CYCLE4-SUMMARY-v1",
        "status": "PASS_COMPLETE" if efficacy else "COMPLETE_GATE_FAIL",
        "scientific_status": "SEED0_SELECTION_WITH_SEED1_SEED2_HELD_REPLICATION",
        "routes": routes,
        **analyses,
        "frozen_v23_four_tunnel_mean_delta_auprc": frozen_v23,
        "efficacy_gate_pass": efficacy,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "selection_payload_sha256": selection["payload_sha256"],
        "claim_boundary": (
            "Cycle 4 is iterative development on the supplied four tunnels. Probability fusion "
            "uses raw image evidence only and does not establish new-site generalisation."
        ),
    }
    payload["payload_sha256"] = base.payload_sha256(payload)
    base.write_json_atomic(RUN_ROOT / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze", "screen", "select", "replicate", "summarize", "run"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_protocol()
    elif args.command == "screen":
        result = evaluate_seed(0, args.batch_size)
    elif args.command == "select":
        result = select()
    elif args.command == "replicate":
        verify_selection()
        evaluate_seed(1, args.batch_size)
        result = evaluate_seed(2, args.batch_size)
    elif args.command == "summarize":
        result = summarize()
    else:
        freeze_protocol()
        evaluate_seed(0, args.batch_size)
        select()
        verify_selection()
        evaluate_seed(1, args.batch_size)
        evaluate_seed(2, args.batch_size)
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
