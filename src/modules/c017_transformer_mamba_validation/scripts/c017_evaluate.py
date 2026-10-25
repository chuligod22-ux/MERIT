#!/usr/bin/env python
"""Frozen raw-versus-MERIT evaluation for the C017 detector extension."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
MODIFIED = C017.parent
C016_SCRIPTS = MODIFIED / "c016_merit_v24_development" / "scripts"
for directory in (HERE, C016_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import c016_cycle5 as cycle5
import c016_pipeline as base
import c017_models as models
import c017_train as training

PROTOCOL = C017 / "protocol" / "c017_protocol_v1.json"
SELECTION = C017 / "protocol" / "c017_selection_v1.json"
TRAINING_ROOT = C017 / "runs" / "training_v1"
RUN_ROOT = C017 / "runs" / "evaluation_v1"
PROFILES = ("sharp0025", "inv0125", "inv1250")
WEIGHTS = (0.25, 0.50, 0.75, 1.00)


class C017EvaluationError(RuntimeError):
    """Raised when an evaluation invariant is violated."""


def condition_specifications() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "identity": {"profile": None, "weight": 0.0},
    }
    for profile in PROFILES:
        for weight in WEIGHTS:
            label = f"{profile}__linear{round(weight * 100):03d}"
            result[label] = {"profile": profile, "weight": weight}
    return result


CONDITIONS = condition_specifications()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_protocol() -> dict[str, Any]:
    protocol = training.validate_protocol(PROTOCOL)
    if protocol["evaluation"]["candidate_conditions"] != CONDITIONS:
        raise C017EvaluationError("Evaluation conditions changed after freeze")
    if training.sha256_file(Path(__file__)) != protocol["code"]["evaluation"]["sha256"]:
        raise C017EvaluationError("Evaluation code changed after protocol freeze")
    cycle5.verify_protocol()
    cycle5.verify_selection()
    return protocol


def checkpoint_path(model_name: str, fold: str, restorer_seed: int) -> Path:
    detector_seed = base.DETECTOR_SEEDS[restorer_seed]
    return (
        TRAINING_ROOT
        / model_name
        / f"fold_{fold}"
        / f"seed_{detector_seed}"
        / "final_epoch.pt"
    )


def load_detector(
    protocol: Mapping[str, Any], model_name: str, fold: str, seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    run_dir = checkpoint_path(model_name, fold, seed).parent
    final_path = run_dir / "run_final.json"
    if not final_path.is_file():
        raise C017EvaluationError(f"Incomplete detector training: {run_dir}")
    run = training.load_json(final_path)
    checkpoint = run_dir / str(run["checkpoint"])
    if training.sha256_file(checkpoint) != run["checkpoint_sha256"]:
        raise C017EvaluationError(f"Detector checkpoint hash mismatch: {checkpoint}")
    if run["protocol_payload_sha256"] != protocol["payload_sha256"]:
        raise C017EvaluationError("Detector checkpoint protocol mismatch")
    detector = models.build_model(model_name, Path(protocol["assets"]["segformer_pretrained_dir"]))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    detector.load_state_dict(state["model_state"], strict=True)
    detector = detector.to(device).eval()
    return detector, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "detector_seed": base.DETECTOR_SEEDS[seed],
    }


def fuse(raw: np.ndarray, profile: np.ndarray, weight: float) -> np.ndarray:
    return np.clip(raw + np.float32(weight) * (profile - raw), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def _unit(fold: str, seed: int, batch_size: int) -> dict[str, Any]:
    protocol = verify_protocol()
    c009_common, d2_eval, _, detector_protocol, d2, cv, external, tail = (
        base._detection_modules()
    )
    _, external_protocol = tail
    cohort = base.field_common.read_csv(base.D2_COHORT)
    fold_rows = [row for row in cohort if row["tunnel"] == fold]
    if not fold_rows:
        raise C017EvaluationError(f"Empty D2 fold: {fold}")
    if not torch.cuda.is_available():
        raise C017EvaluationError("CUDA is required for C017 evaluation")
    device = torch.device("cuda")
    restorer, restorer_provenance = base.c010.load_model(seed, device)
    detectors: dict[str, torch.nn.Module] = {}
    detector_provenance: dict[str, Any] = {}
    for model_name in models.MODEL_NAMES:
        detectors[model_name], detector_provenance[model_name] = load_detector(
            protocol, model_name, fold, seed, device
        )
    dataset = external.ExternalSyntheticDataset(fold_rows, external_protocol)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    bins = int(detector_protocol["detectors"]["histogram_bins"])
    threshold = float(detector_protocol["detectors"]["fixed_threshold"])
    accumulators = {
        model_name: collections.defaultdict(
            lambda: {
                condition: external.Accumulator(bins) for condition in CONDITIONS
            }
        )
        for model_name in models.MODEL_NAMES
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
        inputs = torch.from_numpy(c009_common.prompt_tensor(normalized, zeros, zeros)).to(device)
        with base.c010.pilot.autocast(device):
            transformed = base.profiles.make_candidates(inputs, restorer.base, PROFILES)
        encoded = base.encode_candidates(transformed)
        candidates = {"raw": raw, **encoded}
        for model_name, detector in detectors.items():
            probabilities = base.detect_candidate_map(
                candidates, detector, device, external, conditions_per_call=2
            )
            fused = {
                condition: (
                    probabilities["raw"]
                    if specification["profile"] is None
                    else fuse(
                        probabilities["raw"],
                        probabilities[str(specification["profile"])],
                        float(specification["weight"]),
                    )
                )
                for condition, specification in CONDITIONS.items()
            }
            for sample_index, parent in enumerate(parents):
                for condition, probability in fused.items():
                    profile = CONDITIONS[condition]["profile"]
                    candidate = raw if profile is None else encoded[str(profile)]
                    external.update_accumulator(
                        accumulators[model_name][str(parent)][condition],
                        probability[sample_index : sample_index + 1],
                        masks[sample_index : sample_index + 1],
                        raw[sample_index : sample_index + 1],
                        candidate[sample_index : sample_index + 1],
                        cv,
                        threshold,
                        False,
                    )
        if batch_index == 1 or batch_index == len(loader) or batch_index % 20 == 0:
            print(
                json.dumps(
                    {
                        "stage": "c017_evaluation",
                        "fold": fold,
                        "seed": seed,
                        "batch": batch_index,
                        "batches": len(loader),
                    }
                ),
                flush=True,
            )
    parent_rows: list[dict[str, Any]] = []
    for model_name, by_parent in accumulators.items():
        for parent, by_condition in by_parent.items():
            metrics = {
                condition: external.finalize_accumulator(accumulator, d2, cv)
                for condition, accumulator in by_condition.items()
            }
            raw_metric = metrics["identity"]
            for condition in CONDITIONS:
                burden = d2_eval.matched_sensitivity_burden(
                    by_condition[condition], raw_metric
                )
                parent_rows.append(
                    {
                        "tunnel": fold,
                        "detector": model_name,
                        "parent": parent,
                        "condition": condition,
                        "delta_auprc": float(
                            metrics[condition]["pixel_auprc"]
                            - raw_metric["pixel_auprc"]
                        ),
                        "delta_f1": float(
                            metrics[condition]["f1_at_0_5"]
                            - raw_metric["f1_at_0_5"]
                        ),
                        "delta_matched_fpr": float(
                            burden[
                                "delta_false_positive_pixel_rate_restored_minus_degraded"
                            ]
                        ),
                    }
                )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C017-EVALUATION-UNIT-v1",
        "status": "PASS",
        "fold": fold,
        "seed": seed,
        "tile_count": len(dataset),
        "parents": len({row["parent"] for row in parent_rows}),
        "parent_rows": parent_rows,
        "restorer_provenance": restorer_provenance,
        "detector_provenance": detector_provenance,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    del restorer, detectors, dataset, loader, accumulators
    torch.cuda.empty_cache()
    return payload


def evaluate_seed(seed: int, batch_size: int) -> dict[str, Any]:
    verify_protocol()
    if seed in (1, 2):
        verify_selection()
    progress = RUN_ROOT / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for fold in base.TUNNELS:
        path = progress / f"fold_{fold}_seed{seed}.json"
        if path.is_file():
            value = training.load_json(path)
            if (
                value.get("schema_version") == "MERIT-C017-EVALUATION-UNIT-v1"
                and payload_sha256(value) == value.get("payload_sha256")
            ):
                continue
        base.write_json_atomic(path, _unit(fold, seed, batch_size))
    result: dict[str, Any] = {
        "schema_version": "MERIT-C017-EVALUATION-SEED-v1",
        "status": "PASS_COMPLETE",
        "seed": seed,
        "units": len(list(progress.glob(f"fold_*_seed{seed}.json"))),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    result["payload_sha256"] = payload_sha256(result)
    base.write_json_atomic(RUN_ROOT / f"SEED_{seed}_COMPLETE.json", result)
    return result


def _load_rows(seeds: Iterable[int]) -> list[dict[str, Any]]:
    requested = set(seeds)
    rows: list[dict[str, Any]] = []
    units = set()
    for path in sorted((RUN_ROOT / "progress").glob("fold_*_seed*.json")):
        payload = training.load_json(path)
        if payload_sha256(payload) != payload.get("payload_sha256"):
            raise C017EvaluationError(f"Evaluation unit hash mismatch: {path}")
        seed = int(payload["seed"])
        if seed in requested:
            units.add((str(payload["fold"]), seed))
            rows.extend({**row, "seed": seed} for row in payload["parent_rows"])
    if len(units) != len(requested) * len(base.TUNNELS):
        raise C017EvaluationError(f"Incomplete rows for seeds {sorted(requested)}")
    return rows


def _score(rows: list[dict[str, Any]], detector: str, condition: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["detector"] == detector and row["condition"] == condition
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
        "positive_tunnels": sum(value["delta_auprc"] > 0 for value in per_tunnel.values()),
        "per_tunnel": per_tunnel,
    }


def _route_tuple(
    scores: Mapping[str, Mapping[str, dict[str, Any]]], route_tuple: tuple[str, ...]
) -> dict[str, Any]:
    selected = {
        detector: scores[detector][condition]
        for detector, condition in zip(models.MODEL_NAMES, route_tuple)
    }
    per_tunnel = {
        tunnel: base.mean(
            selected[detector]["per_tunnel"][tunnel]["delta_auprc"]
            for detector in models.MODEL_NAMES
        )
        for tunnel in base.TUNNELS
    }
    return {
        "routes": dict(zip(models.MODEL_NAMES, route_tuple)),
        "detectors": selected,
        "panel_delta_auprc": base.mean(
            value["delta_auprc"] for value in selected.values()
        ),
        "panel_delta_matched_fpr": base.mean(
            value["delta_matched_fpr"] for value in selected.values()
        ),
        "panel_per_tunnel_delta_auprc": per_tunnel,
        "magnitude": sum(float(CONDITIONS[name]["weight"]) for name in route_tuple),
    }


def select() -> dict[str, Any]:
    protocol = verify_protocol()
    files = sorted((RUN_ROOT / "progress").glob("fold_*_seed0.json"))
    if len(files) != len(base.TUNNELS):
        raise C017EvaluationError("Seed-0 screen is incomplete")
    rows = _load_rows((0,))
    scores = {
        detector: {
            condition: _score(rows, detector, condition) for condition in CONDITIONS
        }
        for detector in models.MODEL_NAMES
    }
    eligible = []
    for route_tuple in itertools.product(CONDITIONS, repeat=len(models.MODEL_NAMES)):
        candidate = _route_tuple(scores, route_tuple)
        if (
            candidate["panel_delta_matched_fpr"] <= 0.0
            and all(value > 0.0 for value in candidate["panel_per_tunnel_delta_auprc"].values())
            and all(
                score["delta_auprc"] >= 0.0
                and (condition == "identity" or score["positive_tunnels"] >= 2)
                for condition, score in zip(route_tuple, candidate["detectors"].values())
            )
        ):
            eligible.append(candidate)
    if not eligible:
        raise C017EvaluationError("Seed-0 screen found no eligible route tuple")
    selected = max(
        eligible,
        key=lambda item: (
            float(item["panel_delta_auprc"]),
            -float(item["magnitude"]),
            json.dumps(item["routes"], sort_keys=True),
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C017-SELECTION-v1",
        "status": "FROZEN_BEFORE_SEED1_SEED2_OUTPUTS",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "seed0_unit_hashes": {
            path.name: training.sha256_file(path) for path in files
        },
        "eligible_route_tuples": len(eligible),
        "selection": selected,
        "all_detector_scores": scores,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    if SELECTION.exists():
        existing = training.load_json(SELECTION)
        if existing != payload:
            raise C017EvaluationError("Frozen selection differs from recomputation")
        return existing
    base.write_json_exclusive(SELECTION, payload)
    return payload


def verify_selection() -> dict[str, Any]:
    protocol = verify_protocol()
    if not SELECTION.is_file():
        raise C017EvaluationError("Missing frozen C017 selection")
    payload = training.load_json(SELECTION)
    if (
        payload.get("schema_version") != "MERIT-C017-SELECTION-v1"
        or payload.get("status") != "FROZEN_BEFORE_SEED1_SEED2_OUTPUTS"
        or payload_sha256(payload) != payload.get("payload_sha256")
        or payload.get("protocol_payload_sha256") != protocol["payload_sha256"]
    ):
        raise C017EvaluationError("Invalid frozen C017 selection")
    for name, expected in payload["seed0_unit_hashes"].items():
        if training.sha256_file(RUN_ROOT / "progress" / name) != expected:
            raise C017EvaluationError(f"Seed-0 unit changed: {name}")
    return payload


def _aggregate(
    rows: list[dict[str, Any]],
    routes: Mapping[str, str],
    detector_names: tuple[str, ...],
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    selected = [
        row
        for row in rows
        if row["detector"] in detector_names
        and row["condition"] == routes[row["detector"]]
    ]
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
    expected = len(detector_names) * 14
    if len(detector_parent_rows) != expected:
        raise C017EvaluationError(
            f"Expected {expected} detector-parent rows, found {len(detector_parent_rows)}"
        )
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
        key: base.stratified_bootstrap(
            panel_rows, key, base.BOOTSTRAP_SEED + seed_offset + index
        )
        for index, key in enumerate(("delta_auprc", "delta_f1", "delta_matched_fpr"))
    }
    detectors = {}
    for index, detector in enumerate(detector_names):
        detector_rows = [row for row in detector_parent_rows if row["detector"] == detector]
        detectors[detector] = {
            key: base.stratified_bootstrap(
                detector_rows,
                key,
                base.BOOTSTRAP_SEED + seed_offset + 10 * (metric_index + 1) + index,
            )
            for metric_index, key in enumerate(
                ("delta_auprc", "delta_f1", "delta_matched_fpr")
            )
        }
    return panel, detectors, panel_rows


def _panel_gate(panel: Mapping[str, Any], detectors: Mapping[str, Any]) -> bool:
    return bool(
        panel["delta_auprc"]["ci95_low"] > 0.0
        and panel["delta_auprc"]["positive_tunnels"] == 4
        and panel["delta_f1"]["mean"] > 0.0
        and panel["delta_matched_fpr"]["mean"] <= 0.0
        and all(value["delta_auprc"]["mean"] >= 0.0 for value in detectors.values())
    )


def summarize() -> dict[str, Any]:
    protocol = verify_protocol()
    selection = verify_selection()
    new_routes = selection["selection"]["routes"]
    cnn_selection = cycle5.verify_selection()
    cnn_routes = cnn_selection["selection"]["routes"]
    cnn_detectors = tuple(base.field_common.DETECTORS)
    extended_detectors = (*cnn_detectors, *models.MODEL_NAMES)
    analyses = {}
    for label, seeds, offset in (
        ("all_seeds", base.SEEDS, 1000),
        ("held_replication_seeds_1_2", (1, 2), 2000),
    ):
        new_rows = _load_rows(seeds)
        cnn_rows = cycle5._load_rows(seeds)
        new_panel, new_detector_results, _ = _aggregate(
            new_rows, new_routes, models.MODEL_NAMES, offset
        )
        cnn_panel, cnn_detector_results, _ = _aggregate(
            cnn_rows, cnn_routes, cnn_detectors, offset + 100
        )
        extended_panel, extended_detector_results, _ = _aggregate(
            [*cnn_rows, *new_rows],
            {**cnn_routes, **new_routes},
            extended_detectors,
            offset + 200,
        )
        analyses[label] = {
            "new_model_panel": new_panel,
            "new_model_detectors": new_detector_results,
            "cnn_reference_panel": cnn_panel,
            "cnn_reference_detectors": cnn_detector_results,
            "extended_five_detector_panel": extended_panel,
            "extended_five_detector_results": extended_detector_results,
            "new_model_gate_pass": _panel_gate(new_panel, new_detector_results),
            "extended_five_detector_gate_pass": _panel_gate(
                extended_panel, extended_detector_results
            ),
            "one_percentage_point_target_pass": new_panel["delta_auprc"]["mean"]
            >= float(protocol["evaluation"]["practical_target_delta_auprc"]),
        }
    efficacy = all(
        analyses[label][gate]
        for label in analyses
        for gate in ("new_model_gate_pass", "extended_five_detector_gate_pass")
    )
    practical = all(
        analyses[label]["one_percentage_point_target_pass"] for label in analyses
    )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C017-FINAL-SUMMARY-v1",
        "status": "PASS_COMPLETE" if efficacy else "COMPLETE_GATE_FAIL",
        "scientific_status": "SEED0_SELECTION_WITH_SEED1_SEED2_HELD_REPLICATION",
        "new_model_routes": new_routes,
        "frozen_cnn_routes": cnn_routes,
        **analyses,
        "efficacy_gate_pass": efficacy,
        "one_percentage_point_target_pass": practical,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "selection_payload_sha256": selection["payload_sha256"],
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["payload_sha256"] = payload_sha256(payload)
    base.write_json_atomic(RUN_ROOT / "summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("screen", "select", "replicate", "summarize", "run")
    )
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.command == "screen":
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
        evaluate_seed(0, args.batch_size)
        select()
        evaluate_seed(1, args.batch_size)
        evaluate_seed(2, args.batch_size)
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
