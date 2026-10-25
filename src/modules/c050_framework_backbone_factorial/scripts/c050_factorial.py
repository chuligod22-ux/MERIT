"""Matched C040 factorial comparison of MERIT framework and backbone effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C050 = HERE.parent
REVISION = C050.parent
PROTOCOL = C050 / "protocol" / "factorial_protocol_v1.json"
OUTPUT = C050 / "runs" / "factorial_v1"

for directory in (
    REVISION / "c010_merit_v2_development" / "scripts",
    REVISION / "c039_c036_independent_confirmation" / "scripts",
    REVISION / "c049_restormer_psnr_independent_confirmation" / "scripts",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import c010_confirmatory_eval as naf_eval  # noqa: E402
import c039_common as mpr_common  # noqa: E402
import c039_confirmation as mpr_eval  # noqa: E402
import c049_common as rest_common  # noqa: E402


BACKBONES = ("nafnet", "mprnet", "restormer")
ARMS = ("base", "merit")


class C050Error(RuntimeError):
    """Raised when a C050 frozen invariant is violated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload = dict(protocol)
    recorded = payload.pop("payload_sha256", None)
    expected = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if recorded != expected:
        raise C050Error("C050 protocol payload mismatch")
    if protocol.get("status") != "FROZEN_BEFORE_C050_CROSS_BACKBONE_OUTPUT":
        raise C050Error("C050 protocol is not frozen")
    for item in protocol["source_assets"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise C050Error(f"Frozen source mismatch: {path}")
    return protocol


def load_samples(protocol: dict[str, Any]):
    _, controlled, metrics, _ = rest_common.c048_common.c041_common.import_dependencies()
    sample_protocol = {
        "banks": {"c040": protocol["bank"]},
        "conditions": protocol["conditions"],
        "degradation_seeds": protocol["degradation_seeds"],
    }
    samples = rest_common.c048_common.c041_development.load_samples(
        "c040", sample_protocol, controlled
    )
    expected = (
        int(protocol["bank_crops"])
        * len(protocol["conditions"])
        * len(protocol["degradation_seeds"])
    )
    if len(samples) != expected:
        raise C050Error(f"Unexpected C040 sample coverage: {len(samples)} != {expected}")
    parents = {sample["parent_sha256"] for sample in samples}
    if len(parents) != int(protocol["bank_parents"]):
        raise C050Error("Unexpected C040 parent coverage")
    return samples, controlled, metrics


def infer_nafnet(seed: int, samples: list[dict[str, Any]]):
    _, freeze = naf_eval.verify_freeze()
    device = torch.device("cuda")
    model, provenance = naf_eval.load_model(seed, freeze, device)
    base_outputs: list[np.ndarray] = []
    merit_outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(samples), 16):
            batch = samples[begin : begin + 16]
            degraded = np.stack([item["degraded"] for item in batch]).astype(np.float32)
            prompted = naf_eval.pilot.common.prompt_tensor(
                degraded,
                [item["p_b"] for item in batch],
                [item["p_n"] for item in batch],
            )
            inputs = torch.from_numpy(prompted).to(device)
            with naf_eval.pilot.autocast(device):
                merit_preclip, base_preclip, _, _ = model.forward_components(inputs)
            base_outputs.append(
                np.clip(base_preclip.float().cpu().numpy()[:, 0], 0.0, 1.0).astype(
                    np.float32
                )
            )
            merit_outputs.append(
                np.clip(merit_preclip.float().cpu().numpy()[:, 0], 0.0, 1.0).astype(
                    np.float32
                )
            )
    del model
    torch.cuda.empty_cache()
    return np.concatenate(base_outputs), np.concatenate(merit_outputs), provenance, device


def infer_mprnet(seed: int, samples: list[dict[str, Any]], controlled: Any):
    freeze = mpr_common.load_freeze()
    evaluation, _, _, _ = mpr_common.c022_common.import_dependencies()
    components, provenance, device = mpr_eval.prior_confirmation.infer_components(
        seed, samples, freeze, evaluation, controlled
    )
    merit, gate_weights = mpr_eval.compose(components, samples, freeze)
    provenance = dict(provenance)
    provenance["high_gate_weights"] = gate_weights
    base = components["base"]
    del components
    torch.cuda.empty_cache()
    return base, merit, provenance, device


def infer_restormer(seed: int, samples: list[dict[str, Any]]):
    source_protocol = rest_common.load_protocol()
    return rest_common.c048_evaluate.infer(seed, samples, source_protocol)[:3] + (
        torch.device("cuda"),
    )


def evaluate(backbone: str, seed: int) -> dict[str, Any]:
    protocol = load_protocol()
    if backbone not in BACKBONES or seed not in protocol["model_seeds"]:
        raise C050Error("Unfrozen backbone or model seed")
    if not torch.cuda.is_available():
        raise C050Error("CUDA is required")
    samples, controlled, metrics = load_samples(protocol)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if backbone == "nafnet":
        base, merit, provenance, device = infer_nafnet(seed, samples)
    elif backbone == "mprnet":
        base, merit, provenance, device = infer_mprnet(seed, samples, controlled)
    else:
        base, merit, provenance, device = infer_restormer(seed, samples)

    clean = np.stack([item["clean"] for item in samples]).astype(np.float32)
    source = np.stack([item["degraded"] for item in samples]).astype(np.float32)
    source_psnr = metrics.psnr_batch(clean, source)
    base_psnr = metrics.psnr_batch(clean, base)
    merit_psnr = metrics.psnr_batch(clean, merit)
    records = []
    for index, sample in enumerate(samples):
        records.append(
            {
                "backbone": backbone,
                "model_seed": int(seed),
                "sample_id": sample["sample_id"],
                "condition": sample["condition"],
                "stochastic_seed": int(sample["stochastic_seed"]),
                "parent_sha256": sample["parent_sha256"],
                "source_psnr_db": float(source_psnr[index]),
                "base_psnr_db": float(base_psnr[index]),
                "merit_psnr_db": float(merit_psnr[index]),
                "base_gain_db": float(base_psnr[index] - source_psnr[index]),
                "merit_gain_db": float(merit_psnr[index] - source_psnr[index]),
                "framework_delta_db": float(merit_psnr[index] - base_psnr[index]),
            }
        )
    records_path = OUTPUT / f"{backbone}_seed{seed}_records.jsonl"
    write_jsonl(records_path, records)
    result = {
        "schema_version": "MERIT-C050-FACTORIAL-CELL-v1",
        "status": "COMPLETE",
        "backbone": backbone,
        "model_seed": int(seed),
        "records": len(records),
        "parents": len({row["parent_sha256"] for row in records}),
        "records_path": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "protocol_payload_sha256": protocol["payload_sha256"],
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(OUTPUT / f"{backbone}_seed{seed}.json", result)
    return result


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
    }


def factorial_statistics(cell_means: np.ndarray) -> dict[str, np.ndarray]:
    """Return factorial contrasts for arrays ending in [backbone, arm]."""
    if cell_means.shape[-2:] != (len(BACKBONES), len(ARMS)):
        raise ValueError("Expected [..., 3 backbones, 2 arms]")
    framework = np.mean(cell_means[..., :, 1] - cell_means[..., :, 0], axis=-1)
    backbone_marginal = np.mean(cell_means, axis=-1)
    contrasts = np.stack(
        [
            np.abs(backbone_marginal[..., left] - backbone_marginal[..., right])
            for left, right in combinations(range(len(BACKBONES)), 2)
        ],
        axis=-1,
    )
    max_backbone = np.max(contrasts, axis=-1)
    grand = np.mean(cell_means, axis=(-2, -1))
    arm_marginal = np.mean(cell_means, axis=-2)
    ss_framework = len(BACKBONES) * np.sum(
        (arm_marginal - grand[..., None]) ** 2, axis=-1
    )
    ss_backbone = len(ARMS) * np.sum(
        (backbone_marginal - grand[..., None]) ** 2, axis=-1
    )
    residual = (
        cell_means
        - backbone_marginal[..., :, None]
        - arm_marginal[..., None, :]
        + grand[..., None, None]
    )
    ss_interaction = np.sum(residual**2, axis=(-2, -1))
    total = ss_framework + ss_backbone + ss_interaction
    return {
        "framework_main_effect_db": framework,
        "max_backbone_marginal_contrast_db": max_backbone,
        "framework_minus_backbone_db": framework - max_backbone,
        "framework_to_backbone_ratio": framework / np.maximum(max_backbone, 1e-12),
        "ss_framework": ss_framework,
        "ss_backbone": ss_backbone,
        "ss_interaction": ss_interaction,
        "framework_factor_share": ss_framework / np.maximum(total, 1e-18),
        "framework_to_backbone_ss_ratio": ss_framework
        / np.maximum(ss_backbone, 1e-18),
    }


def parent_tensor(rows: list[dict[str, Any]], protocol: dict[str, Any]):
    parents = sorted({row["parent_sha256"] for row in rows})
    conditions = [item["id"] for item in protocol["conditions"]]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["parent_sha256"], row["backbone"], row["condition"]), []
        ).append(row)
    values = np.full(
        (len(parents), len(BACKBONES), len(conditions), len(ARMS)),
        np.nan,
        dtype=np.float64,
    )
    for parent_index, parent in enumerate(parents):
        for backbone_index, backbone in enumerate(BACKBONES):
            for condition_index, condition in enumerate(conditions):
                subset = grouped.get((parent, backbone, condition), [])
                if not subset:
                    raise C050Error(
                        f"Missing parent cell: {parent[:12]} {backbone} {condition}"
                    )
                values[parent_index, backbone_index, condition_index, 0] = np.mean(
                    [row["base_gain_db"] for row in subset]
                )
                values[parent_index, backbone_index, condition_index, 1] = np.mean(
                    [row["merit_gain_db"] for row in subset]
                )
    if not np.isfinite(values).all():
        raise C050Error("Non-finite parent tensor")
    return parents, conditions, values


def summarize_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    parents, conditions, values = parent_tensor(rows, protocol)
    point_cells = values.mean(axis=(0, 2))
    point = factorial_statistics(point_cells)
    replicates = int(protocol["bootstrap_replicates"])
    rng = np.random.default_rng(int(protocol["bootstrap_seed"]))
    bootstrap: dict[str, list[np.ndarray]] = {}
    condition_draws: list[np.ndarray] = []
    for begin in range(0, replicates, 200):
        count = min(200, replicates - begin)
        indices = rng.integers(0, len(parents), size=(count, len(parents)))
        sampled = values[indices]
        condition_draws.append(
            np.mean(sampled[..., 1] - sampled[..., 0], axis=1)
        )
        cell_draws = np.mean(sampled, axis=(1, 3))
        stats = factorial_statistics(cell_draws)
        for key, value in stats.items():
            bootstrap.setdefault(key, []).append(np.asarray(value))
    draws = {key: np.concatenate(parts) for key, parts in bootstrap.items()}
    delta_draws = np.concatenate(condition_draws, axis=0)

    factorial = {}
    for key, point_value in point.items():
        summary = interval(draws[key])
        summary["estimate"] = float(np.asarray(point_value))
        factorial[key] = summary

    by_cell: dict[str, Any] = {}
    cell_failures: list[str] = []
    for backbone_index, backbone in enumerate(BACKBONES):
        by_cell[backbone] = {}
        for condition_index, condition in enumerate(conditions):
            parent_delta = (
                values[:, backbone_index, condition_index, 1]
                - values[:, backbone_index, condition_index, 0]
            )
            summary = interval(delta_draws[:, backbone_index, condition_index])
            summary["estimate"] = float(parent_delta.mean())
            seed_means = {
                str(seed): float(
                    np.mean(
                        [
                            row["framework_delta_db"]
                            for row in rows
                            if row["backbone"] == backbone
                            and row["condition"] == condition
                            and row["model_seed"] == seed
                        ]
                    )
                )
                for seed in protocol["model_seeds"]
            }
            passed = summary["ci95_lower"] > 0.0 and min(seed_means.values()) > 0.0
            if not passed:
                cell_failures.append(f"{backbone}:{condition}")
            by_cell[backbone][condition] = {
                "status": "PASS" if passed else "FAIL",
                "framework_delta_psnr_db": summary,
                "seed_means": seed_means,
            }

    backbone_marginal = point_cells.mean(axis=1)
    pairwise = {}
    for left, right in combinations(range(len(BACKBONES)), 2):
        key = f"{BACKBONES[left]}_minus_{BACKBONES[right]}"
        pairwise[key] = float(backbone_marginal[left] - backbone_marginal[right])
    gates = {
        "all_backbone_condition_ci_lower_gt_zero": not cell_failures,
        "all_backbone_condition_seed_means_gt_zero": not cell_failures,
        "framework_main_effect_ci_lower_gt_zero": factorial[
            "framework_main_effect_db"
        ]["ci95_lower"]
        > 0.0,
        "framework_minus_max_backbone_ci_lower_gt_zero": factorial[
            "framework_minus_backbone_db"
        ]["ci95_lower"]
        > 0.0,
        "framework_to_backbone_ss_ratio_ci_lower_gt_one": factorial[
            "framework_to_backbone_ss_ratio"
        ]["ci95_lower"]
        > 1.0,
    }
    passed = all(gates.values())
    return {
        "schema_version": "MERIT-C050-FACTORIAL-SUMMARY-v1",
        "status": "PASS" if passed else "FAIL",
        "claim_status": "SUPPORTED_WITH_LIMITS" if passed else "NOT_SUPPORTED",
        "authorized_claim": (
            "Within the three tested restoration backbones and the matched C040 "
            "synthetic conditions, the final MERIT framework configuration had a "
            "larger PSNR-gain effect than the largest backbone marginal contrast."
            if passed
            else "The tested matched factorial does not support a framework effect larger than the backbone effect."
        ),
        "parents": len(parents),
        "records": len(rows),
        "backbones": list(BACKBONES),
        "conditions": conditions,
        "endpoint": "PSNR gain over the identical degraded source, dB",
        "factorial": factorial,
        "backbone_marginal_gain_db": {
            backbone: float(backbone_marginal[index])
            for index, backbone in enumerate(BACKBONES)
        },
        "pairwise_backbone_marginal_contrasts_db": pairwise,
        "by_backbone_condition": by_cell,
        "cell_failures": cell_failures,
        "gates": gates,
        "claim_boundary": protocol["claim_boundary"],
        "analysis_exposure": protocol["analysis_exposure"],
    }


def summarize() -> dict[str, Any]:
    protocol = load_protocol()
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for seed in protocol["model_seeds"]:
            cell_path = OUTPUT / f"{backbone}_seed{seed}.json"
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            records_path = Path(cell["records_path"])
            if (
                cell["protocol_payload_sha256"] != protocol["payload_sha256"]
                or sha256_file(records_path) != cell["records_sha256"]
            ):
                raise C050Error(f"Cell mismatch: {cell_path}")
            rows.extend(
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            cells.append(cell)
    result = summarize_rows(rows, protocol)
    result["protocol_payload_sha256"] = protocol["payload_sha256"]
    result["cell_record_sha256"] = {
        f"{cell['backbone']}_seed{cell['model_seed']}": cell["records_sha256"]
        for cell in cells
    }
    result["elapsed_seconds_total"] = float(
        sum(cell["elapsed_seconds"] for cell in cells)
    )
    result["peak_cuda_memory_bytes"] = int(
        max(cell["peak_cuda_memory_bytes"] for cell in cells)
    )
    write_json(OUTPUT / "factorial_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "summarize"))
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.action == "evaluate":
        if args.backbone is None or args.seed is None:
            raise C050Error("evaluate requires --backbone and --seed")
        result = evaluate(args.backbone, args.seed)
    else:
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
