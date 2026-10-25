"""Run the frozen C038 routes once on the unopened C036 ISO200 bank."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import c039_common as common


sys.path.insert(0, str(common.C032 / "scripts"))
sys.path.insert(0, str(common.C035 / "scripts"))
import c030_confirmation as prior_confirmation  # noqa: E402
from c035_evaluate import load_model  # noqa: E402


def fixed_route(
    base: np.ndarray, components: dict[str, np.ndarray], route: dict[str, float]
) -> np.ndarray:
    value = base + np.float32(route["frequency_alpha"]) * (
        components["high_frequency"] - base
    )
    value = value + np.float32(route["gate_alpha"]) * (components["gate"] - base)
    if "psnr_alpha" in route:
        value = value + np.float32(route["psnr_alpha"]) * (components["psnr"] - base)
    value = value + np.float32(route["perceptual_alpha"]) * (
        components["perceptual"] - base
    )
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def compose(
    components: dict[str, np.ndarray], samples: list[dict[str, Any]], freeze: dict[str, Any]
) -> tuple[np.ndarray, list[float]]:
    base = components["base"]
    candidate = base.copy()
    conditions = np.asarray([sample["condition"] for sample in samples])
    low = conditions == "id_low"
    mid = conditions == "id_mid"
    high = conditions == "id_high"

    low_route = freeze["routes"]["id_low"]
    full_residual = gaussian_filter(
        components["full"] - base,
        sigma=(0.0, float(freeze["low_full_sigma_px"]), float(freeze["low_full_sigma_px"])),
        mode="reflect",
    )
    candidate[low] = (
        base[low]
        + np.float32(low_route["shared_alpha"]) * (components["shared"][low] - base[low])
        + np.float32(low_route["full_beta"]) * full_residual[low]
    )

    mid_route = freeze["routes"]["id_mid"]
    candidate[mid] = (
        base[mid]
        + np.float32(mid_route["frequency_alpha"])
        * (components["mid_frequency"][mid] - base[mid])
        + np.float32(mid_route["shared_alpha"]) * (components["shared"][mid] - base[mid])
    )

    high_route = freeze["routes"]["id_high"]
    route_a = fixed_route(base, components, high_route["route_a"])
    route_b = fixed_route(base, components, high_route["route_b"])
    device = torch.device("cuda")
    gate = load_model(high_route["variant"], device)
    indices = np.flatnonzero(high)
    outputs = []
    weights = []
    for start in range(0, len(indices), 8):
        selected = indices[start : start + 8]
        batch_base = torch.from_numpy(base[selected, None]).to(device)
        batch_a = torch.from_numpy(route_a[selected, None]).to(device)
        batch_b = torch.from_numpy(route_b[selected, None]).to(device)
        with torch.inference_mode():
            residual, batch_weights = gate(batch_base, batch_a, batch_b)
            output = torch.clamp(
                batch_base + float(high_route["gate_alpha"]) * residual, 0.0, 1.0
            )
        outputs.append(output[:, 0].cpu().numpy().astype(np.float32))
        weights.append(batch_weights.mean(dim=(0, 2, 3)).cpu().numpy())
    candidate[high] = np.concatenate(outputs)

    candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)
    prompts = np.asarray(
        [(sample["p_b"] != 0.0) or (sample["p_n"] != 0.0) for sample in samples]
    )[:, None, None]
    candidate = np.where(prompts, candidate, base).astype(np.float32)
    return candidate, np.stack(weights).mean(axis=0).astype(float).tolist()


def evaluate(seed: int) -> dict[str, Any]:
    freeze = common.load_freeze()
    if seed not in freeze["confirmation"]["model_seeds"]:
        raise common.C039Error(f"Unfrozen model seed: {seed}")
    evaluation, controlled, metrics, _ = common.c022_common.import_dependencies()
    samples = prior_confirmation.load_samples(freeze, controlled)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    components, provenance, device = prior_confirmation.infer_components(
        seed, samples, freeze, evaluation, controlled
    )
    candidate, gate_weights = compose(components, samples, freeze)
    identity_error = float(
        np.max(
            np.abs(
                np.where(
                    np.zeros((1, 1, 1), dtype=bool),
                    candidate[:1],
                    components["base"][:1],
                )
                - components["base"][:1]
            )
        )
    )
    clean = np.stack([sample["clean"] for sample in samples]).astype(np.float32)
    base = components["base"]
    delta_psnr = metrics.psnr_batch(clean, candidate) - metrics.psnr_batch(clean, base)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    delta_lpips = metrics.lpips_batch(perceptual, clean, candidate, device) - metrics.lpips_batch(
        perceptual, clean, base, device
    )
    delta_gradient, delta_overshoot = metrics.structural_metric_deltas(
        clean, base, candidate, controlled
    )
    records = [
        {
            "model_seed": seed,
            "sample_id": sample["sample_id"].replace("c030__", "c036__"),
            "condition": sample["condition"],
            "stochastic_seed": sample["stochastic_seed"],
            "parent_sha256": sample["parent_sha256"],
            "delta_psnr_db": float(delta_psnr[index]),
            "delta_lpips_alex": float(delta_lpips[index]),
            "delta_gradient_mae": float(delta_gradient[index]),
            "delta_overshoot_p95": float(delta_overshoot[index]),
        }
        for index, sample in enumerate(samples)
    ]
    records_path = common.OUTPUT_ROOT / f"seed{seed}_records.jsonl"
    common.c022_common.write_jsonl(records_path, records)
    result = {
        "schema_version": "MERIT-C039-C036-CONFIRMATION-CELL-v1",
        "status": "COMPLETE",
        "model_seed": seed,
        "records": len(records),
        "parents": len({row["parent_sha256"] for row in records}),
        "identity_max_abs_error": identity_error,
        "high_gate_weights": gate_weights,
        "records_path": str(records_path.resolve()),
        "records_sha256": common.sha256_file(records_path),
        "confirmation_freeze_payload_sha256": freeze["payload_sha256"],
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    common.c023.write_json(common.OUTPUT_ROOT / f"seed{seed}.json", result)
    return result


def bootstrap(values: np.ndarray, seed: int, replicates: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def summarize() -> dict[str, Any]:
    freeze = common.load_freeze()
    records = []
    cells = []
    for seed in freeze["confirmation"]["model_seeds"]:
        cell = json.loads((common.OUTPUT_ROOT / f"seed{seed}.json").read_text(encoding="utf-8"))
        path = Path(cell["records_path"])
        if (
            cell["confirmation_freeze_payload_sha256"] != freeze["payload_sha256"]
            or common.sha256_file(path) != cell["records_sha256"]
        ):
            raise common.C039Error("C036 confirmation cell mismatch")
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        cells.append(cell)
    metric_names = (
        "delta_psnr_db",
        "delta_lpips_alex",
        "delta_gradient_mae",
        "delta_overshoot_p95",
    )
    results = {}
    for condition_index, condition in enumerate(item["id"] for item in freeze["conditions"]):
        subset = [row for row in records if row["condition"] == condition]
        parents = sorted({row["parent_sha256"] for row in subset})
        summaries = {}
        for metric_index, metric in enumerate(metric_names):
            parent_values = np.asarray(
                [
                    np.mean([float(row[metric]) for row in subset if row["parent_sha256"] == parent])
                    for parent in parents
                ]
            )
            summaries[metric] = bootstrap(
                parent_values,
                int(freeze["confirmation"]["bootstrap_seed"])
                + condition_index * 10
                + metric_index,
                int(freeze["confirmation"]["bootstrap_replicates"]),
            )
        seed_psnr = {
            str(seed): float(
                np.mean([row["delta_psnr_db"] for row in subset if row["model_seed"] == seed])
            )
            for seed in freeze["confirmation"]["model_seeds"]
        }
        passed = bool(
            len(parents) == int(freeze["confirmation"]["parents"])
            and min(seed_psnr.values()) > 0.0
            and summaries["delta_psnr_db"]["ci95_lower"] > 0.0
            and summaries["delta_lpips_alex"]["ci95_upper"] < 0.0
            and summaries["delta_gradient_mae"]["ci95_upper"] < 0.0
            and summaries["delta_overshoot_p95"]["ci95_upper"] <= 0.00005
        )
        results[condition] = {
            "status": "PASS" if passed else "FAIL",
            "parents": len(parents),
            "seed_delta_psnr_means": seed_psnr,
            "metrics": summaries,
        }
    all_pass = all(value["status"] == "PASS" for value in results.values())
    summary = {
        "schema_version": "MERIT-C039-C036-CONFIRMATION-SUMMARY-v1",
        "status": "PASS" if all_pass else "FAIL",
        "all_conditions_pass": all_pass,
        "confirmation_freeze_payload_sha256": freeze["payload_sha256"],
        "conditions": results,
        "identity_max_abs_error": max(cell["identity_max_abs_error"] for cell in cells),
        "records": len(records),
        "parents": len({row["parent_sha256"] for row in records}),
        "high_gate_weights_by_seed": {
            str(cell["model_seed"]): cell["high_gate_weights"] for cell in cells
        },
    }
    common.c023.write_json(common.OUTPUT_ROOT / "confirmation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "summarize"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.action == "evaluate" and args.seed is not None:
        result = evaluate(args.seed)
    elif args.action == "summarize":
        result = summarize()
    else:
        raise common.C039Error("evaluate requires --seed")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
