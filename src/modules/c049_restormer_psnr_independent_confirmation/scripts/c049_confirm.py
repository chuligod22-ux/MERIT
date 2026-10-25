"""Run and summarize the frozen C049 Restormer independent confirmation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch

import c049_common as common


METRICS = (
    "delta_psnr_db",
    "delta_lpips_alex",
    "delta_gradient_mae",
    "delta_overshoot_p95",
)


def evaluate(seed: int) -> dict[str, Any]:
    protocol = common.load_protocol()
    if seed not in protocol["model_seeds"]:
        raise common.C049Error("Unfrozen C049 model seed")
    evaluation, controlled, metrics, _ = (
        common.c048_common.c041_common.import_dependencies()
    )
    sample_protocol = {
        "banks": {"c040": protocol["bank"]},
        "conditions": protocol["conditions"],
        "degradation_seeds": protocol["degradation_seeds"],
    }
    samples = common.c048_common.c041_development.load_samples(
        "c040", sample_protocol, controlled
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    base, candidate, provenance, identity_error, device = (
        common.c048_evaluate.infer(seed, samples, protocol)
    )
    clean = np.stack([item["clean"] for item in samples]).astype(np.float32)
    delta_psnr = metrics.psnr_batch(clean, candidate) - metrics.psnr_batch(clean, base)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    delta_lpips = (
        metrics.lpips_batch(perceptual, clean, candidate, device)
        - metrics.lpips_batch(perceptual, clean, base, device)
    )
    delta_gradient, delta_overshoot = metrics.structural_metric_deltas(
        clean, base, candidate, controlled
    )
    records = []
    for index, sample in enumerate(samples):
        records.append(
            {
                "bank": "c040",
                "model_seed": int(seed),
                "sample_id": sample["sample_id"],
                "condition": sample["condition"],
                "stochastic_seed": sample["stochastic_seed"],
                "parent_sha256": sample["parent_sha256"],
                "delta_psnr_db": float(delta_psnr[index]),
                "delta_lpips_alex": float(delta_lpips[index]),
                "delta_gradient_mae": float(delta_gradient[index]),
                "delta_overshoot_p95": float(delta_overshoot[index]),
            }
        )
    common.OUTPUT.mkdir(parents=True, exist_ok=True)
    records_path = common.OUTPUT / f"c040_seed{seed}_records.jsonl"
    common.write_jsonl(records_path, records)
    result = {
        "schema_version": "MERIT-C049-CONFIRMATION-CELL-v1",
        "status": "COMPLETE",
        "model_seed": int(seed),
        "records": len(records),
        "parents": len({row["parent_sha256"] for row in records}),
        "identity_max_abs_error": identity_error,
        "records_path": str(records_path.resolve()),
        "records_sha256": common.sha256_file(records_path),
        "protocol_payload_sha256": protocol["payload_sha256"],
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    common.write_json(common.OUTPUT / f"c040_seed{seed}.json", result)
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
    protocol = common.load_protocol()
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for seed in protocol["model_seeds"]:
        cell_path = common.OUTPUT / f"c040_seed{seed}.json"
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        records_path = Path(cell["records_path"])
        if (
            cell["protocol_payload_sha256"] != protocol["payload_sha256"]
            or common.sha256_file(records_path) != cell["records_sha256"]
        ):
            raise common.C049Error(f"Cell mismatch: {cell_path}")
        rows.extend(
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        cells.append(cell)

    results: dict[str, Any] = {}
    failures: list[str] = []
    for condition_index, condition in enumerate(
        item["id"] for item in protocol["conditions"]
    ):
        subset = [row for row in rows if row["condition"] == condition]
        parents = sorted({row["parent_sha256"] for row in subset})
        summaries = {}
        for metric_index, metric in enumerate(METRICS):
            parent_values = np.asarray(
                [
                    np.mean(
                        [row[metric] for row in subset if row["parent_sha256"] == parent]
                    )
                    for parent in parents
                ]
            )
            summaries[metric] = bootstrap(
                parent_values,
                int(protocol["bootstrap_seed"]) + condition_index * 10 + metric_index,
                int(protocol["bootstrap_replicates"]),
            )
        seed_means = {
            str(seed): float(
                np.mean(
                    [
                        row["delta_psnr_db"]
                        for row in subset
                        if row["model_seed"] == seed
                    ]
                )
            )
            for seed in protocol["model_seeds"]
        }
        passed = bool(
            min(seed_means.values()) > 0.0
            and summaries["delta_psnr_db"]["ci95_lower"] > 0.0
        )
        if not passed:
            failures.append(condition)
        results[condition] = {
            "status": "PASS" if passed else "FAIL",
            "parents": len(parents),
            "seed_delta_psnr_means": seed_means,
            "metrics": summaries,
        }
    identity_error = max(cell["identity_max_abs_error"] for cell in cells)
    passed = not failures and identity_error == 0.0
    result = {
        "schema_version": "MERIT-C049-INDEPENDENT-CONFIRMATION-SUMMARY-v1",
        "status": "PASS" if passed else "FAIL",
        "claim_authorized": passed,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "bank_payload_sha256": protocol["bank_payload_sha256"],
        "routes": protocol["routes"],
        "conditions": results,
        "records": len(rows),
        "parents": len({row["parent_sha256"] for row in rows}),
        "identity_max_abs_error": identity_error,
        "psnr_gate_failures": failures,
        "diagnostic_metrics_are_not_acceptance_gates": [
            "delta_lpips_alex",
            "delta_gradient_mae",
            "delta_overshoot_p95",
        ],
    }
    common.write_json(common.OUTPUT / "confirmation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "summarize"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.action == "evaluate" and args.seed is not None:
        output = evaluate(args.seed)
    elif args.action == "summarize":
        output = summarize()
    else:
        raise common.C049Error("evaluate requires model seed")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
