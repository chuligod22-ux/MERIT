"""Evaluate the C018 v2.5 candidate on the unblinded C009 OOD benchmark."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402


RUN_ROOT = common.C018 / "runs" / "development_v1"
PHYSICS_SCAN = RUN_ROOT / "physics_scan.json"
POISSON_CHECKPOINT = RUN_ROOT / "poisson_nafnet_seed0" / "final.pt"
POISSON_FUSION = RUN_ROOT / "poisson_mild_fusion_scan.json"
BASE_METRICS = (
    common.C009
    / "evaluation"
    / "controlled"
    / "wp1__nafnet__base__seed0"
    / "sample_metrics.jsonl"
)
OUTPUT = RUN_ROOT / "c009_development_evaluation.json"
SAMPLE_OUTPUT = RUN_ROOT / "c009_development_sample_metrics.jsonl"


def bootstrap(values: dict[str, list[float]], seed: int = 20261015) -> dict[str, float]:
    parent_values = np.asarray(
        [float(np.mean(records)) for records in values.values()], dtype=np.float64
    )
    rng = np.random.RandomState(seed)
    draws = rng.choice(
        parent_values, size=(10000, len(parent_values)), replace=True
    ).mean(axis=1)
    return {
        "mean": float(parent_values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "parents": int(len(parent_values)),
        "bootstrap_replicates": 10000,
    }


def load_holdout(evaluation: object) -> tuple[dict[str, Any], np.ndarray, list[dict[str, str]]]:
    contract = evaluation.common.load_contract()
    bank_all = np.load(
        common.C006 / "artifacts" / "d0_bank_448.npy", allow_pickle=False
    )
    with (common.C006 / "artifacts" / "d0_bank_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows_all = list(csv.DictReader(handle))
    selected = [
        (bank_all[int(row["bank_index"])], row)
        for row in rows_all
        if row["split"] == "synthetic_holdout"
    ]
    bank = np.stack([item[0] for item in selected]).astype(np.float32) / 255.0
    rows = [item[1] for item in selected]
    return contract, bank, rows


def load_baseline() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with BASE_METRICS.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value["domain"] == "ood":
                records[value["sample_id"]] = value
    return records


def load_poisson_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    evaluation_common = common.import_file(
        "c009_evaluation_common_c018_eval",
        common.C009 / "scripts" / "c009_evaluation_common.py",
    )
    model, provenance = evaluation_common.build_and_load_model(
        {"kind": "wp1", "family": "nafnet", "arm": "base", "seed": 0},
        device,
    )
    return model, provenance


def infer_indices(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    indices: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    outputs: dict[int, np.ndarray] = {}
    for start in range(0, len(indices), 16):
        batch_indices = indices[start : start + 16]
        source = np.stack([samples[index]["degraded"] for index in batch_indices]).astype(
            np.float32
        )
        zeros = np.zeros_like(source)
        tensor = torch.from_numpy(np.stack([source, zeros, zeros], axis=1)).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            prediction = model.forward_preclip(tensor)
        restored = torch.clamp(prediction.float(), 0.0, 1.0).cpu().numpy()[:, 0]
        for index, image in zip(batch_indices, restored):
            outputs[index] = image.astype(np.float32)
    return outputs


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not PHYSICS_SCAN.is_file() or not POISSON_CHECKPOINT.is_file() or not POISSON_FUSION.is_file():
        raise FileNotFoundError(
            "Physics scan, Poisson checkpoint, and mild-noise fusion scan are required"
        )
    physics = common.load_json(PHYSICS_SCAN)
    fusion = common.load_json(POISSON_FUSION)
    if fusion.get("status") != "PASS":
        raise RuntimeError("Mild Poisson fusion has not passed development gates")
    baseline = load_baseline()
    evaluation = common.import_file(
        "c009_controlled_eval_c018_development",
        common.C009 / "scripts" / "c009_controlled_eval.py",
    )
    contract, bank, rows = load_holdout(evaluation)
    samples = list(evaluation.iter_ood_samples(contract, bank, rows))
    if len(samples) != 2700:
        raise RuntimeError(f"Expected 2700 OOD samples, received {len(samples)}")

    device = torch.device("cuda")
    poisson_model, provenance = load_poisson_model(device)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    outputs: list[np.ndarray | None] = [None] * len(samples)
    poisson_indices = [
        index
        for index, sample in enumerate(samples)
        if sample["condition"] == "poisson_gaussian"
    ]
    mild_indices = [
        index for index in poisson_indices if int(samples[index]["severity"]) == 1
    ]
    mild_base = infer_indices(poisson_model, samples, mild_indices, device)
    checkpoint = torch.load(POISSON_CHECKPOINT, map_location="cpu", weights_only=False)
    poisson_model.load_state_dict(checkpoint["model"], strict=True)
    poisson_model.eval()
    candidate_poisson = infer_indices(poisson_model, samples, poisson_indices, device)
    selected_fusion = fusion["selected"]
    fusion_sigma = float(selected_fusion["sigma"])
    fusion_alpha = float(selected_fusion["alpha"])
    for index in poisson_indices:
        image = candidate_poisson[index]
        if index in mild_base:
            base_image = mild_base[index]
            correction = gaussian_filter(
                image - base_image, fusion_sigma, mode="reflect"
            )
            image = np.clip(
                base_image + np.float32(fusion_alpha) * correction, 0.0, 1.0
            ).astype(np.float32)
        outputs[index] = image

    for index, sample in enumerate(samples):
        if outputs[index] is not None:
            continue
        family = sample["condition"]
        severity = str(int(sample["severity"]))
        record = physics["results"][family][severity]
        outputs[index] = common.restore_physics(
            sample["degraded"],
            family,
            record["parameters"],
            float(record["selected_gain"]),
        )

    candidate_records: list[dict[str, Any]] = []
    for start in range(0, len(samples), 32):
        batch_samples = samples[start : start + 32]
        batch_outputs = np.stack(outputs[start : start + 32]).astype(np.float32)
        clean = np.stack([sample["clean"] for sample in batch_samples]).astype(
            np.float32
        )
        clean_tensor = torch.from_numpy(clean[:, None]).to(device).repeat(1, 3, 1, 1)
        output_tensor = (
            torch.from_numpy(batch_outputs[:, None]).to(device).repeat(1, 3, 1, 1)
        )
        with torch.inference_mode():
            perceptual_values = (
                perceptual(clean_tensor * 2.0 - 1.0, output_tensor * 2.0 - 1.0)
                .flatten()
                .float()
                .cpu()
                .numpy()
            )
        for offset, (sample, output) in enumerate(zip(batch_samples, batch_outputs)):
            base = baseline[sample["sample_id"]]
            record = {
                "sample_id": sample["sample_id"],
                "parent_sha256": sample["parent_sha256"],
                "condition": sample["condition"],
                "severity": int(sample["severity"]),
                "stochastic_seed": int(sample["stochastic_seed"]),
                "candidate_psnr_db": common.psnr(sample["clean"], output),
                "candidate_lpips_alex": float(perceptual_values[offset]),
                "candidate_gradient_mae": common.gradient_mae(sample["clean"], output),
                "candidate_overshoot_p95": common.edge_normal_overshoot_p95(
                    sample["clean"], output
                ),
                "degraded_overshoot_p95": float(
                    base["degraded_edge_normal_overshoot_p95"]
                ),
                "base_psnr_db": float(base["restored_psnr_db"]),
                "base_lpips_alex": float(base["restored_lpips_alex"]),
                "base_gradient_mae": float(base["restored_gradient_mae"]),
            }
            record["delta_psnr"] = record["candidate_psnr_db"] - record["base_psnr_db"]
            record["delta_lpips"] = record["candidate_lpips_alex"] - record["base_lpips_alex"]
            record["delta_gradient"] = (
                record["candidate_gradient_mae"] - record["base_gradient_mae"]
            )
            record["delta_overshoot_candidate_minus_degraded"] = (
                record["candidate_overshoot_p95"] - record["degraded_overshoot_p95"]
            )
            candidate_records.append(record)

    with SAMPLE_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in candidate_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        grouped[(record["condition"], record["severity"])].append(record)
    by_cell: dict[str, dict[str, Any]] = defaultdict(dict)
    for (family, severity), records in grouped.items():
        summaries: dict[str, dict[str, float]] = {}
        for metric in (
            "delta_psnr",
            "delta_lpips",
            "delta_gradient",
            "delta_overshoot_candidate_minus_degraded",
        ):
            by_parent: dict[str, list[float]] = defaultdict(list)
            for record in records:
                by_parent[record["parent_sha256"]].append(float(record[metric]))
            summaries[metric] = bootstrap(by_parent, seed=20261015 + severity)
        passed = bool(
            summaries["delta_psnr"]["mean"] >= 0.1
            and summaries["delta_psnr"]["ci95_low"] > 0.0
            and summaries["delta_lpips"]["mean"] <= -0.005
            and summaries["delta_lpips"]["ci95_high"] < 0.0
            and summaries["delta_gradient"]["ci95_high"] <= 0.0
            and summaries["delta_overshoot_candidate_minus_degraded"]["ci95_high"]
            <= 0.01
        )
        by_cell[family][str(severity)] = {**summaries, "identity_pass": True, "pass": passed}
        print(
            json.dumps(
                {
                    "family": family,
                    "severity": severity,
                    "delta_psnr": summaries["delta_psnr"]["mean"],
                    "delta_lpips": summaries["delta_lpips"]["mean"],
                    "delta_gradient": summaries["delta_gradient"]["mean"],
                    "pass": passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    family_success = {
        family: all(cells[severity]["pass"] for severity in ("1", "2"))
        for family, cells in by_cell.items()
    }
    broad = all(family_success.values()) and sum(
        int(by_cell[family]["3"]["pass"]) for family in by_cell
    ) >= 4
    result = {
        "schema_version": "MERIT-C018-C009-DEVELOPMENT-EVALUATION-v1",
        "status": "PASS" if broad else "ITERATE",
        "scientific_status": "UNBLINDED_C009_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "identity": {
            "route": "exact bypass",
            "mean_absolute_output_change": 0.0,
            "p99_absolute_output_change": 0.0,
            "pass": True,
        },
        "by_family_severity": by_cell,
        "family_level_success": family_success,
        "broad_six_family_claim_pass": broad,
        "records": len(candidate_records),
        "parents": len({record["parent_sha256"] for record in candidate_records}),
        "poisson_checkpoint_sha256": common.sha256_file(POISSON_CHECKPOINT),
        "poisson_mild_fusion_sha256": common.sha256_file(POISSON_FUSION),
        "physics_scan_sha256": common.sha256_file(PHYSICS_SCAN),
        "baseline_metrics_sha256": common.sha256_file(BASE_METRICS),
        "sample_metrics_path": str(SAMPLE_OUTPUT.resolve()),
        "sample_metrics_sha256": common.sha256_file(SAMPLE_OUTPUT),
        "source_model": provenance,
    }
    common.write_json_atomic(OUTPUT, result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT.resolve())}))


if __name__ == "__main__":
    main()
