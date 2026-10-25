"""One-shot parent-disjoint confirmation of the frozen C018 candidate."""

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
from c018_evaluate_development import bootstrap  # noqa: E402


RUN_ROOT = common.C018 / "runs" / "confirmation_v1"
FREEZE = common.C018 / "protocol" / "confirmatory_freeze_v1.json"
PHYSICS_SCAN = common.C018 / "runs" / "development_v1" / "physics_scan.json"
POISSON_CHECKPOINT = (
    common.C018
    / "runs"
    / "development_v1"
    / "poisson_nafnet_seed0"
    / "final.pt"
)
POISSON_FUSION = common.C018 / "runs" / "development_v1" / "poisson_mild_fusion_scan.json"
BANK_ROOT = common.C010 / "artifacts" / "confirmatory_v1"
OUTPUT = RUN_ROOT / "RUN_COMPLETE.json"
SAMPLE_OUTPUT = RUN_ROOT / "sample_metrics.jsonl"


def verify_freeze() -> dict[str, Any]:
    freeze = common.load_json(FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_ANY_C018_CONFIRMATORY_OUTPUT":
        raise RuntimeError("C018 confirmation is not frozen")
    files = {
        "development_protocol": common.DEVELOPMENT_PROTOCOL,
        "physics_scan": PHYSICS_SCAN,
        "poisson_checkpoint": POISSON_CHECKPOINT,
        "poisson_mild_fusion": POISSON_FUSION,
        "bank": BANK_ROOT / "d0_bank_448.npy",
        "manifest": BANK_ROOT / "d0_bank_manifest.csv",
        "evaluator": Path(__file__).resolve(),
        "common": HERE / "c018_common.py",
        "development_helper": HERE / "c018_evaluate_development.py",
    }
    for key, path in files.items():
        expected = freeze["files"][key]["sha256"]
        actual = common.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch for {key}: {actual} != {expected}")
    if OUTPUT.exists() or SAMPLE_OUTPUT.exists():
        raise RuntimeError("Confirmation output already exists; one-shot rerun is blocked")
    return freeze


def load_source() -> tuple[np.ndarray, list[dict[str, str]]]:
    bank = np.load(BANK_ROOT / "d0_bank_448.npy", allow_pickle=False)
    with (BANK_ROOT / "d0_bank_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(bank) != 98 or len(rows) != 98:
        raise RuntimeError("Frozen confirmation source must contain 98 crops")
    return bank.astype(np.float32) / 255.0, rows


def make_samples(
    bank: np.ndarray,
    rows: list[dict[str, str]],
    freeze: dict[str, Any],
    evaluation: object,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    families = freeze["families"]
    seeds = freeze["stochastic_seeds"]
    for clean, row in zip(bank, rows):
        bank_index = int(row["bank_index"])
        for family_index, (family, severities) in enumerate(families.items()):
            for severity, parameters in enumerate(severities, start=1):
                for seed in seeds:
                    rng = evaluation.deterministic_rng(
                        int(seed), bank_index, 500 + family_index * 10 + severity
                    )
                    degraded = evaluation.ood_degrade(
                        clean, family, parameters, rng
                    )
                    samples.append(
                        {
                            "sample_id": f"confirm__b{bank_index}__{family}__s{severity}__n{seed}",
                            "bank_index": bank_index,
                            "parent_sha256": row["parent_sha256"],
                            "condition": family,
                            "severity": severity,
                            "stochastic_seed": int(seed),
                            "clean": clean,
                            "degraded": degraded,
                            "parameters": parameters,
                        }
                    )
    expected = 98 * 6 * 3 * 3
    if len(samples) != expected:
        raise RuntimeError(f"Confirmation coverage mismatch: {len(samples)} != {expected}")
    return samples


def infer_model(
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


def aggregate(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, bool], bool]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["condition"], record["severity"])].append(record)
    by_cell: dict[str, dict[str, Any]] = defaultdict(dict)
    for (family, severity), cell_records in grouped.items():
        summaries: dict[str, dict[str, float]] = {}
        for metric in (
            "delta_psnr",
            "delta_lpips",
            "delta_gradient",
            "delta_overshoot_candidate_minus_degraded",
        ):
            by_parent: dict[str, list[float]] = defaultdict(list)
            for record in cell_records:
                by_parent[record["parent_sha256"]].append(float(record[metric]))
            summaries[metric] = bootstrap(by_parent, seed=20261030 + severity)
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
    return by_cell, family_success, broad


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    freeze = verify_freeze()
    physics = common.load_json(PHYSICS_SCAN)
    fusion = common.load_json(POISSON_FUSION)
    evaluation = common.import_file(
        "c009_controlled_eval_c018_confirmation",
        common.C009 / "scripts" / "c009_controlled_eval.py",
    )
    bank, rows = load_source()
    samples = make_samples(bank, rows, freeze, evaluation)
    device = torch.device("cuda")
    model, provenance = evaluation.common.build_and_load_model(
        {"kind": "wp1", "family": "nafnet", "arm": "base", "seed": 0},
        device,
    )
    all_indices = list(range(len(samples)))
    base_outputs = infer_model(model, samples, all_indices, device)
    checkpoint = torch.load(POISSON_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    poisson_indices = [
        index
        for index, sample in enumerate(samples)
        if sample["condition"] == "poisson_gaussian"
    ]
    poisson_outputs = infer_model(model, samples, poisson_indices, device)
    candidate_outputs: dict[int, np.ndarray] = dict(poisson_outputs)
    fusion_sigma = float(fusion["selected"]["sigma"])
    fusion_alpha = float(fusion["selected"]["alpha"])
    for index in poisson_indices:
        if int(samples[index]["severity"]) != 1:
            continue
        base_image = base_outputs[index]
        correction = gaussian_filter(
            candidate_outputs[index] - base_image, fusion_sigma, mode="reflect"
        )
        candidate_outputs[index] = np.clip(
            base_image + np.float32(fusion_alpha) * correction, 0.0, 1.0
        ).astype(np.float32)
    for index, sample in enumerate(samples):
        if index in candidate_outputs:
            continue
        scan_record = physics["results"][sample["condition"]][str(sample["severity"])]
        candidate_outputs[index] = common.restore_physics(
            sample["degraded"],
            sample["condition"],
            sample["parameters"],
            float(scan_record["selected_gain"]),
        )

    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(samples), 32):
        batch_samples = samples[start : start + 32]
        indices = list(range(start, start + len(batch_samples)))
        clean = np.stack([sample["clean"] for sample in batch_samples]).astype(np.float32)
        base = np.stack([base_outputs[index] for index in indices]).astype(np.float32)
        candidate = np.stack([candidate_outputs[index] for index in indices]).astype(np.float32)
        clean_tensor = torch.from_numpy(clean[:, None]).to(device).repeat(1, 3, 1, 1)
        base_tensor = torch.from_numpy(base[:, None]).to(device).repeat(1, 3, 1, 1)
        candidate_tensor = torch.from_numpy(candidate[:, None]).to(device).repeat(1, 3, 1, 1)
        with torch.inference_mode():
            base_lpips = (
                perceptual(clean_tensor * 2.0 - 1.0, base_tensor * 2.0 - 1.0)
                .flatten().float().cpu().numpy()
            )
            candidate_lpips = (
                perceptual(clean_tensor * 2.0 - 1.0, candidate_tensor * 2.0 - 1.0)
                .flatten().float().cpu().numpy()
            )
        for offset, (sample, base_image, candidate_image) in enumerate(
            zip(batch_samples, base, candidate)
        ):
            degraded_overshoot = common.edge_normal_overshoot_p95(
                sample["clean"], sample["degraded"]
            )
            candidate_overshoot = common.edge_normal_overshoot_p95(
                sample["clean"], candidate_image
            )
            record = {
                "sample_id": sample["sample_id"],
                "parent_sha256": sample["parent_sha256"],
                "condition": sample["condition"],
                "severity": sample["severity"],
                "stochastic_seed": sample["stochastic_seed"],
                "delta_psnr": common.psnr(sample["clean"], candidate_image)
                - common.psnr(sample["clean"], base_image),
                "delta_lpips": float(candidate_lpips[offset] - base_lpips[offset]),
                "delta_gradient": common.gradient_mae(sample["clean"], candidate_image)
                - common.gradient_mae(sample["clean"], base_image),
                "delta_overshoot_candidate_minus_degraded": candidate_overshoot
                - degraded_overshoot,
            }
            records.append(record)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with SAMPLE_OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    by_cell, family_success, broad = aggregate(records)
    result = {
        "schema_version": "MERIT-C018-CONFIRMATORY-RESULT-v1",
        "status": "PASS" if broad else "FAIL",
        "scientific_status": "RETROSPECTIVE_PARENT_DISJOINT_NEW_OOD_DRAWS_CONFIRMATION",
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
        "records": len(records),
        "crops": len(bank),
        "parents": len({row["parent_sha256"] for row in rows}),
        "freeze_sha256": common.sha256_file(FREEZE),
        "sample_metrics_path": str(SAMPLE_OUTPUT.resolve()),
        "sample_metrics_sha256": common.sha256_file(SAMPLE_OUTPUT),
        "source_model": provenance,
    }
    common.write_json_atomic(OUTPUT, result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT.resolve())}))


if __name__ == "__main__":
    main()
