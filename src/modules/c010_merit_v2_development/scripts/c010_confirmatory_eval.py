"""One-shot evaluation on C010 parents excluded from C006/C009 development."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c010_pilot as pilot
from merit_v2_model import CalibratedPromptBlend


C010 = HERE.parent
DESIGN = C010 / "protocol" / "confirmatory_source_design_v1.json"
FREEZE = C010 / "protocol" / "confirmatory_freeze_v1.json"
ARTIFACTS = C010 / "artifacts" / "confirmatory_v1"
OUTPUT = C010 / "runs" / "confirmatory_v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    design = pilot.load_json(DESIGN)
    freeze = pilot.load_json(FREEZE)
    payload = dict(freeze)
    recorded = payload.pop("payload_sha256")
    if recorded != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise pilot.PilotError("Confirmatory freeze payload mismatch")
    if freeze.get("status") != "FROZEN_BEFORE_ANY_CONFIRMATORY_MODEL_OUTPUT":
        raise pilot.PilotError("Confirmatory evaluation is not frozen")
    checks = {
        "design_sha256": DESIGN,
        "bank_sha256": ARTIFACTS / "d0_bank_448.npy",
        "manifest_sha256": ARTIFACTS / "d0_bank_manifest.csv",
        "ledger_sha256": ARTIFACTS / "d0_bank_ledger.json",
        "evaluator_sha256": Path(__file__).resolve(),
        "model_code_sha256": HERE / "merit_v2_model.py",
        "v3_protocol_sha256": C010 / "protocol" / "pilot_v3.json",
    }
    for key, path in checks.items():
        if freeze[key] != pilot.sha256_file(path):
            raise pilot.PilotError(f"Confirmatory freeze hash mismatch: {key}")
    return design, freeze


def load_bank() -> tuple[np.ndarray, list[dict[str, str]]]:
    bank = np.load(ARTIFACTS / "d0_bank_448.npy", allow_pickle=False)
    with (ARTIFACTS / "d0_bank_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if bank.shape != (98, 448, 448) or len(rows) != 98:
        raise pilot.PilotError(f"Unexpected confirmatory bank shape: {bank.shape}")
    return bank.astype(np.float32) / np.float32(255.0), rows


def samples_for(
    design: dict[str, Any], bank: np.ndarray, rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for clean, row in zip(bank, rows):
        bank_index = int(row["bank_index"])
        for condition_index, condition in enumerate(design["synthetic_conditions"]):
            for seed in design["noise_seeds"]:
                samples.append(
                    {
                        "sample_id": f"confirm__b{bank_index}__{condition['id']}__n{seed}",
                        "domain": "id",
                        "parent_sha256": row["parent_sha256"],
                        "condition": condition["id"],
                        "clean": clean,
                        "degraded": pilot.degrade(
                            clean,
                            condition,
                            pilot.deterministic_rng(
                                int(seed), bank_index + 1_000_000, condition_index
                            ),
                        ),
                        "p_b": (
                            float(condition["sigma_h_px"])
                            + float(condition["sigma_v_px"])
                        )
                        / 6.0,
                        "p_n": float(condition["noise_std_8bit"]) / 3.0,
                    }
                )
        samples.append(
            {
                "sample_id": f"identity__b{bank_index}",
                "domain": "identity",
                "parent_sha256": row["parent_sha256"],
                "condition": "identity",
                "clean": clean,
                "degraded": clean.copy(),
                "p_b": 0.0,
                "p_n": 0.0,
            }
        )
    return samples


def load_model(seed: int, freeze: dict[str, Any], device: torch.device):
    base, metadata = pilot.factory.build_official_adapter("nafnet")
    checkpoint_path = (
        pilot.C009 / "runs" / "wp1" / "nafnet" / f"base_seed{seed}" / "final.pt"
    )
    expected = freeze["base_checkpoints"][f"seed{seed}"]["sha256"]
    if pilot.sha256_file(checkpoint_path) != expected:
        raise pilot.PilotError(f"Base checkpoint changed for seed {seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["model"], strict=True)
    calibration = freeze["calibration"]
    model = CalibratedPromptBlend(
        base,
        calibration["severity_anchors"],
        calibration["blend_anchors"],
    ).to(device).eval()
    return model, {
        "seed": seed,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": expected,
        "model": metadata,
    }


def evaluate_seed(
    seed: int,
    samples: list[dict[str, Any]],
    freeze: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, provenance = load_model(seed, freeze, device)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in pilot.batches(samples, 16):
            degraded = np.stack([sample["degraded"] for sample in batch]).astype(np.float32)
            inputs = torch.from_numpy(
                pilot.common.prompt_tensor(
                    degraded,
                    [sample["p_b"] for sample in batch],
                    [sample["p_n"] for sample in batch],
                )
            ).to(device)
            with pilot.autocast(device):
                v2_preclip, base_preclip, _, blend = model.forward_components(inputs)
            v2 = np.clip(v2_preclip.float().cpu().numpy()[:, 0], 0.0, 1.0)
            base = np.clip(base_preclip.float().cpu().numpy()[:, 0], 0.0, 1.0)
            blend_values = blend.float().flatten().cpu().numpy()
            for index, sample in enumerate(batch):
                clean = sample["clean"]
                source = sample["degraded"]
                base_psnr = pilot.psnr(clean, base[index])
                v2_psnr = pilot.psnr(clean, v2[index])
                base_grad = pilot.gradient_mae(clean, base[index])
                v2_grad = pilot.gradient_mae(clean, v2[index])
                base_over = pilot.edge_normal_overshoot_p95(clean, base[index])
                v2_over = pilot.edge_normal_overshoot_p95(clean, v2[index])
                records.append(
                    {
                        "seed": seed,
                        "sample_id": sample["sample_id"],
                        "domain": sample["domain"],
                        "parent_sha256": sample["parent_sha256"],
                        "condition": sample["condition"],
                        "delta_psnr_db": v2_psnr - base_psnr,
                        "delta_gradient_mae": v2_grad - base_grad,
                        "delta_overshoot_p95": v2_over - base_over,
                        "v2_source_mean_abs": float(np.mean(np.abs(v2[index] - source))),
                        "v2_source_p99_abs": float(np.percentile(np.abs(v2[index] - source), 99.0)),
                        "blend": float(blend_values[index]),
                    }
                )
    restoration = [record for record in records if record["domain"] == "id"]
    identity = [record for record in records if record["domain"] == "identity"]
    reps = int(freeze["success_gates"]["bootstrap_replicates"])
    psnr = pilot.parent_bootstrap(restoration, "delta_psnr_db", reps)
    gradient = pilot.parent_bootstrap(restoration, "delta_gradient_mae", reps)
    overshoot = pilot.parent_bootstrap(restoration, "delta_overshoot_p95", reps)
    identity_mean = max(record["v2_source_mean_abs"] for record in identity)
    identity_p99 = max(record["v2_source_p99_abs"] for record in identity)
    gates = {
        "psnr": psnr["mean"] >= 0.1 and psnr["ci_low"] > 0.0,
        "gradient": gradient["mean"] < 0.0,
        "overshoot": overshoot["mean"] <= 0.0,
        "identity": identity_mean <= 1e-7 and identity_p99 <= 1e-7,
    }
    summary = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "provenance": provenance,
        "records": len(records),
        "restoration_records": len(restoration),
        "identity_records": len(identity),
        "delta_psnr_db": psnr,
        "delta_gradient_mae": gradient,
        "delta_overshoot_p95": overshoot,
        "identity": {
            "max_mean_absolute_change": identity_mean,
            "max_p99_absolute_change": identity_p99,
        },
        "gates": gates,
    }
    del model
    torch.cuda.empty_cache()
    return summary, records


def main() -> None:
    if not torch.cuda.is_available():
        raise pilot.PilotError("CUDA is required")
    complete = OUTPUT / "RUN_COMPLETE.json"
    if complete.exists():
        print(complete.read_text(encoding="utf-8"))
        return
    design, freeze = verify_freeze()
    bank, rows = load_bank()
    samples = samples_for(design, bank, rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for seed in freeze["base_seeds"]:
        summary, records = evaluate_seed(int(seed), samples, freeze, device)
        summaries[f"seed{seed}"] = summary
        all_records.extend(records)
        metrics_path = OUTPUT / f"seed{seed}_metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        summary["metrics_path"] = str(metrics_path.resolve())
        summary["metrics_sha256"] = pilot.sha256_file(metrics_path)
    restoration = [record for record in all_records if record["domain"] == "id"]
    identity = [record for record in all_records if record["domain"] == "identity"]
    reps = int(freeze["success_gates"]["bootstrap_replicates"])
    macro_psnr = pilot.parent_bootstrap(restoration, "delta_psnr_db", reps)
    macro_gradient = pilot.parent_bootstrap(restoration, "delta_gradient_mae", reps)
    macro_overshoot = pilot.parent_bootstrap(restoration, "delta_overshoot_p95", reps)
    macro_identity_mean = max(record["v2_source_mean_abs"] for record in identity)
    macro_identity_p99 = max(record["v2_source_p99_abs"] for record in identity)
    macro_gates = {
        "psnr": macro_psnr["mean"] >= 0.1 and macro_psnr["ci_low"] > 0.0,
        "gradient": macro_gradient["mean"] < 0.0,
        "overshoot": macro_overshoot["mean"] <= 0.0,
        "identity": macro_identity_mean <= 1e-7 and macro_identity_p99 <= 1e-7,
        "all_seeds": all(summary["status"] == "PASS" for summary in summaries.values()),
    }
    result = {
        "schema_version": "MERIT-C010-V2-CONFIRMATORY-RESULT-v1",
        "status": "PASS" if all(macro_gates.values()) else "FAIL",
        "scientific_status": "RETROSPECTIVE_UNUSED_CONDITION_PARENT_DISJOINT_SYNTHETIC_CONFIRMATION",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "claim_boundary": design["claim_boundary"],
        "freeze_path": str(FREEZE.resolve()),
        "freeze_sha256": pilot.sha256_file(FREEZE),
        "parents": len({row["parent_sha256"] for row in rows}),
        "crops": len(rows),
        "per_seed": summaries,
        "macro": {
            "delta_psnr_db": macro_psnr,
            "delta_gradient_mae": macro_gradient,
            "delta_overshoot_p95": macro_overshoot,
            "identity": {
                "max_mean_absolute_change": macro_identity_mean,
                "max_p99_absolute_change": macro_identity_p99,
            },
            "gates": macro_gates,
        },
    }
    pilot.write_json_atomic(complete, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
