"""Build the R1-2-aligned paired PSNR/LPIPS block for revised Table 5.

The script computes the missing NAFNet LPIPS diagnostic on the frozen C010
confirmation bank, checks that rerun PSNR reproduces C010, and then combines
the frozen C010, C039, and C049 summaries without comparing absolute values
across their different parent-disjoint confirmation cohorts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C060 = HERE.parent
REVISION = C060.parent
C010 = REVISION / "c010_merit_v2_development"
C039 = REVISION / "c039_c036_independent_confirmation"
C049 = REVISION / "c049_restormer_psnr_independent_confirmation"
OUTPUT = C060 / "artifacts" / "table5_final_v1"

sys.path.insert(0, str(C010 / "scripts"))
import c010_confirmatory_eval as c010  # noqa: E402
import c010_pilot as pilot  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def lpips_scores(
    model: torch.nn.Module,
    clean: torch.Tensor,
    candidate: torch.Tensor,
) -> np.ndarray:
    clean_rgb = clean.repeat(1, 3, 1, 1) * 2.0 - 1.0
    candidate_rgb = candidate.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.inference_mode():
        values = model(clean_rgb, candidate_rgb, normalize=False)
    return values.flatten().float().cpu().numpy().astype(np.float64)


def evaluate_nafnet_seed(
    seed: int,
    samples: list[dict[str, Any]],
    freeze: dict[str, Any],
    perceptual: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    pilot.set_determinism(seed)
    model, provenance = c010.load_model(seed, freeze, device)
    records: list[dict[str, Any]] = []
    restoration = [sample for sample in samples if sample["domain"] == "id"]
    with torch.inference_mode():
        for batch in pilot.batches(restoration, 8):
            degraded = np.stack([sample["degraded"] for sample in batch]).astype(np.float32)
            inputs = torch.from_numpy(
                pilot.common.prompt_tensor(
                    degraded,
                    [sample["p_b"] for sample in batch],
                    [sample["p_n"] for sample in batch],
                )
            ).to(device)
            with pilot.autocast(device):
                candidate_preclip, base_preclip, _, _ = model.forward_components(inputs)
            candidate = candidate_preclip.float().clamp(0.0, 1.0)
            base = base_preclip.float().clamp(0.0, 1.0)
            clean = torch.from_numpy(
                np.stack([sample["clean"] for sample in batch]).astype(np.float32)[:, None]
            ).to(device)
            candidate_lpips = lpips_scores(perceptual, clean, candidate)
            base_lpips = lpips_scores(perceptual, clean, base)
            candidate_np = candidate[:, 0].cpu().numpy()
            base_np = base[:, 0].cpu().numpy()
            for index, sample in enumerate(batch):
                records.append(
                    {
                        "model_seed": seed,
                        "sample_id": sample["sample_id"],
                        "condition": sample["condition"],
                        "parent_sha256": sample["parent_sha256"],
                        "delta_psnr_db": pilot.psnr(sample["clean"], candidate_np[index])
                        - pilot.psnr(sample["clean"], base_np[index]),
                        "delta_lpips_alex": float(candidate_lpips[index] - base_lpips[index]),
                    }
                )
    del model
    torch.cuda.empty_cache()
    records_path = OUTPUT / f"nafnet_seed{seed}_records.jsonl"
    write_jsonl(records_path, records)
    metadata = {
        "seed": seed,
        "records": len(records),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "records_sha256": sha256_file(records_path),
    }
    write_json(OUTPUT / f"nafnet_seed{seed}.json", metadata)
    return records


def summarize_records(
    records: list[dict[str, Any]], freeze: dict[str, Any]
) -> dict[str, Any]:
    replicates = int(freeze["success_gates"]["bootstrap_replicates"])
    conditions: dict[str, Any] = {}
    for condition in ("id_low", "id_mid", "id_high"):
        subset = [row for row in records if row["condition"] == condition]
        conditions[condition] = {
            "parents": len({row["parent_sha256"] for row in subset}),
            "delta_psnr_db": pilot.parent_bootstrap(subset, "delta_psnr_db", replicates),
            "delta_lpips_alex": pilot.parent_bootstrap(
                subset, "delta_lpips_alex", replicates
            ),
            "seed_delta_psnr_means": {
                str(seed): float(
                    np.mean(
                        [
                            row["delta_psnr_db"]
                            for row in subset
                            if row["model_seed"] == seed
                        ]
                    )
                )
                for seed in freeze["base_seeds"]
            },
        }
    pooled = {
        "parents": len({row["parent_sha256"] for row in records}),
        "delta_psnr_db": pilot.parent_bootstrap(records, "delta_psnr_db", replicates),
        "delta_lpips_alex": pilot.parent_bootstrap(
            records, "delta_lpips_alex", replicates
        ),
    }
    return {"conditions": conditions, "pooled": pooled}


def run_nafnet() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen C010 NAFNet evaluation")
    design, freeze = c010.verify_freeze()
    bank, rows = c010.load_bank()
    samples = c010.samples_for(design, bank, rows)
    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    all_records: list[dict[str, Any]] = []
    for seed in freeze["base_seeds"]:
        records_path = OUTPUT / f"nafnet_seed{seed}_records.jsonl"
        metadata_path = OUTPUT / f"nafnet_seed{seed}.json"
        if records_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if (
                len(cached) == 882
                and metadata.get("records") == 882
                and metadata.get("records_sha256") == sha256_file(records_path)
                and metadata.get("checkpoint_sha256")
                == freeze["base_checkpoints"][f"seed{seed}"]["sha256"]
                and all(int(row["model_seed"]) == int(seed) for row in cached)
            ):
                all_records.extend(cached)
                continue
        all_records.extend(
            evaluate_nafnet_seed(int(seed), samples, freeze, perceptual, device)
        )
    summary = summarize_records(all_records, freeze)
    frozen = json.loads(
        (C010 / "runs" / "confirmatory_v1" / "RUN_COMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    reproduced = summary["pooled"]["delta_psnr_db"]
    expected = frozen["macro"]["delta_psnr_db"]
    max_abs_error = max(
        abs(reproduced["mean"] - expected["mean"]),
        abs(reproduced["ci_low"] - expected["ci_low"]),
        abs(reproduced["ci_high"] - expected["ci_high"]),
    )
    if max_abs_error > 1e-8:
        raise RuntimeError(f"C010 PSNR reproduction mismatch: {max_abs_error}")
    result = {
        "schema_version": "MERIT-C060-NAFNET-LPIPS-v1",
        "status": "PASS",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "records": len(all_records),
        "parents": len({row["parent_sha256"] for row in all_records}),
        "crops": len(rows),
        "model_seeds": [int(seed) for seed in freeze["base_seeds"]],
        "conditions": [item["id"] for item in design["synthetic_conditions"]],
        "c010_psnr_reproduction_max_abs_error": max_abs_error,
        "summary": summary,
        "provenance": {
            "c010_freeze": str(c010.FREEZE.resolve()),
            "c010_freeze_sha256": sha256_file(c010.FREEZE),
            "c010_run_complete_sha256": sha256_file(
                C010 / "runs" / "confirmatory_v1" / "RUN_COMPLETE.json"
            ),
            "lpips": "AlexNet LPIPS; grayscale repeated to RGB; input scaled to [-1,1]",
        },
    }
    write_json(OUTPUT / "nafnet_summary.json", result)
    return result


def metric_cell(metric: dict[str, Any], c039_style: bool) -> dict[str, float]:
    if c039_style:
        return {
            "mean": float(metric["mean"]),
            "ci_low": float(metric["ci95_lower"]),
            "ci_high": float(metric["ci95_upper"]),
        }
    return {
        "mean": float(metric["mean"]),
        "ci_low": float(metric["ci_low"]),
        "ci_high": float(metric["ci_high"]),
    }


def assemble() -> dict[str, Any]:
    naf_path = OUTPUT / "nafnet_summary.json"
    if not naf_path.exists():
        raise RuntimeError("Run the nafnet action before assembling Table 5")
    naf = json.loads(naf_path.read_text(encoding="utf-8"))
    mpr_path = C039 / "runs" / "confirmation_v1" / "confirmation_summary.json"
    rest_path = C049 / "runs" / "confirmation_v1" / "confirmation_summary.json"
    mpr = json.loads(mpr_path.read_text(encoding="utf-8"))
    rest = json.loads(rest_path.read_text(encoding="utf-8"))
    if naf["status"] != "PASS" or mpr["status"] != "PASS" or rest["status"] != "PASS":
        raise RuntimeError("One or more frozen backbone confirmations did not pass")

    cohort = {
        "NAFNet": {"parents": 68, "crops": 98, "bank": "C010 unused-condition bank"},
        "MPRNet": {"parents": 133, "crops": 191, "bank": "C036 ISO200 bank"},
        "Restormer": {"parents": 302, "crops": 450, "bank": "C040 ISO400 bank"},
    }
    rows: list[dict[str, Any]] = []
    for backbone, source, style in (
        ("NAFNet", naf["summary"], False),
        ("MPRNet", mpr, True),
        ("Restormer", rest, True),
    ):
        condition_block = source["conditions"]
        for condition in ("id_low", "id_mid", "id_high"):
            item = condition_block[condition]
            psnr = metric_cell(item["delta_psnr_db"] if backbone == "NAFNet" else item["metrics"]["delta_psnr_db"], style)
            lpips_metric = item["delta_lpips_alex"] if backbone == "NAFNet" else item["metrics"]["delta_lpips_alex"]
            lpips_value = metric_cell(lpips_metric, style)
            rows.append(
                {
                    "backbone": backbone,
                    "condition": condition.replace("id_", ""),
                    "parents": cohort[backbone]["parents"],
                    "crops": cohort[backbone]["crops"],
                    "delta_psnr_mean_db": psnr["mean"],
                    "delta_psnr_ci95_low_db": psnr["ci_low"],
                    "delta_psnr_ci95_high_db": psnr["ci_high"],
                    "delta_lpips_alex_mean": lpips_value["mean"],
                    "delta_lpips_alex_ci95_low": lpips_value["ci_low"],
                    "delta_lpips_alex_ci95_high": lpips_value["ci_high"],
                }
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "table5_final.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": "MERIT-C060-R1-2-ALIGNED-TABLE5-v1",
        "status": "PASS",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "estimand": "final frozen MERIT route minus corresponding backbone baseline on paired controlled references",
        "interpretation": {
            "positive_delta_psnr": "higher agreement with the operational source reference",
            "negative_delta_lpips": "lower AlexNet LPIPS distance to the operational source reference",
            "nafnet_lpips": "improves at low severity and when pooled, but worsens at medium and high severity",
            "cross_backbone_ranking": "not permitted because confirmation cohorts and backbone-specific routes differ",
            "actual_field": "PSNR and LPIPS are N/A because no registered paired source reference exists",
        },
        "conditions": {
            "low": {"sigma_h_px": 0.7, "sigma_v_px": 0.6, "noise_std_8bit": 0.75},
            "mid": {"sigma_h_px": 1.6, "sigma_v_px": 1.3, "noise_std_8bit": 1.5},
            "high": {"sigma_h_px": 2.7, "sigma_v_px": 2.3, "noise_std_8bit": 2.75},
        },
        "cohorts": cohort,
        "rows": rows,
        "provenance": {
            "nafnet_summary": str(naf_path.resolve()),
            "nafnet_summary_sha256": sha256_file(naf_path),
            "mprnet_summary": str(mpr_path.resolve()),
            "mprnet_summary_sha256": sha256_file(mpr_path),
            "restormer_summary": str(rest_path.resolve()),
            "restormer_summary_sha256": sha256_file(rest_path),
            "csv_sha256": sha256_file(csv_path),
        },
    }
    write_json(OUTPUT / "table5_final.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("nafnet", "assemble", "all"))
    args = parser.parse_args()
    if args.action in ("nafnet", "all"):
        run_nafnet()
    result = assemble() if args.action in ("assemble", "all") else {"status": "PASS"}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
