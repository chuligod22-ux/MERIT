"""Develop a bounded base/candidate frequency fusion for mild Poisson noise."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime

import lpips
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


HERE = __import__("pathlib").Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402
from c018_evaluate_development import bootstrap  # noqa: E402


RUN_ROOT = common.C018 / "runs" / "development_v1"
CHECKPOINT = RUN_ROOT / "poisson_nafnet_seed0" / "final.pt"
OUTPUT = RUN_ROOT / "poisson_mild_fusion_scan.json"


def infer(model, degraded: np.ndarray, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(degraded), 16):
        source = degraded[start : start + 16]
        zeros = np.zeros_like(source)
        inputs = torch.from_numpy(np.stack([source, zeros, zeros], axis=1)).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            prediction = model.forward_preclip(inputs)
        outputs.append(torch.clamp(prediction.float(), 0.0, 1.0).cpu().numpy()[:, 0])
    return np.concatenate(outputs).astype(np.float32)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    evaluation = common.import_file(
        "c009_controlled_eval_c018_mild",
        common.C009 / "scripts" / "c009_controlled_eval.py",
    )
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
    parameters = contract["wp4_ood"]["families"]["poisson_gaussian"][0]
    family_index = list(contract["wp4_ood"]["families"]).index("poisson_gaussian")
    samples: list[dict] = []
    for clean, row in zip(bank, rows):
        bank_index = int(row["bank_index"])
        for seed in contract["wp4_ood"]["stochastic_seeds"]:
            rng = evaluation.deterministic_rng(
                int(seed), bank_index, 100 + family_index * 10 + 1
            )
            samples.append(
                {
                    "clean": clean,
                    "degraded": evaluation.ood_degrade(
                        clean, "poisson_gaussian", parameters, rng
                    ),
                    "parent_sha256": row["parent_sha256"],
                }
            )
    clean = np.stack([sample["clean"] for sample in samples]).astype(np.float32)
    degraded = np.stack([sample["degraded"] for sample in samples]).astype(np.float32)
    device = torch.device("cuda")
    model, provenance = evaluation.common.build_and_load_model(
        {"kind": "wp1", "family": "nafnet", "arm": "base", "seed": 0}, device
    )
    base = infer(model, degraded, device)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    candidate = infer(model, degraded, device)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    clean_tensor = torch.from_numpy(clean[:, None]).to(device).repeat(1, 3, 1, 1)
    with torch.inference_mode():
        base_lpips = (
            perceptual(
                clean_tensor * 2.0 - 1.0,
                torch.from_numpy(base[:, None]).to(device).repeat(1, 3, 1, 1) * 2.0 - 1.0,
            )
            .flatten().float().cpu().numpy()
        )

    configurations: dict[str, dict] = {}
    for sigma in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        correction = gaussian_filter(
            candidate - base, sigma=(0.0, sigma, sigma), mode="reflect"
        )
        for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25):
            fused = np.clip(base + np.float32(alpha) * correction, 0.0, 1.0).astype(
                np.float32
            )
            with torch.inference_mode():
                fused_lpips = (
                    perceptual(
                        clean_tensor * 2.0 - 1.0,
                        torch.from_numpy(fused[:, None]).to(device).repeat(1, 3, 1, 1)
                        * 2.0
                        - 1.0,
                    )
                    .flatten().float().cpu().numpy()
                )
            metrics: dict[str, dict[str, list[float]]] = {
                key: defaultdict(list)
                for key in ("delta_psnr", "delta_lpips", "delta_gradient", "delta_overshoot")
            }
            for index, sample in enumerate(samples):
                parent = sample["parent_sha256"]
                metrics["delta_psnr"][parent].append(
                    common.psnr(clean[index], fused[index])
                    - common.psnr(clean[index], base[index])
                )
                metrics["delta_lpips"][parent].append(
                    float(fused_lpips[index] - base_lpips[index])
                )
                metrics["delta_gradient"][parent].append(
                    common.gradient_mae(clean[index], fused[index])
                    - common.gradient_mae(clean[index], base[index])
                )
                metrics["delta_overshoot"][parent].append(
                    common.edge_normal_overshoot_p95(clean[index], fused[index])
                    - common.edge_normal_overshoot_p95(clean[index], degraded[index])
                )
            summary = {
                key: bootstrap(values, seed=20261016) for key, values in metrics.items()
            }
            passed = bool(
                summary["delta_psnr"]["mean"] >= 0.1
                and summary["delta_psnr"]["ci95_low"] > 0.0
                and summary["delta_lpips"]["mean"] <= -0.005
                and summary["delta_lpips"]["ci95_high"] < 0.0
                and summary["delta_gradient"]["ci95_high"] <= 0.0
                and summary["delta_overshoot"]["ci95_high"] <= 0.01
            )
            configurations[f"sigma{sigma:.2f}_alpha{alpha:.2f}"] = {
                "sigma": sigma,
                "alpha": alpha,
                **summary,
                "pass": passed,
            }
            print(
                json.dumps(
                    {
                        "sigma": sigma,
                        "alpha": alpha,
                        "dpsnr": summary["delta_psnr"]["mean"],
                        "dlpips": summary["delta_lpips"]["mean"],
                        "dgradient": summary["delta_gradient"]["mean"],
                        "pass": passed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    passing = [value for value in configurations.values() if value["pass"]]
    selected = max(
        passing or configurations.values(),
        key=lambda value: (
            bool(value["pass"]),
            -abs(float(value["alpha"]) - 0.5),
            value["delta_psnr"]["ci95_low"],
        ),
    )
    result = {
        "schema_version": "MERIT-C018-POISSON-MILD-FUSION-SCAN-v1",
        "status": "PASS" if passing else "ITERATE",
        "scientific_status": "UNBLINDED_C009_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected": selected,
        "passing_configurations": len(passing),
        "configurations": configurations,
        "checkpoint_sha256": common.sha256_file(CHECKPOINT),
        "source_model": provenance,
    }
    common.write_json_atomic(OUTPUT, result)
    print(json.dumps({"status": result["status"], "selected": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
