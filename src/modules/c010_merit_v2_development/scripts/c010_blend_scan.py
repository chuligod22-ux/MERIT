"""Measure the development-set ceiling of prompt-conditioned base blending."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c010_pilot as pilot


OUTPUT = pilot.C010 / "runs" / "blend_scan_v1.json"


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = pilot.load_json(pilot.DEFAULT_PROTOCOL)
    pilot.set_determinism(int(protocol["training"]["seed"]))
    model, provenance = pilot.build_model(protocol, device)
    model.eval()
    contract = pilot.common.load_contract()
    bank, rows = pilot.load_holdout(contract)
    samples = [
        sample
        for sample in pilot.development_samples(contract, bank, rows)
        if sample["domain"] == "id"
    ]
    alphas = np.linspace(0.75, 1.25, 41, dtype=np.float32)
    sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(alphas), dtype=np.float64)
    )
    counts: dict[str, int] = defaultdict(int)
    baseline: dict[str, float] = defaultdict(float)
    with torch.inference_mode():
        for batch in pilot.batches(samples, 16):
            degraded = np.stack([sample["degraded"] for sample in batch]).astype(np.float32)
            source = torch.from_numpy(degraded[:, None]).to(device)
            zeros = torch.zeros_like(source)
            inputs = torch.cat([source, zeros, zeros], dim=1)
            with pilot.autocast(device):
                base_tensor = model.base.forward_preclip(inputs)
            base = np.clip(base_tensor.float().cpu().numpy()[:, 0], 0.0, 1.0)
            for index, sample in enumerate(batch):
                clean = sample["clean"]
                condition = str(sample["condition"])
                base_psnr = pilot.psnr(clean, base[index])
                baseline[condition] += base_psnr
                baseline["all"] += base_psnr
                counts[condition] += 1
                counts["all"] += 1
                direction = base[index] - degraded[index]
                for alpha_index, alpha in enumerate(alphas):
                    candidate = np.clip(degraded[index] + alpha * direction, 0.0, 1.0)
                    value = pilot.psnr(clean, candidate)
                    sums[condition][alpha_index] += value
                    sums["all"][alpha_index] += value
    result: dict[str, object] = {
        "schema_version": "MERIT-C010-BLEND-SCAN-v1",
        "status": "EXPLORATORY_DEVELOPMENT_ONLY",
        "provenance": provenance,
        "alpha_grid": [float(value) for value in alphas],
        "conditions": {},
    }
    condition_results: dict[str, object] = {}
    for condition in sorted(sums):
        means = sums[condition] / counts[condition]
        best_index = int(np.argmax(means))
        base_mean = baseline[condition] / counts[condition]
        condition_results[condition] = {
            "records": counts[condition],
            "base_mean_psnr_db": float(base_mean),
            "best_alpha": float(alphas[best_index]),
            "best_mean_psnr_db": float(means[best_index]),
            "best_delta_psnr_db": float(means[best_index] - base_mean),
            "curve": [float(value - base_mean) for value in means],
        }
    result["conditions"] = condition_results
    pilot.write_json_atomic(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
