"""Evaluate the frozen WP2 prompt perturbations on NAFNet MERIT-full."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch

import c009_controlled_eval as controlled
import c009_evaluation_common as common


C009 = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = C009 / "evaluation" / "wp2_prompt"


def prompt_variants(
    correct_blur: np.ndarray,
    correct_noise: np.ndarray,
    contract: dict[str, Any],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    settings = contract["wp2_prompt_perturbation"]["conditions"]
    values: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("correct", correct_blur, correct_noise),
        ("shuffled", np.roll(correct_blur, 1), np.roll(correct_noise, 1)),
        ("swapped", correct_noise, correct_blur),
        (
            "constant",
            np.full_like(correct_blur, float(settings["constant"]["p_b"])),
            np.full_like(correct_noise, float(settings["constant"]["p_n"])),
        ),
        ("zero", np.zeros_like(correct_blur), np.zeros_like(correct_noise)),
    ]
    for factor in settings["mismatch_multipliers"]:
        values.append(
            (
                f"mismatch_{factor:g}",
                np.clip(correct_blur * float(factor), 0.0, 1.0),
                np.clip(correct_noise * float(factor), 0.0, 1.0),
            )
        )
    return values


def run(seed: int, output_root: Path) -> dict[str, Any]:
    contract = common.load_contract()
    specification = {
        "kind": "wp1",
        "family": "nafnet",
        "arm": "merit_full",
        "seed": seed,
    }
    model_name = common.model_id(specification)
    output_dir = output_root / model_name
    marker_path = output_dir / "RUN_COMPLETE.json"
    if marker_path.exists():
        marker = common.load_json(marker_path)
        marker["status"] = "ALREADY_COMPLETE"
        return marker
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise common.EvaluationError("CUDA is required")
    model, provenance = common.build_and_load_model(specification, device)
    bank, manifest_rows = controlled.load_holdout(contract)
    samples = [
        sample
        for sample in controlled.iter_main_samples(contract, bank, manifest_rows)
        if sample["domain"] == "id"
    ]
    if len(samples) != 450:
        raise common.EvaluationError(f"Expected 450 ID samples, found {len(samples)}")
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in controlled.batched(samples, 10):
        degraded = np.stack([sample["degraded"] for sample in batch]).astype(np.float32)
        clean = np.stack([sample["clean"] for sample in batch]).astype(np.float32)
        correct_blur = np.asarray([sample["p_b"] for sample in batch], dtype=np.float32)
        correct_noise = np.asarray([sample["p_n"] for sample in batch], dtype=np.float32)
        for condition, blur, noise in prompt_variants(correct_blur, correct_noise, contract):
            inputs = common.prompt_tensor(degraded, blur, noise)
            with torch.inference_mode(), common.autocast(device):
                preclip_tensor = model.forward_preclip(torch.from_numpy(inputs).to(device))
            preclip = preclip_tensor.float().cpu().numpy()[:, 0]
            if not np.all(np.isfinite(preclip)):
                raise common.EvaluationError(f"Non-finite WP2 output: seed={seed} {condition}")
            restored = np.clip(preclip, 0.0, 1.0).astype(np.float32)
            with torch.inference_mode():
                clean_tensor = torch.from_numpy(clean[:, None]).to(device).repeat(1, 3, 1, 1)
                restored_tensor = torch.from_numpy(restored[:, None]).to(device).repeat(1, 3, 1, 1)
                lpips_values = (
                    lpips_model(clean_tensor * 2 - 1, restored_tensor * 2 - 1)
                    .flatten()
                    .float()
                    .cpu()
                    .numpy()
                )
            for index, sample in enumerate(batch):
                records.append(
                    {
                        "model_id": model_name,
                        "sample_id": sample["sample_id"],
                        "bank_index": sample["bank_index"],
                        "parent_sha256": sample["parent_sha256"],
                        "condition": sample["condition"],
                        "stochastic_seed": sample["stochastic_seed"],
                        "prompt_condition": condition,
                        "correct_p_b": float(correct_blur[index]),
                        "correct_p_n": float(correct_noise[index]),
                        "eval_p_b": float(blur[index]),
                        "eval_p_n": float(noise[index]),
                        "prompt_absolute_error": float(
                            abs(float(blur[index]) - float(correct_blur[index]))
                            + abs(float(noise[index]) - float(correct_noise[index]))
                        ),
                        "restored_psnr_db": controlled.psnr(clean[index], restored[index]),
                        "restored_lpips_alex": float(lpips_values[index]),
                        "restored_gradient_mae": controlled.gradient_mae(clean[index], restored[index]),
                        "preclip_out_of_range_fraction": float(
                            np.mean((preclip[index] < 0.0) | (preclip[index] > 1.0))
                        ),
                    }
                )
    expected = 450 * 9
    if len(records) != expected:
        raise common.EvaluationError(f"WP2 coverage mismatch: {len(records)} != {expected}")
    output = output_dir / "prompt_metrics.jsonl"
    fd, temporary = tempfile.mkstemp(dir=output_dir, prefix=".prompt.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    marker: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP2-PROMPT-RUN-COMPLETE-v1",
        "status": "PASS",
        "model_id": model_name,
        "seed": seed,
        "records": len(records),
        "prompt_conditions": 9,
        "provenance": provenance,
        "contract_sha256": common.EXPECTED_CONTRACT_SHA256,
        "output_path": str(output.resolve()),
        "output_sha256": common.sha256_file(output),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    marker["payload_sha256"] = hashlib.sha256(common.canonical_bytes(marker)).hexdigest()
    common.write_json_atomic(marker_path, marker)
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.seed, args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
