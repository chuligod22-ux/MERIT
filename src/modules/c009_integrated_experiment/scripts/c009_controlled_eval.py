"""Run one frozen C009 checkpoint on D0 controlled and WP4 OOD endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import lpips
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
DEFAULT_OUTPUT = C009 / "evaluation" / "controlled"


def load_holdout(contract: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, str]]]:
    bank_path = common.verify_asset(contract["assets"]["d0_bank"], "D0 bank")
    manifest_path = common.verify_asset(
        contract["assets"]["d0_manifest"], "D0 manifest"
    )
    bank = np.load(bank_path, allow_pickle=False)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = [
        int(row["bank_index"])
        for row in rows
        if row["split"] == "synthetic_holdout"
    ]
    holdout_rows = [rows[index] for index in indices]
    if len(indices) != 50:
        raise common.EvaluationError(f"Expected 50 holdout rows, found {len(indices)}")
    return bank[np.asarray(indices)].astype(np.float32) / np.float32(255.0), holdout_rows


def deterministic_rng(seed: int, bank_index: int, condition_index: int) -> np.random.RandomState:
    mixed = (seed * 1_000_003 + bank_index * 1009 + condition_index * 9176) % (2**32 - 1)
    return np.random.RandomState(mixed)


def id_degrade(
    clean: np.ndarray,
    condition: dict[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    degraded = gaussian_filter(
        clean,
        sigma=(float(condition["sigma_v_px"]), float(condition["sigma_h_px"])),
        mode="reflect",
    ).astype(np.float32)
    degraded += rng.randn(*clean.shape).astype(np.float32) * np.float32(
        float(condition["noise_std_8bit"]) / 255.0
    )
    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def motion_kernel(length: int, angle: float) -> np.ndarray:
    size = int(length) if int(length) % 2 == 1 else int(length) + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size), flags=cv2.INTER_LINEAR)
    total = float(kernel.sum())
    if total <= 0:
        raise common.EvaluationError("Invalid motion kernel")
    return kernel / total


def ood_degrade(
    clean: np.ndarray,
    family: str,
    parameters: dict[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    if family == "poisson_gaussian":
        peak = float(parameters["peak_photons"])
        output = rng.poisson(np.clip(clean, 0.0, 1.0) * peak).astype(np.float32) / peak
        output += rng.randn(*clean.shape).astype(np.float32) * np.float32(
            float(parameters["additive_std_8bit"]) / 255.0
        )
    elif family == "linear_motion_psf":
        kernel = motion_kernel(int(parameters["length_px"]), float(parameters["angle_deg"]))
        output = cv2.filter2D(clean, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    elif family == "spatially_varying_blur":
        left = gaussian_filter(clean, float(parameters["sigma_left_px"]), mode="reflect")
        right = gaussian_filter(clean, float(parameters["sigma_right_px"]), mode="reflect")
        weight = np.linspace(0.0, 1.0, clean.shape[1], dtype=np.float32)[None, :]
        output = left * (1.0 - weight) + right * weight
    elif family == "saturation_clipping":
        output = np.minimum(clean * float(parameters["gain"]), float(parameters["clip_high"]))
        output /= float(parameters["clip_high"])
    elif family == "illumination_gradient_vignetting":
        height, width = clean.shape
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        edge_gain = float(parameters["edge_gain"])
        vertical_gain = float(parameters["vertical_gain"])
        gain = (1.0 - (1.0 - edge_gain) * x**2) * (1.0 - (1.0 - vertical_gain) * y)
        output = clean * gain
    elif family == "veiling_scattering":
        output = clean * float(parameters["contrast"]) + float(parameters["veil"])
    else:
        raise common.EvaluationError(f"Unknown OOD family: {family}")
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def gradient_mae(clean: np.ndarray, candidate: np.ndarray) -> float:
    clean_dx = np.diff(clean, axis=1)
    clean_dy = np.diff(clean, axis=0)
    candidate_dx = np.diff(candidate, axis=1)
    candidate_dy = np.diff(candidate, axis=0)
    return float(
        0.5 * (np.mean(np.abs(clean_dx - candidate_dx)) + np.mean(np.abs(clean_dy - candidate_dy)))
    )


def edge_normal_overshoot_p95(clean: np.ndarray, candidate: np.ndarray) -> float:
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 1e-6]
    if positive.size == 0:
        return 0.0
    threshold = float(np.percentile(positive, 75.0))
    mask = magnitude >= threshold
    yy, xx = np.indices(clean.shape, dtype=np.float32)
    nx = gx / np.maximum(magnitude, 1e-6)
    ny = gy / np.maximum(magnitude, 1e-6)
    distance = np.float32(1.5)
    plus_x, plus_y = xx + nx * distance, yy + ny * distance
    minus_x, minus_y = xx - nx * distance, yy - ny * distance
    clean_plus = cv2.remap(clean, plus_x, plus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    clean_minus = cv2.remap(clean, minus_x, minus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    cand_plus = cv2.remap(candidate, plus_x, plus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    cand_minus = cv2.remap(candidate, minus_x, minus_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    lower = np.minimum(clean_plus, clean_minus)
    upper = np.maximum(clean_plus, clean_minus)
    overshoot = np.maximum.reduce(
        [cand_plus - upper, lower - cand_plus, cand_minus - upper, lower - cand_minus, np.zeros_like(clean)]
    )
    return float(np.percentile(overshoot[mask], 95.0))


def psnr(clean: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((clean.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))


def iter_main_samples(
    contract: dict[str, Any], bank: np.ndarray, rows: list[dict[str, str]]
) -> Iterator[dict[str, Any]]:
    conditions = contract["controlled_d0"]["in_distribution_conditions"]
    seeds = contract["controlled_d0"]["noise_seeds"]
    for row_index, (clean, row) in enumerate(zip(bank, rows)):
        bank_index = int(row["bank_index"])
        for condition_index, condition in enumerate(conditions):
            for seed in seeds:
                rng = deterministic_rng(int(seed), bank_index, condition_index)
                degraded = id_degrade(clean, condition, rng)
                yield {
                    "sample_id": f"id__b{bank_index}__{condition['id']}__n{seed}",
                    "domain": "id",
                    "bank_index": bank_index,
                    "parent_sha256": row["parent_sha256"],
                    "condition": condition["id"],
                    "stochastic_seed": int(seed),
                    "clean": clean,
                    "degraded": degraded,
                    "p_b": (float(condition["sigma_h_px"]) + float(condition["sigma_v_px"])) / 6.0,
                    "p_n": float(condition["noise_std_8bit"]) / 3.0,
                }
        yield {
            "sample_id": f"identity__b{bank_index}",
            "domain": "identity",
            "bank_index": bank_index,
            "parent_sha256": row["parent_sha256"],
            "condition": "identity",
            "stochastic_seed": -1,
            "clean": clean,
            "degraded": clean.copy(),
            "p_b": 0.0,
            "p_n": 0.0,
        }


def iter_ood_samples(
    contract: dict[str, Any], bank: np.ndarray, rows: list[dict[str, str]]
) -> Iterator[dict[str, Any]]:
    seeds = contract["wp4_ood"]["stochastic_seeds"]
    for row_index, (clean, row) in enumerate(zip(bank, rows)):
        bank_index = int(row["bank_index"])
        for family_index, (family, severities) in enumerate(contract["wp4_ood"]["families"].items()):
            for severity_index, parameters in enumerate(severities, start=1):
                for seed in seeds:
                    rng = deterministic_rng(
                        int(seed), bank_index, 100 + family_index * 10 + severity_index
                    )
                    degraded = ood_degrade(clean, family, parameters, rng)
                    yield {
                        "sample_id": f"ood__b{bank_index}__{family}__s{severity_index}__n{seed}",
                        "domain": "ood",
                        "bank_index": bank_index,
                        "parent_sha256": row["parent_sha256"],
                        "condition": family,
                        "severity": severity_index,
                        "stochastic_seed": int(seed),
                        "clean": clean,
                        "degraded": degraded,
                        "p_b": 0.0,
                        "p_n": 0.0,
                    }


def prompt_mode(specification: dict[str, Any], sample: dict[str, Any]) -> tuple[float, float]:
    if sample["domain"] in {"identity", "ood"}:
        return 0.0, 0.0
    if specification["kind"] == "wp1":
        return (
            (sample["p_b"], sample["p_n"])
            if specification["arm"] == "merit_full"
            else (0.0, 0.0)
        )
    if specification.get("wp") == "wp2":
        return 0.0, 0.0
    return sample["p_b"], sample["p_n"]


def batched(values: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def metrics_for_batch(
    samples: list[dict[str, Any]],
    preclip: np.ndarray,
    restored: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    clean = np.stack([sample["clean"] for sample in samples]).astype(np.float32)
    clean_tensor = torch.from_numpy(clean[:, None]).to(device)
    restored_tensor = torch.from_numpy(restored[:, None]).to(device)
    clean_rgb = clean_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
    restored_rgb = restored_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.inference_mode():
        lpips_values = (
            lpips_model(clean_rgb, restored_rgb, normalize=False)
            .flatten()
            .float()
            .cpu()
            .numpy()
        )
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        target = clean[index]
        source = sample["degraded"]
        output = restored[index]
        raw_output = preclip[index]
        records.append(
            {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "bank_index": sample["bank_index"],
                "parent_sha256": sample["parent_sha256"],
                "condition": sample["condition"],
                "severity": sample.get("severity"),
                "stochastic_seed": sample["stochastic_seed"],
                "prompt_blur": sample["eval_p_b"],
                "prompt_noise": sample["eval_p_n"],
                "degraded_psnr_db": psnr(target, source),
                "restored_psnr_db": psnr(target, output),
                "restored_lpips_alex": float(lpips_values[index]),
                "restored_gradient_mae": gradient_mae(target, output),
                "degraded_gradient_mae": gradient_mae(target, source),
                "restored_edge_normal_overshoot_p95": edge_normal_overshoot_p95(target, output),
                "degraded_edge_normal_overshoot_p95": edge_normal_overshoot_p95(target, source),
                "preclip_out_of_range_fraction": float(np.mean((raw_output < 0.0) | (raw_output > 1.0))),
                "mean_absolute_output_change": float(np.mean(np.abs(output - source))),
                "p99_absolute_output_change": float(np.percentile(np.abs(output - source), 99.0)),
                "input_saturation_fraction": float(np.mean((source <= 0.0) | (source >= 1.0))),
                "output_saturation_fraction": float(np.mean((output <= 0.0) | (output >= 1.0))),
            }
        )
    return records


def run(specification: dict[str, Any], output_root: Path, include_ood: bool) -> dict[str, Any]:
    contract = common.load_contract()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise common.EvaluationError("CUDA is required")
    model_name = common.model_id(specification)
    output_dir = output_root / model_name
    complete_path = output_dir / "RUN_COMPLETE.json"
    if complete_path.exists():
        result = common.load_json(complete_path)
        result["status"] = "ALREADY_COMPLETE"
        return result
    output_dir.mkdir(parents=True, exist_ok=True)
    model, provenance = common.build_and_load_model(specification, device)
    bank, rows = load_holdout(contract)
    samples = list(iter_main_samples(contract, bank, rows))
    if include_ood and specification["kind"] == "wp1":
        samples.extend(iter_ood_samples(contract, bank, rows))
    family = specification.get("family", "nafnet")
    batch_sizes = {"nafnet": 16, "restormer": 8, "mprnet": 8}
    batch_size = batch_sizes[family]
    torch.cuda.reset_peak_memory_stats(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    rows_output: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in batched(samples, batch_size):
        degraded = np.stack([sample["degraded"] for sample in batch]).astype(np.float32)
        prompts = [prompt_mode(specification, sample) for sample in batch]
        for sample, (p_b, p_n) in zip(batch, prompts):
            sample["eval_p_b"] = float(p_b)
            sample["eval_p_n"] = float(p_n)
        inputs = common.prompt_tensor(
            degraded,
            [value[0] for value in prompts],
            [value[1] for value in prompts],
        )
        input_tensor = torch.from_numpy(inputs).to(device)
        with torch.inference_mode(), common.autocast(device):
            preclip_tensor = model.forward_preclip(input_tensor)
        preclip = preclip_tensor.float().cpu().numpy()[:, 0]
        if not np.all(np.isfinite(preclip)):
            raise common.EvaluationError(f"Non-finite model output: {model_name}")
        restored = np.clip(preclip, 0.0, 1.0).astype(np.float32)
        rows_output.extend(
            metrics_for_batch(batch, preclip, restored, lpips_model, device)
        )
    elapsed = time.perf_counter() - started
    output_jsonl = output_dir / "sample_metrics.jsonl"
    fd, temporary = tempfile.mkstemp(dir=output_dir, prefix=".metrics.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in rows_output:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, output_jsonl)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    expected_main = 50 * (3 * 3 + 1)
    expected_ood = 50 * 6 * 3 * 3 if include_ood and specification["kind"] == "wp1" else 0
    if len(rows_output) != expected_main + expected_ood:
        raise common.EvaluationError(
            f"Coverage mismatch: {len(rows_output)} != {expected_main + expected_ood}"
        )
    marker: dict[str, Any] = {
        "schema_version": "MERIT-C009-CONTROLLED-RUN-COMPLETE-v1",
        "status": "PASS",
        "model_id": model_name,
        "specification": specification,
        "provenance": provenance,
        "contract_sha256": common.EXPECTED_CONTRACT_SHA256,
        "records": len(rows_output),
        "main_records": expected_main,
        "ood_records": expected_ood,
        "sample_metrics_path": str(output_jsonl.resolve()),
        "sample_metrics_sha256": common.sha256_file(output_jsonl),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    marker["payload_sha256"] = __import__("hashlib").sha256(common.canonical_bytes(marker)).hexdigest()
    common.write_json_atomic(complete_path, marker)
    return marker


def find_specification(model_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    specifications = common.wp1_specifications(contract) + common.wp23_specifications(contract)
    matches = [spec for spec in specifications if common.model_id(spec) == model_name]
    if len(matches) != 1:
        raise common.EvaluationError(f"Unknown or duplicate model id: {model_name}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-ood", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = common.load_contract()
    specification = find_specification(args.model_id, contract)
    result = run(specification, args.output_root.resolve(), not args.skip_ood)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
