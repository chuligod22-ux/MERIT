"""Freeze and verify the paired C009 D0 training stream.

This module does not train a model.  It verifies the immutable C006 bank and
materialises short, deterministic stream fingerprints for the G11 base versus
MERIT-full paired comparison.  Both arms share the exact degraded image,
target, sample order, and degradation draws; only the two prompt channels are
allowed to differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
WORKSPACE_ROOT = C009_ROOT.parents[2]
DEFAULT_CONTRACT = C009_ROOT / "protocol" / "g11_training_contract_v2.json"
DEFAULT_OUTPUT = C009_ROOT / "protocol" / "g11_training_stream_freeze_v2.json"


class ContractError(RuntimeError):
    """Raised when a frozen G11 training invariant is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def verify_source_bank(
    contract: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, str]], dict[str, Any]]:
    source = contract["source_bank"]
    paths = {
        "protocol": resolve_workspace_path(source["protocol_path"]),
        "bank": resolve_workspace_path(source["bank_path"]),
        "manifest": resolve_workspace_path(source["manifest_path"]),
        "ledger": resolve_workspace_path(source["ledger_path"]),
    }
    expected_hashes = {
        "protocol": source["protocol_sha256"],
        "bank": source["bank_sha256"],
        "manifest": source["manifest_sha256"],
    }
    for label, expected in expected_hashes.items():
        actual = sha256_file(paths[label])
        if actual.lower() != str(expected).lower():
            raise ContractError(
                f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
            )

    ledger = load_json(paths["ledger"])
    ledger_payload = dict(ledger)
    recorded_payload_hash = ledger_payload.pop("ledger_payload_sha256")
    actual_payload_hash = sha256_bytes(canonical_json_bytes(ledger_payload))
    if recorded_payload_hash != actual_payload_hash:
        raise ContractError("C006 ledger internal payload hash mismatch")
    if actual_payload_hash != source["ledger_payload_sha256"]:
        raise ContractError("C006 ledger differs from the G11 frozen contract")
    if ledger["bank_content_sha256"] != source["bank_content_sha256"]:
        raise ContractError("C006 bank content hash differs from the G11 contract")

    bank = np.load(paths["bank"], allow_pickle=False)
    with paths["manifest"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if list(bank.shape) != source["shape"] or str(bank.dtype) != source["dtype"]:
        raise ContractError(
            f"Unexpected bank shape/dtype: {bank.shape}/{bank.dtype}"
        )
    if len(rows) != len(bank):
        raise ContractError("Bank and manifest row counts differ")

    split_counts: dict[str, int] = {}
    parent_sets: dict[str, set[str]] = {}
    content_digest = hashlib.sha256()
    for expected_index, (crop, row) in enumerate(zip(bank, rows)):
        if int(row["bank_index"]) != expected_index:
            raise ContractError("Manifest bank_index is not contiguous")
        crop_bytes = np.ascontiguousarray(crop).tobytes(order="C")
        if sha256_bytes(crop_bytes) != row["crop_sha256"]:
            raise ContractError(f"Crop hash mismatch at index {expected_index}")
        content_digest.update(expected_index.to_bytes(8, byteorder="big"))
        content_digest.update(crop_bytes)
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        parent_sets.setdefault(split, set()).add(row["parent_sha256"])
        if row["source_dataset"] != "D0":
            raise ContractError(f"Non-D0 row at index {expected_index}")
    if content_digest.hexdigest() != source["bank_content_sha256"]:
        raise ContractError("Recomputed bank content hash mismatch")
    expected_counts = {
        "train": int(source["train_rows"]),
        "synthetic_holdout": int(source["synthetic_holdout_rows"]),
    }
    if split_counts != expected_counts:
        raise ContractError(
            f"Split counts differ: expected={expected_counts} actual={split_counts}"
        )
    overlap = parent_sets["train"] & parent_sets["synthetic_holdout"]
    if overlap:
        raise ContractError(
            f"Physical-parent leakage across splits: {len(overlap)} hashes"
        )
    all_parents = parent_sets["train"] | parent_sets["synthetic_holdout"]
    if len(all_parents) != int(source["unique_physical_parents"]):
        raise ContractError("Unique physical-parent count mismatch")
    return bank, rows, {
        "split_counts": split_counts,
        "train_unique_parents": len(parent_sets["train"]),
        "holdout_unique_parents": len(parent_sets["synthetic_holdout"]),
        "parent_overlap": 0,
    }


def latent_stream(
    bank: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    batch_size: int,
    degradation: dict[str, list[float]],
) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    rng = np.random.RandomState(seed)
    patch_size = int(bank.shape[-1])
    while True:
        selected = train_indices[
            rng.randint(0, len(train_indices), size=batch_size)
        ]
        degraded_batch: list[np.ndarray] = []
        target_batch: list[np.ndarray] = []
        samples: list[dict[str, float | int]] = []
        for bank_index in selected:
            clean = bank[int(bank_index)].astype(np.float32) / np.float32(255.0)
            sigma_h = float(rng.uniform(*degradation["sigma_h_px"]))
            sigma_v = float(rng.uniform(*degradation["sigma_v_px"]))
            noise_std = float(rng.uniform(*degradation["noise_std_8bit"]))
            degraded = gaussian_filter(
                clean, sigma=(sigma_v, sigma_h), mode="reflect"
            ).astype(np.float32)
            noise = rng.randn(patch_size, patch_size).astype(np.float32)
            degraded = degraded + noise * np.float32(noise_std / 255.0)
            degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)
            degraded_batch.append(degraded)
            target_batch.append(clean[None])
            samples.append(
                {
                    "bank_index": int(bank_index),
                    "sigma_h_px": sigma_h,
                    "sigma_v_px": sigma_v,
                    "noise_std_8bit": noise_std,
                }
            )
        yield (
            np.stack(degraded_batch).astype(np.float32, copy=False),
            np.stack(target_batch).astype(np.float32, copy=False),
            {"samples": samples},
        )


def arm_inputs(
    degraded: np.ndarray,
    metadata: dict[str, Any],
    arm: str,
) -> np.ndarray:
    if degraded.ndim != 3:
        raise ContractError(f"Expected [B,H,W] degraded input, got {degraded.shape}")
    blur_prompt = np.zeros_like(degraded, dtype=np.float32)
    noise_prompt = np.zeros_like(degraded, dtype=np.float32)
    if arm == "merit_full":
        for index, sample in enumerate(metadata["samples"]):
            blur_prompt[index].fill(
                0.5
                * (float(sample["sigma_h_px"]) + float(sample["sigma_v_px"]))
                / 3.0
            )
            noise_prompt[index].fill(float(sample["noise_std_8bit"]) / 3.0)
    elif arm != "base":
        raise ValueError(f"Unknown arm: {arm}")
    return np.stack([degraded, blur_prompt, noise_prompt], axis=1).astype(
        np.float32, copy=False
    )


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def freeze_stream(
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    bank, rows, bank_audit = verify_source_bank(contract)
    train_indices = np.array(
        [int(row["bank_index"]) for row in rows if row["split"] == "train"],
        dtype=np.int64,
    )
    settings = contract["stream"]
    comparison = contract["paired_comparison"]
    batch_size = int(contract["training"]["batch_size"])
    prefix_batches = int(settings["stream_prefix_batches_per_seed"])
    seed_records: list[dict[str, Any]] = []
    for seed in comparison["seeds"]:
        stream = latent_stream(
            bank,
            train_indices,
            int(seed),
            batch_size,
            settings["degradation"],
        )
        batch_records: list[dict[str, Any]] = []
        seed_latent_digest = hashlib.sha256()
        for batch_index in range(prefix_batches):
            degraded, target, metadata = next(stream)
            base = arm_inputs(degraded, metadata, "base")
            full = arm_inputs(degraded, metadata, "merit_full")
            if not np.array_equal(base[:, 0], full[:, 0]):
                raise ContractError("Paired channel-0 mismatch")
            if np.any(base[:, 1:]):
                raise ContractError("Base prompt channels are not zero")
            if not np.all(np.isfinite(full)):
                raise ContractError("Non-finite MERIT-full input")
            if np.min(full[:, 1:]) < 0.0 or np.max(full[:, 1:]) > 1.0:
                raise ContractError("MERIT-full prompt outside [0,1]")
            latent_payload = {
                "batch_index": batch_index,
                "metadata": metadata,
                "degraded_sha256": array_sha256(degraded),
                "target_sha256": array_sha256(target),
            }
            seed_latent_digest.update(canonical_json_bytes(latent_payload))
            batch_records.append(
                {
                    **latent_payload,
                    "base_input_sha256": array_sha256(base),
                    "merit_full_input_sha256": array_sha256(full),
                    "paired_channel0_equal": True,
                    "base_prompts_all_zero": True,
                }
            )
        seed_records.append(
            {
                "seed": int(seed),
                "latent_prefix_sha256": seed_latent_digest.hexdigest(),
                "batches": batch_records,
            }
        )

    output = {
        "schema_version": "MERIT-C009-G11-TRAINING-STREAM-FREEZE-v2",
        "status": "PASS",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "bank_audit": bank_audit,
        "train_indices_sha256": array_sha256(train_indices),
        "batch_size": batch_size,
        "prefix_batches_per_seed": prefix_batches,
        "seed_records": seed_records,
        "interpretation": (
            "The frozen prefixes prove deterministic latent sampling and the "
            "allowed prompt-only input difference. They do not close G11 or "
            "authorize full training."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_stream(args.contract.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
