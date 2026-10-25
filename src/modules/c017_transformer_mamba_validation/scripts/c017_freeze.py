#!/usr/bin/env python
"""Create the immutable C017 training and evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import transformers

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
MODIFIED = C017.parent
C016 = MODIFIED / "c016_merit_v24_development"
PROTOCOL = C017 / "protocol" / "c017_protocol_v1.json"
D2_ROOT = Path(r"<LOCAL_PATH_REDACTED>")


class FreezeError(RuntimeError):
    """Raised when the prospective C017 contract cannot be frozen."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FreezeError(f"Missing frozen asset: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_protocol() -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import c017_evaluate as evaluation
    import c017_models as models

    cv_protocol = MODIFIED / "c005_sensitivity" / "d2_tunnel_cv" / "protocol.json"
    d2_protocol = MODIFIED / "c005_freeze" / "d2_detector" / "protocol.json"
    cv_value = json.loads(cv_protocol.read_text(encoding="utf-8-sig"))
    asset_dir = C017 / "assets" / "nvidia_segformer_b0_ade512"
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C017-PROTOCOL-v1",
        "frozen": True,
        "status": "FROZEN_BEFORE_TRAINING_AND_MERIT_OUTPUTS",
        "objective": (
            "Test whether MERIT improves crack detection for a transformer and a Mamba-family "
            "detector under the same four-tunnel leave-one-tunnel-out protocol."
        ),
        "models": list(models.MODEL_NAMES),
        "model_contracts": {
            "segformer_b0": {
                "architecture": "NVIDIA SegFormer-B0 ADE20K pretrained encoder/decoder with a newly initialized one-logit crack classifier",
                "pretraining": "nvidia/segformer-b0-finetuned-ade-512-512 local immutable snapshot",
                "input": "512x512 ImageNet-normalized grayscale repeated to RGB",
                "output": "one 512x512 binary crack logit",
            },
            "mamba2_unet": {
                "architecture": "Mamba2 U-Net hybrid: convolutional 32/48/64/96/128 encoder, two official Transformers Mamba2 blocks at 32x32, symmetric convolutional decoder",
                "pretraining": "none",
                "input": "512x512 ImageNet-normalized grayscale repeated to RGB",
                "output": "one 512x512 binary crack logit",
                "naming_boundary": "This is a purpose-built Mamba2 U-Net hybrid, not VM-UNet or CrackMamba.",
            },
        },
        "data": {
            "d2_root": str(D2_ROOT),
            "tile_size": 512,
            "base_tile_count": 7060,
            "parent_count": 14,
            "tunnels": cv_value["tunnels"],
            "outer_split": "leave one whole tunnel out",
            "field_evaluation_cohort": "1703 frozen tiles from the same 14 parents and four supplied tunnels",
        },
        "training": {
            "seeds": list(cv_value["seed_list"]),
            "epochs": 12,
            "workers": 4,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "loss": "BCEWithLogits plus soft Dice, equal weights, no pos_weight",
            "augmentation": "deterministic 0/90/180/270 degree rotation only",
            "precision": "CUDA bfloat16 autocast with float32 loss",
            "checkpoint": "fixed final epoch; no held-out checkpoint selection or early stopping",
            "resume": "epoch-specific shuffle seeds and atomic epoch checkpoints",
            "models": {
                "segformer_b0": {
                    "physical_batch_size": 64,
                    "gradient_accumulation": 1,
                    "effective_batch_size": 64,
                },
                "mamba2_unet": {
                    "physical_batch_size": 32,
                    "gradient_accumulation": 2,
                    "effective_batch_size": 64,
                },
            },
        },
        "evaluation": {
            "candidate_conditions": evaluation.CONDITIONS,
            "selection_data": "restorer/model seed 0 across all four development tunnels",
            "held_replication_data": "restorer/model seeds 1 and 2, inaccessible to route selection",
            "selection_objective": "maximum equal-model equal-tunnel new-model-panel delta AUPRC",
            "selection_constraints": (
                "mean matched-sensitivity delta FPR <=0; every tunnel panel delta AUPRC >0; "
                "each model mean delta AUPRC >=0; active routes positive on at least 2/4 tunnels"
            ),
            "primary_analysis": "equal average of SegFormer-B0 and Mamba2 U-Net hybrid",
            "robustness_analysis": "equal average of the frozen CNN panel plus the two new models",
            "final_gate": (
                "For both all seeds and held seeds 1/2: positive 95% bootstrap lower bound, "
                "positive mean in 4/4 tunnels, positive F1 mean, nonworsening matched-sensitivity "
                "FPR, and nonnegative mean delta AUPRC for every included detector."
            ),
            "practical_target_delta_auprc": 0.01,
            "practical_target_role": "predeclared substantive-effect indicator, reported separately from the efficacy gate",
            "bootstrap": "10,000 parent-cluster resamples stratified by tunnel",
        },
        "assets": {
            "cv_protocol": record(cv_protocol),
            "d2_protocol": record(d2_protocol),
            "field_cohort": record(
                MODIFIED / "c007_d2_external_synthetic" / "artifacts" / "external_v1" / "cohort.csv"
            ),
            "c016_cycle5_protocol": record(C016 / "protocol" / "v24_detection_cycle5_protocol_v1.json"),
            "c016_cycle5_selection": record(C016 / "protocol" / "v24_detection_cycle5_selection_v1.json"),
            "c016_cycle5_summary": record(C016 / "runs" / "detection_cycle5" / "summary.json"),
            "segformer_pretrained_dir": str(asset_dir.resolve()),
            "segformer_config": record(asset_dir / "config.json"),
            "segformer_weights": record(asset_dir / "model.safetensors"),
        },
        "code": {
            "models": record(HERE / "c017_models.py"),
            "training": record(HERE / "c017_train.py"),
            "evaluation": record(HERE / "c017_evaluate.py"),
        },
        "source_provenance": {
            "VMamba4CrackSeg_commit": git_commit(C017 / "third_party" / "VMamba4CrackSeg"),
            "state_spaces_mamba_commit": git_commit(C017 / "third_party" / "mamba"),
            "implementation_note": (
                "VMamba4CrackSeg was audited, but its CUDA selective-scan extension is unavailable "
                "in this Windows environment. The tested Mamba model uses the official "
                "Transformers Mamba2 PyTorch implementation."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "claim_boundary": (
            "C017 is iterative development and stochastic replication on the four supplied "
            "tunnels. It measures architecture robustness on those tunnels and does not establish "
            "prospective new-site generalisation."
        ),
    }
    payload["payload_sha256"] = payload_hash(payload)
    return payload


def main() -> int:
    payload = build_protocol()
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if PROTOCOL.exists():
        existing = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
        if existing != payload:
            raise FreezeError("Existing frozen C017 protocol differs from recomputation")
    else:
        with PROTOCOL.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
