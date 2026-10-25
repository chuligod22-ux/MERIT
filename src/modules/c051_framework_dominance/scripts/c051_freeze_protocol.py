"""Freeze C051 code and asset identities before the first official training run."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

import c051_common as common


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise common.C051Error(f"Missing freeze asset: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def execute() -> dict[str, Any]:
    protocol = common.load_json(common.PROTOCOL)
    if protocol.get("status") not in {
        "DRAFT_PENDING_CODE_HASH_FREEZE",
        "FROZEN_BEFORE_C051_TRAINING_OR_EVALUATION",
    }:
        raise common.C051Error(f"Unexpected protocol status: {protocol.get('status')}")
    manifest_path = common.ASSETS / "assets_manifest.json"
    manifest = common.load_json(manifest_path)
    if common.payload_sha256(manifest) != manifest.get("payload_sha256"):
        raise common.C051Error("Asset manifest payload mismatch")
    script_names = (
        "c051_common.py",
        "c051_nafnet.py",
        "c051_backbones.py",
        "c051_dataset.py",
        "c051_prepare.py",
        "c051_train.py",
        "c051_queue.py",
        "c051_evaluate.py",
        "c051_evaluation_queue.py",
        "c051_analyze.py",
        "c051_audit.py",
        "c051_freeze_protocol.py",
        "test_c051.py",
    )
    protocol["assets"] = {
        "training_assets_manifest": record(manifest_path),
        "field_cohort": record(
            common.C007_EXTERNAL / "artifacts" / "external_v1" / "cohort.csv"
        ),
        "field_protocol": record(common.C007_EXTERNAL / "protocol.json"),
        "multidetector_protocol": record(common.C007_MULTI / "protocol.json"),
        "legacy_task_segmenter": record(
            common.NRIQA
            / "03_src"
            / "crack_seg"
            / "models"
            / "model_unet_vgg16_best.pt"
        ),
    }
    protocol["code"] = {
        name: record(Path(__file__).resolve().parent / name) for name in script_names
    }
    protocol["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    protocol["status"] = "FROZEN_BEFORE_C051_TRAINING_OR_EVALUATION"
    protocol["payload_sha256"] = common.payload_sha256(protocol)
    common.write_json_atomic(common.PROTOCOL, protocol)
    return protocol


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
