"""Freeze the one-shot C049 protocol before any C040 model inference."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import c049_common as common


def asset(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def main() -> None:
    if common.PROTOCOL.exists() or common.OUTPUT.exists():
        raise common.C049Error("C049 protocol/output already exists")
    development_protocol = common.c048_common.load_protocol()
    development_summary_path = common.C048 / "runs" / "development_v1" / "development_summary.json"
    development_summary = json.loads(development_summary_path.read_text(encoding="utf-8"))
    if development_summary.get("status") != "PASS" or not development_summary.get(
        "confirmation_authorized"
    ):
        raise common.C049Error("C048 did not authorize confirmation")
    bank_freeze_path = common.C040 / "protocol" / "bank_freeze_v1.json"
    bank_freeze = json.loads(bank_freeze_path.read_text(encoding="utf-8"))
    if bank_freeze.get("status") != "FROZEN_BEFORE_ANY_C041_RESTORMER_MODEL_OUTPUT":
        raise common.C049Error("C040 bank is not frozen")
    for key in ("bank", "manifest"):
        item = bank_freeze[key]
        if common.sha256_file(Path(item["path"])) != item["sha256"]:
            raise common.C049Error(f"C040 {key} hash mismatch")
    source_paths = [
        Path(__file__).resolve(),
        common.HERE / "c049_common.py",
        common.HERE / "c049_confirm.py",
        common.C048 / "scripts" / "c048_common.py",
        common.C048 / "scripts" / "c048_evaluate.py",
        common.C048 / "protocol" / "development_protocol_v1.json",
        development_summary_path,
        bank_freeze_path,
        Path(bank_freeze["bank"]["path"]),
        Path(bank_freeze["manifest"]["path"]),
        common.REVISION / "c046_restormer_constrained_perceptual_adapter" / "runs" / "training_v1" / "penalty10" / "final.pt",
    ]
    protocol = {
        "schema_version": "MERIT-C049-RESTORMER-INDEPENDENT-CONFIRMATION-PROTOCOL-v1",
        "status": "FROZEN_BEFORE_ANY_C040_MODEL_INFERENCE",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "objective": "One-shot independent confirmation of Restormer PSNR non-inferiority across low, mid, and high degradation.",
        "scope": "PSNR is the primary endpoint. LPIPS and structural metrics are prespecified diagnostics.",
        "bank": {"bank": bank_freeze["bank"], "manifest": bank_freeze["manifest"]},
        "bank_payload_sha256": bank_freeze["payload_sha256"],
        "bank_crops": bank_freeze["crops"],
        "bank_parents": bank_freeze["parents"],
        "bank_prior_parent_overlap": bank_freeze["prior_parent_overlap"],
        "conditions": development_protocol["conditions"],
        "routes": development_protocol["routes"],
        "model_seeds": [0, 1, 2],
        "degradation_seeds": [202610490],
        "gate": {
            "each_condition_each_model_seed_delta_psnr_mean_gt": 0.0,
            "each_condition_parent_bootstrap_delta_psnr_ci_lower_gt": 0.0,
            "identity_max_abs_error_eq": 0.0,
        },
        "bootstrap_seed": 202610491,
        "bootstrap_replicates": 5000,
        "one_shot_rule": "No route, threshold, model seed, or degradation seed may change after this freeze. Report PASS or FAIL without repair on C040.",
        "development_protocol_payload_sha256": development_protocol["payload_sha256"],
        "development_summary_sha256": common.sha256_file(development_summary_path),
        "source_assets": [asset(path) for path in source_paths],
    }
    protocol["payload_sha256"] = hashlib.sha256(
        common.canonical_bytes(protocol)
    ).hexdigest()
    common.write_json(common.PROTOCOL, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
