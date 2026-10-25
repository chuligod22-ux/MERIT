"""Freeze C038 routes before opening the C036 ISO200 confirmation bank."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import c039_common as common


def asset(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "sha256": common.sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    if common.FREEZE.exists():
        raise common.C039Error(f"Refusing to overwrite: {common.FREEZE}")
    if common.OUTPUT_ROOT.exists() and any(common.OUTPUT_ROOT.iterdir()):
        raise common.C039Error("C036 confirmation outputs already exist")
    development = common.c038_common.load_protocol()
    selection_path = common.C038 / "runs" / "development_v1" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    reserve_path = common.C036 / "protocol" / "bank_freeze_v1.json"
    reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
    prior_freeze_path = common.C032 / "protocol" / "confirmation_freeze_v1.json"
    prior_freeze = json.loads(prior_freeze_path.read_text(encoding="utf-8"))
    if selection.get("status") != "PASS" or not selection.get("confirmation_authorized"):
        raise common.C039Error("C038 development did not authorize confirmation")
    if reserve.get("status") != "FROZEN_BEFORE_ANY_C037_MODEL_OUTPUT":
        raise common.C039Error("C036 reserve is not intact")
    high_fixed = development["high_fixed"]
    high_selected = selection["selected"]["id_high"]["route"]
    high_route_b = {
        **high_fixed["route_b"],
        "perceptual_alpha": high_selected["route_b_perceptual_alpha"],
    }
    sources = [
        Path(__file__),
        common.HERE / "c039_common.py",
        common.HERE / "c039_confirmation.py",
        common.C038 / "protocol" / "development_protocol_v1.json",
        selection_path,
        reserve_path,
        Path(reserve["bank"]["path"]),
        Path(reserve["manifest"]["path"]),
        common.C032 / "scripts" / "c030_confirmation.py",
        prior_freeze_path,
        common.C035 / "scripts" / "c035_model.py",
        common.C035 / "scripts" / "c035_evaluate.py",
        common.C035 / "runs" / "training_v1" / high_fixed["gate_variant"] / "final.pt",
        common.C035 / "runs" / "training_v1" / high_fixed["gate_variant"] / "RUN_COMPLETE.json",
    ]
    sources.extend(Path(item["path"]) for item in prior_freeze["source_assets"])
    sources.extend(Path(item["path"]) for item in development["source_assets"])
    value = {
        "schema_version": "MERIT-C039-C036-CONFIRMATION-FREEZE-v1",
        "status": "FROZEN_BEFORE_ANY_C036_MODEL_OUTPUT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_protocol_payload_sha256": development["payload_sha256"],
        "development_selection_sha256": common.sha256_file(selection_path),
        "routes": {
            "id_low": selection["selected"]["id_low"]["route"],
            "id_mid": selection["selected"]["id_mid"]["route"],
            "id_high": {
                "variant": high_fixed["gate_variant"],
                "gate_alpha": high_fixed["gate_alpha"],
                "route_a": high_fixed["route_a"],
                "route_b": high_route_b,
            },
        },
        "low_full_sigma_px": development["low_full_sigma_px"],
        "conditions": prior_freeze["conditions"],
        "confirmation": {
            "bank": reserve["bank"],
            "manifest": reserve["manifest"],
            "bank_freeze_payload_sha256": reserve["payload_sha256"],
            "crops": reserve["crops"],
            "parents": reserve["parents"],
            "model_seeds": [0, 1, 2],
            "degradation_seeds": [202610390, 202610391, 202610392],
            "bootstrap_seed": 2026103999,
            "bootstrap_replicates": 5000,
        },
        "strict_gate": (
            "Each condition and model-seed mean PSNR >0; parent-bootstrap PSNR lower >0; "
            "LPIPS and gradient upper <0; overshoot upper <=0.00005; exact zero-prompt identity."
        ),
        "claim_boundary": reserve["claim_boundary"],
        "source_assets": [asset(path) for path in dict.fromkeys(sources)],
    }
    value["payload_sha256"] = hashlib.sha256(common.c023.canonical_bytes(value)).hexdigest()
    common.c023.write_json(common.FREEZE, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
