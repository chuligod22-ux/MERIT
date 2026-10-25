"""Freeze the C050 matched factorial before new NAFNet/MPRNet C040 output."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C050 = HERE.parent
REVISION = C050.parent
PROTOCOL = C050 / "protocol" / "factorial_protocol_v1.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    if PROTOCOL.exists():
        raise RuntimeError(f"Refusing to overwrite frozen protocol: {PROTOCOL}")
    c049_protocol_path = (
        REVISION
        / "c049_restormer_psnr_independent_confirmation"
        / "protocol"
        / "confirmation_protocol_v1.json"
    )
    c049 = json.loads(c049_protocol_path.read_text(encoding="utf-8"))
    source_paths = [
        Path(__file__).resolve(),
        HERE / "c050_factorial.py",
        REVISION
        / "c010_merit_v2_development"
        / "protocol"
        / "confirmatory_freeze_v1.json",
        REVISION
        / "c010_merit_v2_development"
        / "scripts"
        / "c010_confirmatory_eval.py",
        REVISION / "c010_merit_v2_development" / "scripts" / "merit_v2_model.py",
        REVISION
        / "c039_c036_independent_confirmation"
        / "protocol"
        / "confirmation_freeze_v1.json",
        REVISION
        / "c039_c036_independent_confirmation"
        / "scripts"
        / "c039_confirmation.py",
        REVISION
        / "c049_restormer_psnr_independent_confirmation"
        / "protocol"
        / "confirmation_protocol_v1.json",
        REVISION
        / "c049_restormer_psnr_independent_confirmation"
        / "scripts"
        / "c049_common.py",
        REVISION
        / "c048_restormer_psnr_integrated_development"
        / "scripts"
        / "c048_evaluate.py",
    ]
    value: dict[str, Any] = {
        "schema_version": "MERIT-C050-FRAMEWORK-BACKBONE-FACTORIAL-PROTOCOL-v1",
        "status": "FROZEN_BEFORE_C050_CROSS_BACKBONE_OUTPUT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Test whether the final MERIT framework configuration has a larger "
            "matched PSNR-gain effect than restoration-backbone choice."
        ),
        "analysis_exposure": (
            "Retrospective matched factorial. Restormer C040 results existed before "
            "this analysis freeze; NAFNet and MPRNet C040 outputs did not. This is "
            "not a new prospective independent confirmation."
        ),
        "claim_boundary": (
            "Inference is limited to NAFNet, MPRNet, and Restormer under the matched "
            "C040 synthetic low/mid/high degradation protocol. It does not establish "
            "backbone interchangeability, all-architecture dominance, real-field "
            "detection efficacy, or natural physical-width accuracy."
        ),
        "bank": c049["bank"],
        "bank_payload_sha256": c049["bank_payload_sha256"],
        "bank_crops": c049["bank_crops"],
        "bank_parents": c049["bank_parents"],
        "conditions": c049["conditions"],
        "degradation_seeds": c049["degradation_seeds"],
        "model_seeds": c049["model_seeds"],
        "backbones": ["nafnet", "mprnet", "restormer"],
        "arms": ["base", "merit"],
        "primary_endpoint": (
            "PSNR(output, clean) - PSNR(identical degraded source, clean), in dB"
        ),
        "independent_unit": (
            "physical parent; crops, model seeds, and the fixed degradation draw are "
            "averaged within parent before paired bootstrap"
        ),
        "factorial_estimands": {
            "framework_main_effect": "mean over backbones of MERIT minus base",
            "backbone_main_effect": (
                "largest absolute pairwise contrast between backbone marginal means, "
                "averaged over base and MERIT arms"
            ),
            "dominance_margin": (
                "framework main effect minus largest backbone marginal contrast"
            ),
            "factor_sums_of_squares": (
                "framework, backbone, and interaction sums of squares from the 3x2 "
                "cell means"
            ),
        },
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 202610500,
        "success_gates": {
            "each_backbone_condition_parent_ci_lower_gt_db": 0.0,
            "each_backbone_condition_each_model_seed_mean_gt_db": 0.0,
            "framework_main_effect_ci_lower_gt_db": 0.0,
            "framework_minus_max_backbone_ci_lower_gt_db": 0.0,
            "framework_to_backbone_ss_ratio_ci_lower_gt": 1.0,
        },
        "reporting_rule": (
            "Report PASS or FAIL without changing routes, seeds, cohort, estimands, or "
            "gates after C050 output. Preserve interaction and failed-cell results."
        ),
        "source_assets": [asset(path) for path in source_paths],
    }
    value["payload_sha256"] = hashlib.sha256(canonical_bytes(value)).hexdigest()
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
