"""Freeze the C018 confirmation inputs, code, candidate, and gates."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402


OUTPUT = common.C018 / "protocol" / "confirmatory_freeze_v1.json"
DEVELOPMENT_RESULT = (
    common.C018 / "runs" / "development_v1" / "c009_development_evaluation.json"
)
PHYSICS_SCAN = common.C018 / "runs" / "development_v1" / "physics_scan.json"
POISSON_CHECKPOINT = (
    common.C018
    / "runs"
    / "development_v1"
    / "poisson_nafnet_seed0"
    / "final.pt"
)
POISSON_FUSION = common.C018 / "runs" / "development_v1" / "poisson_mild_fusion_scan.json"
BANK_ROOT = common.C010 / "artifacts" / "confirmatory_v1"
EVALUATOR = HERE / "c018_evaluate_confirmation.py"


def record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Freeze already exists: {OUTPUT}")
    development = common.load_json(DEVELOPMENT_RESULT)
    if development.get("status") != "PASS" or not development.get(
        "broad_six_family_claim_pass"
    ):
        raise RuntimeError("Development must pass before confirmation is frozen")
    c009_contract = common.load_json(
        common.C009 / "protocol" / "full_evaluation_contract_v1.json"
    )
    files = {
        "development_protocol": record(common.DEVELOPMENT_PROTOCOL),
        "development_result": record(DEVELOPMENT_RESULT),
        "physics_scan": record(PHYSICS_SCAN),
        "poisson_checkpoint": record(POISSON_CHECKPOINT),
        "poisson_mild_fusion": record(POISSON_FUSION),
        "bank": record(BANK_ROOT / "d0_bank_448.npy"),
        "manifest": record(BANK_ROOT / "d0_bank_manifest.csv"),
        "evaluator": record(EVALUATOR),
        "common": record(HERE / "c018_common.py"),
        "development_helper": record(HERE / "c018_evaluate_development.py"),
    }
    payload = {
        "schema_version": "MERIT-C018-CONFIRMATORY-FREEZE-v1",
        "status": "FROZEN_BEFORE_ANY_C018_CONFIRMATORY_OUTPUT",
        "frozen_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "description": "C010 parent-disjoint 98-crop, 68-parent source cohort with OOD draws never evaluated during C018 development",
            "crops": 98,
            "parents": 68,
            "relationship_to_c009": "parent SHA-256 overlap 0 as recorded by the existing C010 freeze"
        },
        "stochastic_seeds": [202610150, 202610151, 202610152],
        "families": c009_contract["wp4_ood"]["families"],
        "comparison": "Frozen C018 candidate versus the original C009 NAFNet base seed 0 on matched samples",
        "identity": "Exact source bypass; mean and p99 absolute change must equal zero",
        "gates": {
            "delta_psnr_mean_gte_db": 0.1,
            "delta_psnr_ci95_low_gt_db": 0.0,
            "delta_lpips_mean_lte": -0.005,
            "delta_lpips_ci95_high_lt": 0.0,
            "delta_gradient_ci95_high_lte": 0.0,
            "delta_overshoot_ci95_high_lte": 0.01,
            "family_success": "severity 1 and severity 2 both pass",
            "broad_success": "all six families pass and at least four of six severity-3 cells pass",
            "bootstrap_replicates": 10000,
            "cluster": "physical parent SHA-256"
        },
        "files": files,
        "claim_boundary": "Retrospective parent-disjoint synthetic OOD confirmation. It does not establish blind degradation estimation, new acquisition, physical-width metrology, or cross-site field restoration accuracy."
    }
    common.write_json_atomic(OUTPUT, payload)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT.resolve())}))


if __name__ == "__main__":
    main()
