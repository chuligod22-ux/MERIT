"""Create the compact C012 release summary from verified artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
C012 = HERE.parent
MODIFIED = C012.parent
C011_SCRIPTS = MODIFIED / "c011_field_validation" / "scripts"
if str(C011_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C011_SCRIPTS))

import c011_common as common


def main() -> None:
    paths = {
        "protocol": C012 / "protocol" / "holdout_freeze_v1.json",
        "development": C012 / "runs" / "development_v2" / "summary.json",
        "holdout": C012 / "runs" / "holdout_v1" / "summary.json",
        "audit": C012 / "runs" / "c012_audit.json",
        "inference": HERE / "c012_infer.py",
    }
    values = {name: common.load_json(path) for name, path in paths.items() if path.suffix == ".json"}
    if values["protocol"]["status"] != "FROZEN_BEFORE_HOLDOUT_OUTPUTS":
        raise RuntimeError("Protocol is not frozen")
    if values["development"]["global_gate_pass"] is not True:
        raise RuntimeError("Development gate is not PASS")
    if values["holdout"]["global_gate_pass"] is not True:
        raise RuntimeError("Holdout gate is not PASS")
    if values["audit"]["status"] != "PASS":
        raise RuntimeError("Integrity audit is not PASS")
    holdout = values["holdout"]
    result = {
        "schema_version": "MERIT-C012-FINAL-v1",
        "status": "PASS_COMPLETE_INTERNAL_HOLDOUT",
        "scientific_status": "USABLE_INTERNAL_FIELD_PACKAGE_PROSPECTIVE_EXTERNAL_CONFIRMATION_REQUIRED",
        "profiles": holdout["profiles"],
        "holdout": {
            "delta_auprc": holdout["d2"]["panel"]["delta_auprc"],
            "delta_f1": holdout["d2"]["panel"]["delta_f1"],
            "delta_matched_fpr": holdout["d2"]["panel"]["delta_matched_fpr"],
            "delta_absolute_error_mm": holdout["width"]["delta_absolute_error_mm"],
            "relative_error_reduction_percent": holdout["width"]["relative_error_reduction_percent"],
            "improved_nominal_widths": holdout["width"]["improved_nominal_widths"],
            "global_gate_pass": holdout["global_gate_pass"],
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": common.sha256_file(path)}
            for name, path in paths.items()
        },
        "claim_boundary": "Detector/task-conditioned support on a sealed internal split of existing C011 assets. New prospective external field acquisition and traceable natural-crack physical-width metrology remain required.",
    }
    result["payload_sha256"] = common.payload_sha256(result)
    output = C012 / "runs" / "c012_final_summary.json"
    common.write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
