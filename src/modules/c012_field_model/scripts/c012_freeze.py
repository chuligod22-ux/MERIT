"""Freeze the C012 internal holdout contract before producing holdout outputs."""

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
import c012_eval as evaluation
import c012_profiles as profiles


OUTPUT = C012 / "protocol" / "holdout_freeze_v1.json"


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite frozen protocol: {OUTPUT}")
    holdout_root = C012 / "runs" / "holdout_v1"
    if holdout_root.exists() and any(holdout_root.rglob("*.json")):
        raise RuntimeError("Holdout outputs already exist; freeze is no longer result-blind")
    development_summary = C012 / "runs" / "development_v2" / "summary.json"
    development = common.load_json(development_summary)
    if development.get("global_gate_pass") is not True:
        raise RuntimeError("Development gate did not pass")
    c011_protocol = common.load_json(common.PROTOCOL_PATH)
    payload = {
        "schema_version": "MERIT-C012-HOLDOUT-FREEZE-v1",
        "status": "FROZEN_BEFORE_HOLDOUT_OUTPUTS",
        "created_utc": common.utc_now(),
        "scientific_boundary": {
            "development": "Tunnels A/B and W001-W025 were used for adaptive candidate selection.",
            "holdout": "Tunnels C/D and W026-W050 are sealed internal holdout assets not used for C012 selection.",
            "claim": "A pass is internal heldout field evidence only; prospective external acquisition remains mandatory.",
            "natural_crack_physical_width": "CANNOT_DETERMINE_WITH_AVAILABLE_ASSETS",
        },
        "splits": {
            stage: {"tunnels": list(value["tunnels"]), "width_sources": list(value["width_sources"])}
            for stage, value in evaluation.SPLITS.items()
        },
        "profiles": {"detectors": profiles.DETECTOR_PROFILE, "width": profiles.WIDTH_PROFILE},
        "aggregation": {
            "d2": "Average seed within detector-parent, detector within parent, then tunnel-stratified parent bootstrap.",
            "width": "Pair RAW/profile within seed-source-nominal, average seeds, then source-cluster bootstrap.",
            "bootstrap_replicates": 10000,
            "bootstrap_seed_base": 20261031,
        },
        "gates": {
            "d2": "panel delta AUPRC mean and CI lower >0; every holdout tunnel positive; matched FPR mean <=0; every detector mean delta AUPRC >=0",
            "width": "delta absolute error mean and CI upper <0; >=6/8 nominal widths improve; non-estimable increase <=2 percentage points",
            "global": "D2 and width gates both pass",
        },
        "code": {
            "profiles": common.asset_record(HERE / "c012_profiles.py"),
            "evaluation": common.asset_record(HERE / "c012_eval.py"),
            "summarizer": common.asset_record(HERE / "c012_summarize.py"),
            "profile_tests": common.asset_record(HERE / "test_c012_profiles.py"),
        },
        "development_summary": common.asset_record(development_summary),
        "upstream_c011_protocol": {
            "path": str(common.PROTOCOL_PATH.resolve()),
            "sha256": common.sha256_file(common.PROTOCOL_PATH),
            "payload_sha256": c011_protocol["payload_sha256"],
        },
    }
    payload["payload_sha256"] = common.payload_sha256(payload)
    common.write_json_exclusive(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "path": str(OUTPUT.resolve()), "payload_sha256": payload["payload_sha256"], "file_sha256": common.sha256_file(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
