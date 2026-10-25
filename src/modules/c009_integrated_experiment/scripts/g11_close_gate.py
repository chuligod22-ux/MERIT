"""Independently close the C009 G11 implementation gate from frozen artifacts.

This verifier never imports an upstream model and never starts training.  It
checks source identity, paired-arm smoke evidence, and disclosed FLOP coverage
against the active v3 manifest, then writes a machine-readable closure record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
PROTOCOL = C009_ROOT / "protocol"
SMOKE = C009_ROOT / "smoke"

DEFAULT_MANIFEST = PROTOCOL / "g11_implementation_manifest_v3.json"
DEFAULT_PREFLIGHT = PROTOCOL / "g11_source_preflight_v3.json"
DEFAULT_CONTRACT = PROTOCOL / "g11_training_contract_v2.json"
DEFAULT_STREAM = PROTOCOL / "g11_training_stream_freeze_v2.json"
DEFAULT_OUTPUT = PROTOCOL / "g11_closure_v1.json"

FAMILIES = ("nafnet", "restormer", "mprnet")
EXPECTED_PARAMETERS = {
    "nafnet": 17_111_907,
    "restormer": 26_126_644,
    "mprnet": 20_127_073,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, observed: Any
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
        }
    )


def verify_family(
    family: str,
    manifest_sha: str,
    preflight_sha: str,
    contract_sha: str,
    smoke_runner_sha: str,
    flop_runner_sha: str,
) -> dict[str, Any]:
    smoke_path = SMOKE / f"g11_official_{family}_smoke_v3.json"
    flop_path = SMOKE / f"g11_{family}_torch_flops_v1.json"
    smoke = load_json(smoke_path)
    flop = load_json(flop_path)
    checks: list[dict[str, Any]] = []

    add_check(checks, "smoke_schema_v3", smoke.get("schema_version") == "MERIT-C009-G11-OFFICIAL-SMOKE-v3", smoke.get("schema_version"))
    add_check(checks, "manifest_hash_match", smoke.get("manifest_sha256") == manifest_sha, smoke.get("manifest_sha256"))
    add_check(checks, "preflight_hash_match", smoke.get("preflight_sha256") == preflight_sha, smoke.get("preflight_sha256"))
    add_check(checks, "contract_hash_match", smoke.get("contract_sha256") == contract_sha, smoke.get("contract_sha256"))
    add_check(checks, "smoke_runner_hash_match", smoke.get("runner_sha256") == smoke_runner_sha, smoke.get("runner_sha256"))
    add_check(checks, "paired_initial_state_equal", smoke.get("paired_initial_state_equal") is True, smoke.get("paired_initial_state_equal"))
    add_check(checks, "paired_trainable_parameters_equal", smoke.get("paired_trainable_parameters_equal") is True, smoke.get("paired_trainable_parameters_equal"))

    results = smoke.get("results", [])
    add_check(checks, "two_arms_present", [row.get("arm") for row in results] == ["base", "merit_full"], [row.get("arm") for row in results])
    expected_params = EXPECTED_PARAMETERS[family]
    for row in results:
        arm = str(row.get("arm"))
        add_check(checks, f"{arm}_technical_pass", row.get("status") == "PASS", row.get("status"))
        add_check(checks, f"{arm}_finite", row.get("output_all_finite") is True, row.get("output_all_finite"))
        add_check(checks, f"{arm}_parameter_count", row.get("trainable_parameters") == expected_params, row.get("trainable_parameters"))
        add_check(checks, f"{arm}_timing_window", row.get("warmup_steps", 0) >= 2 and row.get("timed_steps", 0) >= 5, {"warmup": row.get("warmup_steps"), "timed": row.get("timed_steps")})
        out_of_range = row.get("preclip_out_of_range_fraction")
        add_check(checks, f"{arm}_not_fully_clamp_saturated", isinstance(out_of_range, (int, float)) and 0.0 <= out_of_range < 1.0, out_of_range)
        add_check(checks, f"{arm}_preclip_optimization", row.get("optimization_output_domain") == "unclipped selected official output", row.get("optimization_output_domain"))
        add_check(checks, f"{arm}_clamped_task_evaluation", row.get("task_and_evaluation_output_domain") == "clamped to [0,1]", row.get("task_and_evaluation_output_domain"))

    add_check(checks, "flop_manifest_hash_match", flop.get("manifest_sha256") == manifest_sha, flop.get("manifest_sha256"))
    add_check(checks, "flop_preflight_hash_match", flop.get("preflight_sha256") == preflight_sha, flop.get("preflight_sha256"))
    add_check(checks, "flop_runner_hash_match", flop.get("runner_sha256") == flop_runner_sha, flop.get("runner_sha256"))
    add_check(checks, "flop_partial_coverage_disclosed", flop.get("status") == "PARTIAL_SUPPORTED_OPS" and bool(flop.get("coverage_limitation")), flop.get("status"))
    supported_flops = flop.get("supported_operation_flops")
    add_check(checks, "supported_operation_flops_positive", isinstance(supported_flops, int) and supported_flops > 0, supported_flops)

    return {
        "family": family,
        "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "parameters": expected_params,
        "smoke_artifact": str(smoke_path.relative_to(C009_ROOT)).replace("\\", "/"),
        "smoke_sha256": sha256_file(smoke_path),
        "smoke_status": smoke.get("status"),
        "flop_artifact": str(flop_path.relative_to(C009_ROOT)).replace("\\", "/"),
        "flop_sha256": sha256_file(flop_path),
        "supported_operation_gflops": flop.get("supported_operation_gflops"),
        "flop_coverage": "PARTIAL_SUPPORTED_OPS_LOWER_BOUND",
        "checks": checks,
    }


def run_gate(
    manifest_path: Path,
    preflight_path: Path,
    contract_path: Path,
    stream_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    preflight = load_json(preflight_path)
    stream = load_json(stream_path)
    manifest_sha = sha256_file(manifest_path)
    preflight_sha = sha256_file(preflight_path)
    contract_sha = sha256_file(contract_path)
    stream_sha = sha256_file(stream_path)
    smoke_runner_path = HERE / "g11_official_smoke.py"
    flop_runner_path = HERE / "g11_torch_flop_audit.py"
    smoke_runner_sha = sha256_file(smoke_runner_path)
    flop_runner_sha = sha256_file(flop_runner_path)

    global_checks: list[dict[str, Any]] = []
    add_check(global_checks, "active_manifest_v3", manifest.get("schema_version") == "MERIT-C009-G11-IMPLEMENTATION-v3", manifest.get("schema_version"))
    add_check(global_checks, "source_preflight_pass", preflight.get("status") == "PASS", preflight.get("status"))
    add_check(global_checks, "preflight_manifest_hash_match", preflight.get("manifest_sha256") == manifest_sha, preflight.get("manifest_sha256"))
    add_check(global_checks, "training_contract_hash_match", manifest.get("primary_design", {}).get("training", {}).get("contract_sha256") == contract_sha, manifest.get("primary_design", {}).get("training", {}).get("contract_sha256"))
    add_check(global_checks, "paired_stream_pass", stream.get("status") == "PASS", stream.get("status"))
    add_check(global_checks, "paired_stream_hash_match", manifest.get("paired_training_stream_freeze", {}).get("stream_freeze_sha256") == stream_sha, manifest.get("paired_training_stream_freeze", {}).get("stream_freeze_sha256"))
    add_check(global_checks, "smoke_runner_frozen", manifest.get("integration_files", {}).get("smoke_runner", {}).get("sha256") == smoke_runner_sha, smoke_runner_sha)

    family_records = [
        verify_family(
            family,
            manifest_sha,
            preflight_sha,
            contract_sha,
            smoke_runner_sha,
            flop_runner_sha,
        )
        for family in FAMILIES
    ]
    all_pass = all(row["status"] == "PASS" for row in global_checks) and all(
        row["status"] == "PASS" for row in family_records
    )
    result = {
        "schema_version": "MERIT-C009-G11-CLOSURE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "G11",
        "status": "CLOSED_WITH_DISCLOSED_PARTIAL_FLOP_COVERAGE" if all_pass else "BLOCKED",
        "technical_smoke_verdict": "PASS" if all_pass else "FAIL",
        "full_training_allowed": False,
        "authorization_note": "G11 closure does not authorize full training; G13 must close independently.",
        "manifest": {"path": str(manifest_path.relative_to(C009_ROOT)).replace("\\", "/"), "sha256": manifest_sha},
        "preflight": {"path": str(preflight_path.relative_to(C009_ROOT)).replace("\\", "/"), "sha256": preflight_sha},
        "contract": {"path": str(contract_path.relative_to(C009_ROOT)).replace("\\", "/"), "sha256": contract_sha},
        "paired_stream": {"path": str(stream_path.relative_to(C009_ROOT)).replace("\\", "/"), "sha256": stream_sha},
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "global_checks": global_checks,
        "families": family_records,
        "flop_interpretation": "Reported PyTorch eager counts are supported-operation lower bounds with multiply-add counted as two FLOPs. They are not complete total-compute estimates and cannot be used for cross-family efficiency ranking.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_gate(
        args.manifest.resolve(),
        args.preflight.resolve(),
        args.contract.resolve(),
        args.stream.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
