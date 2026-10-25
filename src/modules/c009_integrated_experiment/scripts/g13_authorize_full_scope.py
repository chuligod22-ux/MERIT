"""Create the result-blind author-risk G13 full-scope authorization.

This tool does not reinterpret the measured v1 runtime evidence.  It records
the author's explicit decision to accept a larger operational budget while the
journal extension remains pending.  The flag is deliberately verbose so this
artifact cannot be produced accidentally.
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
DEFAULT_INPUT = C009_ROOT / "protocol" / "g13_feasibility_v1.json"
DEFAULT_OUTPUT = C009_ROOT / "protocol" / "g13_feasibility_v2.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def authorize(input_path: Path, output_path: Path) -> dict[str, Any]:
    prior = load_json(input_path)
    if prior.get("status") != "BLOCKED_NO_GO_CURRENT_SCOPE":
        raise RuntimeError("G13 v1 does not contain the expected NO-GO decision")
    lower_bound = float(
        prior["current_scope_budget"][
            "training_plus_restoration_forward_lower_bound_gpu_hours"
        ]
    )
    if abs(lower_bound - 174.0642868741872) > 1e-9:
        raise RuntimeError("G13 v1 lower bound changed unexpectedly")

    output: dict[str, Any] = {
        "schema_version": "MERIT-C009-G13-FEASIBILITY-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "G13",
        "status": "CLOSED_GO_FULL_SCOPE_AT_AUTHOR_RISK",
        "full_training_allowed": True,
        "decision_basis": {
            "author_decision_date": "2026-09-25",
            "author_instruction": "Execute the complete predeclared scope without reduction.",
            "decision_timing": (
                "Recorded before any WP1 full-training performance result or "
                "endpoint result was inspected."
            ),
            "scientific_scope_change": "NONE",
            "operational_change": (
                "The former 144 GPU-hour planning ceiling is superseded by a "
                "240 GPU-hour monitored execution budget."
            ),
        },
        "scope": {
            "wp1": "3 official backbones x 2 arms x 3 seeds = 18 fixed-final runs",
            "wp2": "3 predeclared zero-prompt MERIT-full NAFNet runs",
            "wp3": "2 additional predeclared structures x 3 seeds",
            "field_evaluation": (
                "All 974 frozen full-resolution frames for all six WP1 checkpoints "
                "per backbone; no family is removed from the field cohort."
            ),
            "result_blind_reduced_candidate_from_v1": "REJECTED_BY_AUTHOR",
        },
        "budget": {
            "v1_model_only_lower_bound_gpu_hours": lower_bound,
            "monitored_execution_budget_gpu_hours": 240.0,
            "contingency_above_model_only_lower_bound_gpu_hours": 240.0 - lower_bound,
            "coverage_warning": (
                "The v1 lower bound excludes detector inference, metrics, image "
                "decode, tiling/blending, artifact I/O, validation, retries, and writing."
            ),
            "hard_monitor": (
                "Pause before exceeding 240 GPU-hours and report the measured cause; "
                "do not silently remove an arm, seed, backbone, or cohort."
            ),
        },
        "deadline_state": {
            "current_official_deadline": "2026-09-27",
            "requested_deadline": "2026-10-25",
            "second_extension_request_sent": "2026-09-24",
            "journal_decision": "PENDING",
            "execution_mode": "AT_RISK_PENDING_JOURNAL_EXTENSION",
            "truthfulness_rule": (
                "This gate closes compute authorization only and does not represent "
                "journal approval of the requested deadline."
            ),
        },
        "hard_stops": [
            "Any frozen source, contract, implementation, preflight, stream, or G11 closure hash mismatch.",
            "Any paired-arm initial-state, sample-stream, target, degradation, or output-domain mismatch.",
            "Any unplanned model, arm, seed, checkpoint selection, or endpoint selection after result inspection.",
            "Any non-finite loss or gradient, corrupted checkpoint, or unavailable recovery path.",
            "Measured execution reaches 240 GPU-hours before numerical freeze.",
        ],
        "required_before_wp1": [
            "Freeze the restart-capable WP1 runner and sequential queue by SHA-256.",
            "Pass a bounded end-to-end checkpoint/resume pilot for all three families and both arms.",
            "Verify adequate local storage and an idle CUDA device.",
        ],
        "source_evidence": {
            "g13_v1_path": str(input_path.resolve()),
            "g13_v1_sha256": sha256_file(input_path),
            "g11_closure_path": str((C009_ROOT / "protocol" / "g11_closure_v1.json").resolve()),
            "g11_closure_sha256": sha256_file(
                C009_ROOT / "protocol" / "g11_closure_v1.json"
            ),
            "training_contract_path": str(
                (C009_ROOT / "protocol" / "g11_training_contract_v2.json").resolve()
            ),
            "training_contract_sha256": sha256_file(
                C009_ROOT / "protocol" / "g11_training_contract_v2.json"
            ),
            "generator_path": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    output["payload_sha256"] = hashlib.sha256(canonical_json_bytes(output)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--author-approved-complete-scope",
        action="store_true",
        help="Required acknowledgement of the author's result-blind full-scope decision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.author_approved_complete_scope:
        raise SystemExit("Missing --author-approved-complete-scope")
    result = authorize(args.input.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
