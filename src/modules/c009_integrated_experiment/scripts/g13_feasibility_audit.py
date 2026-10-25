"""Recompute G13 from official-model training and full-frame timing artifacts."""

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
DEFAULT_OUTPUT = PROTOCOL / "g13_feasibility_v1.json"
FAMILIES = ("nafnet", "restormer", "mprnet")
GPU_HOUR_CEILING = 144.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    closure_path = PROTOCOL / "g11_closure_v1.json"
    closure = load_json(closure_path)
    training: dict[str, dict[str, float]] = {}
    inference: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, str]] = {}

    for family in FAMILIES:
        smoke_path = SMOKE / f"g11_official_{family}_smoke_v3.json"
        smoke = load_json(smoke_path)
        training[family] = {
            row["arm"]: float(row["projected_8000_step_hours_model_and_loss"])
            for row in smoke["results"]
        }
        inference_path = SMOKE / f"g13_{family}_inference_smoke_b32_v2.json"
        inference_smoke = load_json(inference_path)
        projection = inference_smoke["full_frame_projection"]
        inference[family] = {
            "batch_size": int(projection["batch_size"]),
            "tile_count_per_frame": int(projection["tile_count"]),
            "seconds_per_frame_per_checkpoint": float(
                projection["model_only_seconds_per_frame_per_checkpoint"]
            ),
            "six_checkpoints_times_974_frames_gpu_hours": float(
                projection["six_checkpoints_times_974_frames_gpu_hours"]
            ),
            "peak_cuda_memory_bytes": max(
                int(row["peak_cuda_memory_bytes"])
                for row in inference_smoke["records"]
            ),
        }
        evidence[family] = {
            "training_smoke": str(smoke_path.relative_to(C009_ROOT)).replace("\\", "/"),
            "training_smoke_sha256": sha256_file(smoke_path),
            "inference_smoke": str(inference_path.relative_to(C009_ROOT)).replace("\\", "/"),
            "inference_smoke_sha256": sha256_file(inference_path),
        }

    wp1_h = sum(sum(arms.values()) * 3 for arms in training.values())
    wp2_h = training["nafnet"]["merit_full"] * 3
    wp3_h = training["nafnet"]["merit_full"] * 1.25 * 6
    training_h = wp1_h + wp2_h + wp3_h
    inference_h = sum(
        row["six_checkpoints_times_974_frames_gpu_hours"]
        for row in inference.values()
    )
    combined_h = training_h + inference_h
    over_h = combined_h - GPU_HOUR_CEILING

    # This is a prospective, result-blind amendment candidate only.  It is not
    # active until the author approves it and the protocol is re-frozen.
    primary_fullframe_h = inference["nafnet"][
        "six_checkpoints_times_974_frames_gpu_hours"
    ]
    candidate_lower_bound_h = training_h + primary_fullframe_h
    candidate_reserve_h = GPU_HOUR_CEILING - candidate_lower_bound_h

    full_scope_pass = (
        closure.get("status")
        == "CLOSED_WITH_DISCLOSED_PARTIAL_FLOP_COVERAGE"
        and combined_h <= GPU_HOUR_CEILING
    )
    result = {
        "schema_version": "MERIT-C009-G13-FEASIBILITY-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "G13",
        "status": "CLOSED_GO" if full_scope_pass else "BLOCKED_NO_GO_CURRENT_SCOPE",
        "full_training_allowed": bool(full_scope_pass),
        "deadline_state": {
            "current_official_deadline": "2026-09-27",
            "second_extension_request_sent": "2026-09-24",
            "requested_deadline": "2026-10-25",
            "journal_decision": "PENDING",
            "current_deadline_verdict": "NO_GO",
        },
        "g11_closure": {
            "path": str(closure_path.relative_to(C009_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(closure_path),
            "status": closure.get("status"),
        },
        "official_training_projection_gpu_hours": {
            "wp1_18_runs": wp1_h,
            "wp2_additional_zero_prompt_full_3_runs": wp2_h,
            "wp3_two_additional_structures_6_runs_at_1_25x_nafnet_full": wp3_h,
            "total": training_h,
            "coverage": "model and frozen loss only; dataloader, checkpoint I/O, validation, retries, and process startup excluded",
        },
        "official_full_frame_inference_projection_gpu_hours": {
            "cohort_frames": 974,
            "checkpoints_per_family": 6,
            "families": inference,
            "total": inference_h,
            "coverage": "restoration model forward only; image decode, tiling/blending, detector inference, metrics, artifact I/O, and process startup excluded",
        },
        "current_scope_budget": {
            "training_plus_restoration_forward_lower_bound_gpu_hours": combined_h,
            "ceiling_gpu_hours": GPU_HOUR_CEILING,
            "amount_over_ceiling_gpu_hours": over_h,
            "verdict": "NO_GO",
            "reason": "The lower bound already exceeds the ceiling before detector inference, metrics, I/O, validation, retries, and writing/QA.",
        },
        "result_blind_amendment_candidate": {
            "status": "NOT_APPROVED_NOT_FROZEN",
            "scientific_change": "Retain three-backbone paired training and controlled D0/D1 comparisons, but run the full 974-frame high-resolution field stream only for the predeclared primary NAFNet family. Evaluate Restormer and MPRNet on the frozen controlled endpoints and independent external tile endpoint, not on all 974 full frames.",
            "model_only_lower_bound_gpu_hours": candidate_lower_bound_h,
            "remaining_ceiling_reserve_gpu_hours": candidate_reserve_h,
            "what_is_preserved": [
                "three official backbone paired comparison on controlled paired endpoints",
                "NAFNet full-cohort field evaluation",
                "independent external four-tunnel crack endpoint",
                "prompt, architecture, OOD, and printed-width endpoints",
            ],
            "what_is_reduced": "Restormer and MPRNet are not run through every one of the 974 high-resolution D3 frames.",
            "required_before_execution": [
                "Author approval of the scientific scope change before any full result is observed.",
                "A versioned d5690 addendum and amended G13 artifact with exact evaluation cohorts and hashes.",
                "A measured end-to-end pilot including decode, tile blend, detector, metrics, and artifact I/O.",
                "Explicit journal approval of a deadline that leaves at least 10 calendar days after numerical freeze for writing and QA.",
            ],
        },
        "evidence": evidence,
        "decision": {
            "start_full_training_now": False,
            "next_gate": "Author scope decision plus journal extension response, then end-to-end pilot of the approved scope.",
        },
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
