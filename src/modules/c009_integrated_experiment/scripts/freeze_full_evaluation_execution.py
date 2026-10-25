"""Bind the full evaluation implementation and the pre-result addendum."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
REVISION = C009.parent.parent
OUTPUT = C009 / "protocol" / "full_evaluation_execution_freeze_v2.json"
CONTRACT = C009 / "protocol" / "full_evaluation_contract_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite freeze: {OUTPUT}")
    frozen_names = [
        "c009_evaluation_common.py",
        "c009_controlled_eval.py",
        "c009_prompt_eval.py",
        "c009_d2_external_eval.py",
        "c009_d3_fullframe_eval.py",
        "c009_width_eval.py",
        "c009_full_evaluation_queue.py",
        "test_c009_full_evaluation.py",
    ]
    frozen_files = {name: sha256_file(HERE / name) for name in frozen_names}
    fixed_final = sorted((C009 / "runs" / "wp1").glob("**/final.pt"))
    lpips_package = Path(
        r"<LOCAL_PATH_REDACTED>"
    )
    alexnet = Path(
        r"<LOCAL_PATH_REDACTED>"
    )
    legacy_raw = (
        REVISION
        / "40_수정본"
        / "c006_execution"
        / "artifacts"
        / "d3_stream_v1"
        / "d3_endpoint_streaming_final.json"
    )
    trace = (
        C009
        / "artifacts"
        / "width_raw_measurement_v2"
        / "width_chart_raw_traces_v2.jsonl"
    )
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C009-FULL-EVALUATION-EXECUTION-FREEZE-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_FROZEN_BEFORE_C009_ENDPOINT_RESULTS",
        "authorization": "FULL_SCOPE_AT_AUTHOR_RISK_PENDING_JOURNAL_EXTENSION",
        "result_access_at_freeze": (
            "One fixed-final training checkpoint (nafnet/base_seed0) existed and the next "
            "training run was active. No checkpoint had been evaluated and no C009 "
            "endpoint result existed. This v2 closes omissions found in a result-blind "
            "audit; it does not alter any model, seed, cohort, degradation, or endpoint gate."
        ),
        "supersedes": file_record(C009 / "protocol" / "full_evaluation_execution_freeze_v1.json"),
        "training_checkpoints_present_but_unevaluated": [file_record(path) for path in fixed_final],
        "contract": file_record(CONTRACT),
        "frozen_files": frozen_files,
        "generator": file_record(Path(__file__)),
        "scientific_addendum": {
            "reason": (
                "The parent contract fixed WP8 scope but did not spell out the controlled "
                "degradation or the implementation binding for LPIPS and frozen RAW traces. "
                "These details are fixed here before any C009 endpoint result."
            ),
            "wp8_controlled_degradation": {
                "sigma_h_px": 2.6,
                "sigma_v_px": 2.4,
                "noise_std_8bit": 2.5,
                "base_seed": 20260924,
                "per_source_seed": "uint32 big-endian sha256('20260924:'+source_id)[0:4]",
                "implementation": "scipy gaussian_filter reflect then deterministic Gaussian noise and uint8 round/clip",
                "prompt_base": [0.0, 0.0],
                "prompt_merit_full": [0.8333333333333334, 0.8333333333333334],
                "trace_rule": "reuse the frozen RAW v2 orientation and stored trace for both controlled-degraded and restored variants",
            },
            "lpips": {
                "network": "AlexNet",
                "input": "grayscale replicated to RGB, scaled to [-1,1]",
                "alexnet_weight": file_record(alexnet),
                "lpips_linear_weight": file_record(lpips_package),
            },
            "wp6_raw_baseline_reuse": {
                "asset": file_record(legacy_raw),
                "rule": "reuse only the frozen raw detector endpoint; all 18 C009 restored endpoints are recomputed",
            },
            "width_raw_trace": file_record(trace),
            "overshoot_metric": (
                "At clean Sobel-edge pixels above the 75th percentile, sample clean and "
                "candidate at +/-1.5 px along the clean gradient normal and report the "
                "95th percentile excursion outside the clean endpoint envelope."
            ),
            "matched_sensitivity_burden": {
                "reference": "degraded-input recall at the frozen threshold 0.5, per detector, tunnel, parent, and seed pairing",
                "restored_operating_point": "highest 4096-bin score threshold attaining at least the reference recall",
                "burden": "restored minus degraded false-positive pixel rate",
                "gate": "parent-within-tunnel macro bootstrap 95% CI upper bound <= 0",
            },
        },
        "queue": {
            "controlled_wp1_wp3_ood_items": 27,
            "wp2_prompt_items": 3,
            "wp5_external_items": 18,
            "wp8_width_items": 18,
            "wp6_d3_items": 18,
            "total_items": 84,
            "order": [
                "controlled_wp1_wp3_and_ood",
                "wp2_prompt_perturbation",
                "wp5_external_four_tunnel",
                "wp8_printed_width",
                "wp6_different_tunnel_same_system_974",
            ],
            "result_blinding": "queue checks process exit and coverage markers only; it does not select by performance",
            "resume": "completed model/fold/frame artifacts are hash/identity checked and skipped; incomplete atomic units may be recomputed",
        },
        "queue_item_count": 84,
        "gpu_monitor": {
            "hard_budget_hours": 240.0,
            "accounting": "sum full-run manifest elapsed_seconds plus sequential evaluation command wall time",
            "poll_seconds": 30,
            "action": "terminate active evaluation and fail closed at the hard budget",
        },
        "test_evidence": {
            "command": "pytest -q scripts/test_c009_full_evaluation.py",
            "result": "4 passed",
            "scope": [
                "27-model matrix and 6 OOD families",
                "all OOD operators finite and bounded",
                "nine exact prompt perturbations",
                "frozen RAW trace reuse on an actual width ROI",
                "84-item queue and 240 GPU-hour monitor declaration",
            ],
        },
        "hard_stops": [
            "Any frozen execution file or scientific asset hash mismatch.",
            "Any incomplete upstream training block or non-final checkpoint.",
            "Any incomplete predeclared evaluation block presented as complete.",
            "Any non-finite output, coverage mismatch, or 240 GPU-hour budget hit.",
            "Any outcome-based change to model, seed, condition, prompt, detector, tunnel, frame, or width inclusion.",
        ],
    }
    payload["payload_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "path": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "queue_item_count": payload["queue_item_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
