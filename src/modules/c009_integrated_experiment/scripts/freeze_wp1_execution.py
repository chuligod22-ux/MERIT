"""Freeze the C009 WP1 runner, queue, dependencies, and 18-run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
C006_ROOT = C009_ROOT.parent / "c006_execution"
DEFAULT_OUTPUT = C009_ROOT / "protocol" / "wp1_execution_freeze_v1.json"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    g13_path = C009_ROOT / "protocol" / "g13_feasibility_v2.json"
    g13 = load_json(g13_path)
    if g13.get("status") != "CLOSED_GO_FULL_SCOPE_AT_AUTHOR_RISK":
        raise RuntimeError("G13 v2 has not authorized full-scope execution")
    if not g13.get("full_training_allowed"):
        raise RuntimeError("G13 v2 full_training_allowed is false")

    files = {
        "g11_training_contract_v2.json": C009_ROOT / "protocol" / "g11_training_contract_v2.json",
        "g11_implementation_manifest_v3.json": C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json",
        "g11_source_preflight_v3.json": C009_ROOT / "protocol" / "g11_source_preflight_v3.json",
        "g11_closure_v1.json": C009_ROOT / "protocol" / "g11_closure_v1.json",
        "g13_feasibility_v2.json": g13_path,
        "c009_wp1_train.py": HERE / "c009_wp1_train.py",
        "c009_wp1_queue.py": HERE / "c009_wp1_queue.py",
        "g11_training_stream.py": HERE / "g11_training_stream.py",
        "g11_official_model_factory.py": HERE / "g11_official_model_factory.py",
        "g11_official_backbone_adapter.py": HERE / "g11_official_backbone_adapter.py",
        "c006_train.py": C006_ROOT / "c006_train.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing execution files: {missing}")

    matrix = [
        {"family": family, "arm": arm, "seed": seed, "steps": 8000}
        for family in ("nafnet", "restormer", "mprnet")
        for seed in (0, 1, 2)
        for arm in ("base", "merit_full")
    ]
    document: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP1-EXECUTION-FREEZE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "authorization": "FULL_SCOPE_AT_AUTHOR_RISK_PENDING_JOURNAL_EXTENSION",
        "g13_sha256": sha256_file(g13_path),
        "frozen_files": {
            name: sha256_file(path.resolve()) for name, path in sorted(files.items())
        },
        "run_matrix": matrix,
        "run_matrix_sha256": hashlib.sha256(canonical_json_bytes(matrix)).hexdigest(),
        "run_count": len(matrix),
        "paired_order": "family -> seed -> base then merit_full",
        "checkpoint_policy": {
            "fixed_final_step": 8000,
            "recovery_interval_steps": 1000,
            "recovery_is_not_result_selection": True,
            "resume_requires_exact_hashes": True,
        },
        "pilot_policy": {
            "families": ["nafnet", "restormer", "mprnet"],
            "arms": ["base", "merit_full"],
            "seed": 0,
            "steps": 20,
            "forced_resume_boundary": 5,
            "pilot_results_are_technical_only": True,
        },
        "post_run_rule": (
            "Do not select a backbone, arm, seed, or checkpoint from training loss. "
            "Complete all 18 fixed-final runs before confirmatory endpoint analysis."
        ),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    document["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
