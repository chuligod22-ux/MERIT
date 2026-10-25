"""Freeze the result summarizer and report renderer before C009 results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
OUTPUT = C009 / "protocol" / "postprocess_execution_freeze_v3.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    names = [
        "c009_evaluation_common.py",
        "c009_summarize_results.py",
        "c009_generate_final_report.py",
        "c009_postprocess_supervisor.py",
        "test_c009_postprocess.py",
    ]
    payload: dict[str, Any] = {
        "schema_version": "MERIT-C009-POSTPROCESS-EXECUTION-FREEZE-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_FROZEN_BEFORE_C009_RESULTS",
        "result_access_at_freeze": (
            "WP1 nafnet/base_seed0 had reached its fixed-final checkpoint, but no "
            "checkpoint was selected or evaluated and no C009 endpoint result existed. "
            "The v2 postprocessor was stopped while waiting and is superseded by this "
            "stricter G12-complete implementation. No C009 endpoint result existed."
        ),
        "supersedes": {
            "path": str((C009 / "protocol" / "postprocess_execution_freeze_v2.json").resolve()),
            "sha256": sha256_file(C009 / "protocol" / "postprocess_execution_freeze_v2.json"),
        },
        "frozen_files": {name: sha256_file(HERE / name) for name in names},
        "required_queue_item_count": 84,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260924,
        "test_evidence": "5 passed: cluster weighting, equal-tunnel bootstrap, Holm adjustment, matched-sensitivity burden, and global efficacy-floor enforcement",
        "verdict_rule": "Exact global categories in g12_endpoint_gate_freeze_v1.json; no favorable subgroup rescue.",
        "outputs": [
            "evaluation/final_summary/c009_final_summary_v3.json",
            "40_수정본/d5800_merit_c009_full_experiment_results.md",
        ],
    }
    payload["payload_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": payload["status"], "path": str(OUTPUT), "sha256": sha256_file(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
