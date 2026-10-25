"""Freeze C009 WP2/WP3 additional training before their results exist."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
C006_ROOT = C009_ROOT.parent / "c006_execution"
OUTPUT = C009_ROOT / "protocol" / "wp23_execution_freeze_v1.json"


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


def main() -> None:
    files = {
        "wp23_training_contract_v1.json": C009_ROOT / "protocol" / "wp23_training_contract_v1.json",
        "c009_wp23_models.py": HERE / "c009_wp23_models.py",
        "c009_wp23_train.py": HERE / "c009_wp23_train.py",
        "c009_wp23_queue.py": HERE / "c009_wp23_queue.py",
        "c009_wp1_train.py": HERE / "c009_wp1_train.py",
        "g11_training_stream.py": HERE / "g11_training_stream.py",
        "g11_official_model_factory.py": HERE / "g11_official_model_factory.py",
        "g11_training_contract_v2.json": C009_ROOT / "protocol" / "g11_training_contract_v2.json",
        "g11_implementation_manifest_v3.json": C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json",
        "g11_source_preflight_v3.json": C009_ROOT / "protocol" / "g11_source_preflight_v3.json",
        "wp1_execution_freeze_v1.json": C009_ROOT / "protocol" / "wp1_execution_freeze_v1.json",
        "g13_feasibility_v2.json": C009_ROOT / "protocol" / "g13_feasibility_v2.json",
        "c006_train.py": C006_ROOT / "c006_train.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing WP23 freeze inputs: {missing}")
    protocol = json.loads(
        files["wp23_training_contract_v1.json"].read_text(encoding="utf-8")
    )
    matrix = protocol["run_matrix"]
    document: dict[str, Any] = {
        "schema_version": "MERIT-C009-WP23-EXECUTION-FREEZE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "result_state": "No WP1 fixed-final result and no WP2/WP3 result existed at freeze.",
        "authorization": "FULL_SCOPE_AT_AUTHOR_RISK_PENDING_JOURNAL_EXTENSION",
        "frozen_files": {
            name: sha256_file(path.resolve()) for name, path in sorted(files.items())
        },
        "run_matrix": matrix,
        "run_matrix_sha256": hashlib.sha256(canonical_json_bytes(matrix)).hexdigest(),
        "additional_run_count": len(matrix),
        "dependency": "WP1 full queue must complete 18/18 before the forced-resume pilot starts.",
        "pilot": {
            "structures": ["zero_prompt_full", "cascade", "parallel"],
            "seed": 0,
            "stop_step": 5,
            "final_step": 20,
            "efficacy_interpretation": "forbidden"
        },
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    document["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
