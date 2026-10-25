from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
C010 = HERE.parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v3_primary_artifacts_and_hashes_are_complete() -> None:
    protocol = C010 / "protocol" / "pilot_v3.json"
    result = load(C010 / "runs" / "pilot_v3_nafnet_seed0" / "RUN_COMPLETE.json")
    assert result["status"] == "PASS"
    assert result["protocol_sha256"] == sha256(protocol)
    assert result["development"]["status"] == "PASS_DEVELOPMENT_TARGETS"
    assert all(result["development"]["gates"].values())
    assert result["development"]["records"] == 500
    assert result["development"]["delta_psnr_db"]["mean"] >= 0.10
    metrics = Path(result["development"]["metrics_path"])
    assert result["development"]["metrics_sha256"] == sha256(metrics)


def test_fixed_calibration_passes_all_three_seeds_and_identity_smoke() -> None:
    validation = load(C010 / "runs" / "pilot_v3_seed_validation.json")
    assert validation["status"] == "PASS"
    assert set(validation["results"]) == {"seed1", "seed2"}
    for record in validation["results"].values():
        development = record["development"]
        assert development["status"] == "PASS_DEVELOPMENT_TARGETS"
        assert all(development["gates"].values())
        assert development["identity"]["max_p99_absolute_change"] == 0.0
    smoke = load(C010 / "runs" / "inference_identity_output.png.json")
    assert smoke["status"] == "PASS"
    assert smoke["input_sha256"] == smoke["output_sha256"]
    assert smoke["input_sha256"] == sha256(Path(smoke["input_path"]))
    assert smoke["output_sha256"] == sha256(Path(smoke["output_path"]))


def test_unused_condition_confirmatory_freeze_and_results_pass() -> None:
    freeze_path = C010 / "protocol" / "confirmatory_freeze_v1.json"
    freeze = load(freeze_path)
    payload = dict(freeze)
    recorded = payload.pop("payload_sha256")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert recorded == hashlib.sha256(canonical).hexdigest()
    assert freeze["status"] == "FROZEN_BEFORE_ANY_CONFIRMATORY_MODEL_OUTPUT"
    assert freeze["parents"] == 68
    assert freeze["crops"] == 98
    assert freeze["c006_c009_parent_overlap"] == 0
    result = load(C010 / "runs" / "confirmatory_v1" / "RUN_COMPLETE.json")
    assert result["status"] == "PASS"
    assert result["freeze_sha256"] == sha256(freeze_path)
    assert all(result["macro"]["gates"].values())
    assert result["macro"]["delta_psnr_db"]["mean"] >= 0.10
    for seed in (0, 1, 2):
        record = result["per_seed"][f"seed{seed}"]
        assert record["status"] == "PASS"
        assert record["records"] == 980
        assert all(record["gates"].values())
        assert record["metrics_sha256"] == sha256(Path(record["metrics_path"]))
