from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c009_controlled_eval as controlled
import c009_evaluation_common as common
import c009_full_evaluation_queue as queue
import c009_prompt_eval as prompt
import c009_width_eval as width


def test_frozen_contract_and_complete_matrix() -> None:
    contract = common.load_contract()
    wp1 = common.wp1_specifications(contract)
    wp23 = common.wp23_specifications(contract)
    assert len(wp1) == 18
    assert len(wp23) == 9
    assert len({common.model_id(record) for record in wp1 + wp23}) == 27
    assert len(contract["wp4_ood"]["families"]) == 6
    assert len(contract["assets"]["d2_detector_checkpoints"]) == 36
    assert contract["wp6_different_tunnel_same_system"]["eligible_rows"] == 974


def test_controlled_degradations_and_prompt_variants_are_finite() -> None:
    contract = common.load_contract()
    clean = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    for family, severities in contract["wp4_ood"]["families"].items():
        output = controlled.ood_degrade(
            clean, family, severities[0], np.random.RandomState(20260924)
        )
        assert output.shape == clean.shape
        assert output.dtype == np.float32
        assert np.isfinite(output).all()
        assert float(output.min()) >= 0.0
        assert float(output.max()) <= 1.0
    correct_blur = np.linspace(0.1, 0.9, 10, dtype=np.float32)
    correct_noise = np.linspace(0.9, 0.1, 10, dtype=np.float32)
    variants = prompt.prompt_variants(correct_blur, correct_noise, contract)
    assert [value[0] for value in variants] == [
        "correct",
        "shuffled",
        "swapped",
        "constant",
        "zero",
        "mismatch_0.5",
        "mismatch_0.75",
        "mismatch_1.25",
        "mismatch_1.5",
    ]
    assert np.array_equal(variants[1][1], np.roll(correct_blur, 1))


def test_width_variant_reuses_frozen_raw_trace() -> None:
    contract = common.load_contract()
    ledger, traces = width.load_inputs(contract)
    v1, v2 = width.load_measurement_modules()
    row = next(value for value in ledger if float(value["nominal_width_mm"]) == 0.5)
    source = width.SOURCE_ROOT / row["filename"]
    assert common.sha256_file(source) == row["source_sha256"]
    image = v1.read_image(source)
    patch = v1.rectify(image, v1.polygon(row))
    trace = traces[(row["source_id"], 0.5)]
    result = width.measure_variant(patch, trace, row["source_id"], 0.5, v1, v2)
    assert result["trace_source"] == "FROZEN_RAW_V2"
    assert result["trace_orientation"] == trace["trace_orientation"]
    assert result["status"] in {
        "ESTIMABLE",
        "QC_NON_ESTIMABLE",
        "UNCERTAINTY_NON_ESTIMABLE",
    }
    assert result["candidate_profiles"] > 0


def test_queue_declares_all_blocks() -> None:
    source = (HERE / "c009_full_evaluation_queue.py").read_text(encoding="utf-8")
    for script in (
        "c009_controlled_eval.py",
        "c009_prompt_eval.py",
        "c009_d2_external_eval.py",
        "c009_width_eval.py",
        "c009_d3_fullframe_eval.py",
    ):
        assert script in source
    assert "expected_labels = (" in source
    assert "240 GPU-hour hard budget reached" in source


def test_resume_preserves_elapsed_budget_and_completion_prefix() -> None:
    previous = {
        "started_at_kst": "2026-09-25T09:36:32+09:00",
        "completed_items": ["controlled:first", "controlled:second"],
        "evaluation_elapsed_seconds": 255703.22364169685,
        "active_elapsed_seconds": 17583.001031700056,
        "active_block": "wp6_different_tunnel_same_system_974",
        "active_item": "wp1__restormer__base__seed2",
    }
    state = queue.prepare_state(previous, "base-freeze", "resume-addendum")
    assert state["completed_items"] == previous["completed_items"]
    assert state["evaluation_elapsed_seconds"] == pytest.approx(273286.2246733969)
    assert state["active_elapsed_seconds"] == 0.0
    assert state["hard_gpu_budget_hours"] == 240.0
    assert state["resume_count"] == 1
    assert state["interrupted_active_item"] == "wp1__restormer__base__seed2"
    total_hours = (234476.60488980124 + state["evaluation_elapsed_seconds"]) / 3600.0
    assert total_hours == pytest.approx(141.0452304342217)
    queue.validate_completed_prefix(
        state, ["controlled:first", "controlled:second", "wp2_prompt:seed0"]
    )
    queue.record_completed(state, "controlled:second")
    assert len(state["completed_items"]) == 2
    queue.record_completed(state, "wp2_prompt:seed0")
    assert len(state["completed_items"]) == 3


def test_resume_rejects_duplicate_or_nonprefix_completion() -> None:
    with pytest.raises(queue.QueueError, match="duplicate completed items"):
        queue.prepare_state(
            {"completed_items": ["controlled:first", "controlled:first"]},
            "base-freeze",
            "resume-addendum",
        )
    state = queue.prepare_state(
        {"completed_items": ["controlled:second"]},
        "base-freeze",
        "resume-addendum",
    )
    with pytest.raises(queue.QueueError, match="completion order"):
        queue.validate_completed_prefix(
            state, ["controlled:first", "controlled:second"]
        )


def test_budget_stops_before_subprocess_launch(tmp_path: Path) -> None:
    state = {
        "completed_items": [],
        "evaluation_elapsed_seconds": 240.0 * 3600.0,
        "hard_gpu_budget_hours": 240.0,
    }
    with pytest.raises(queue.QueueError, match="before evaluation launch"):
        queue.run_command(
            [sys.executable, "does_not_run.py"],
            tmp_path / "blocked.log",
            tmp_path / "state.json",
            state,
            "blocked",
            "blocked",
            0.0,
        )


def test_resume_addendum_and_interrupted_artifacts_verify() -> None:
    freeze, addendum = queue.verify_freeze(
        queue.DEFAULT_FREEZE, queue.DEFAULT_RESUME_ADDENDUM
    )
    assert freeze["queue_item_count"] == 84
    assert addendum["status"] == "PASS_OPERATIONAL_RESUME_ONLY"
    state = common.load_json(queue.DEFAULT_QUEUE)
    contract = common.load_contract()
    wp1 = common.wp1_specifications(contract)
    wp23 = common.wp23_specifications(contract)
    expected = (
        [f"controlled:{common.model_id(value)}" for value in wp1 + wp23]
        + [f"wp2_prompt:seed{seed}" for seed in (0, 1, 2)]
        + [f"wp5:{common.model_id(value)}" for value in wp1]
        + [f"wp8:{common.model_id(value)}" for value in wp1]
        + [f"wp6:{common.model_id(value)}" for value in wp1]
    )
    queue.validate_completed_prefix(state, expected)
    queue.validate_completed_artifacts(state["completed_items"])
    if "resume_count" not in state:
        assert common.sha256_file(queue.DEFAULT_QUEUE) == addendum["interruption_state"][
            "sha256"
        ]
        assert len(state["completed_items"]) == 76
    else:
        allowed_addenda = {
            common.sha256_file(queue.DEFAULT_RESUME_ADDENDUM),
            *addendum.get("accepted_prior_addendum_sha256s", []),
        }
        assert state["resume_addendum_sha256"] in allowed_addenda
        assert 76 <= len(state["completed_items"]) <= 84


def test_startup_failure_does_not_take_over_foreign_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    common.write_json_atomic(state_path, {"queue_pid": -1, "status": "running"})
    assert not queue.owns_state_file(state_path)
    common.write_json_atomic(
        state_path, {"queue_pid": queue.os.getpid(), "status": "initializing"}
    )
    assert queue.owns_state_file(state_path)


def test_training_gpu_accounting_rejects_invalid_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "runs" / "wp1" / "bad" / "run_manifest.json"
    common.write_json_atomic(
        manifest_path,
        {"status": "complete", "global_step": 8000, "elapsed_seconds": -1.0},
    )
    monkeypatch.setattr(queue, "C009", tmp_path)
    with pytest.raises(queue.QueueError, match="Invalid GPU time"):
        queue.completed_training_gpu_seconds()
