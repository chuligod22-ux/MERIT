from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stateful_stream_matches_frozen_generator_and_resumes() -> None:
    trainer = load_module("c009_wp1_train_test", HERE / "c009_wp1_train.py")
    frozen = load_module("g11_training_stream_test", HERE / "g11_training_stream.py")
    bank = np.arange(7 * 12 * 12, dtype=np.uint16).reshape(7, 12, 12)
    bank = np.mod(bank, 256).astype(np.uint8)
    indices = np.array([0, 2, 3, 6], dtype=np.int64)
    degradation = {
        "sigma_h_px": [0.2, 3.0],
        "sigma_v_px": [0.2, 2.4],
        "noise_std_8bit": [0.0, 3.0],
    }
    reference = frozen.latent_stream(bank, indices, 2, 3, degradation)
    stateful = trainer.PairedTrainingStream(bank, indices, 2, 3, degradation)
    for _ in range(3):
        expected = next(reference)
        actual = stateful.next_batch()
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2] == expected[2]

    saved = stateful.get_state()
    expected_after_save = stateful.next_batch()
    resumed = trainer.PairedTrainingStream(bank, indices, 999, 3, degradation)
    resumed.set_state(saved)
    actual_after_resume = resumed.next_batch()
    np.testing.assert_array_equal(actual_after_resume[0], expected_after_save[0])
    np.testing.assert_array_equal(actual_after_resume[1], expected_after_save[1])
    assert actual_after_resume[2] == expected_after_save[2]


def test_g13_and_execution_freeze_are_internally_bound() -> None:
    trainer = load_module("c009_wp1_gate_test", HERE / "c009_wp1_train.py")
    protocol = C009_ROOT / "protocol"
    contract, freeze = trainer.verify_execution_gate(
        protocol / "g11_training_contract_v2.json",
        protocol / "g11_implementation_manifest_v3.json",
        protocol / "g11_source_preflight_v3.json",
        protocol / "g11_closure_v1.json",
        protocol / "g13_feasibility_v2.json",
        protocol / "wp1_execution_freeze_v1.json",
    )
    assert contract["training"]["steps"] == 8000
    assert freeze["run_count"] == 18
    assert len(freeze["run_matrix"]) == 18
    assert {row["family"] for row in freeze["run_matrix"]} == {
        "nafnet",
        "restormer",
        "mprnet",
    }
    assert {row["arm"] for row in freeze["run_matrix"]} == {
        "base",
        "merit_full",
    }
    expected = hashlib.sha256(
        json.dumps(
            freeze["run_matrix"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert freeze["run_matrix_sha256"] == expected


def test_full_queue_accepts_the_closed_resume_pilot() -> None:
    queue = load_module("c009_wp1_queue_test", HERE / "c009_wp1_queue.py")
    freeze = queue.verify_freeze(
        C009_ROOT / "protocol" / "wp1_execution_freeze_v1.json"
    )
    closure = queue.verify_pilot_closure(
        C009_ROOT / "protocol" / "wp1_pilot_closure_v1.json", freeze
    )
    assert closure["status"] == "PASS"
    assert len(closure["records"]) == 6
    assert all(record["resume_count"] >= 1 for record in closure["records"])
