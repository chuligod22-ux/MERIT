from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("g11_training_stream.py")
SPEC = importlib.util.spec_from_file_location("g11_training_stream", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def small_stream(seed: int):
    bank = np.arange(6 * 8 * 8, dtype=np.uint8).reshape(6, 8, 8)
    indices = np.arange(6, dtype=np.int64)
    degradation = {
        "sigma_h_px": [0.2, 3.0],
        "sigma_v_px": [0.2, 2.4],
        "noise_std_8bit": [0.0, 3.0],
    }
    return MODULE.latent_stream(bank, indices, seed, 2, degradation)


def test_stream_is_deterministic_by_seed() -> None:
    first = next(small_stream(7))
    second = next(small_stream(7))
    for left, right in zip(first[:2], second[:2]):
        assert np.array_equal(left, right)
    assert first[2] == second[2]


def test_stream_changes_with_seed() -> None:
    first = next(small_stream(7))
    second = next(small_stream(8))
    assert MODULE.array_sha256(first[0]) != MODULE.array_sha256(second[0])


def test_arms_only_change_prompt_channels() -> None:
    degraded, target, metadata = next(small_stream(11))
    base = MODULE.arm_inputs(degraded, metadata, "base")
    full = MODULE.arm_inputs(degraded, metadata, "merit_full")
    assert target.shape == (2, 1, 8, 8)
    assert np.array_equal(base[:, 0], full[:, 0])
    assert np.count_nonzero(base[:, 1:]) == 0
    assert np.all(full[:, 1:] >= 0.0)
    assert np.all(full[:, 1:] <= 1.0)
    for index, sample in enumerate(metadata["samples"]):
        expected_blur = 0.5 * (
            sample["sigma_h_px"] + sample["sigma_v_px"]
        ) / 3.0
        expected_noise = sample["noise_std_8bit"] / 3.0
        assert np.all(full[index, 1] == np.float32(expected_blur))
        assert np.all(full[index, 2] == np.float32(expected_noise))


def test_unknown_arm_is_rejected() -> None:
    degraded, _, metadata = next(small_stream(3))
    try:
        MODULE.arm_inputs(degraded, metadata, "unknown")
    except ValueError as error:
        assert "Unknown arm" in str(error)
    else:
        raise AssertionError("Unknown arm should be rejected")
