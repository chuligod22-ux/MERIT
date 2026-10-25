from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


EXECUTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXECUTION_DIR))

from c006_build_bank import crop_origin, discover_parents, load_config, parent_split
from c006_model import NAFNetUNet
from c006_smoke import checkpoint_is_finite
from c006_train import (
    cli_amp_dtype,
    set_determinism,
    state_dict_sha256,
    training_stream,
)


CONFIG_PATH = EXECUTION_DIR / "c006_protocol.json"


def test_frozen_protocol_shape_and_factors() -> None:
    config = load_config(CONFIG_PATH)
    assert config["patch_size"] == 448
    assert config["training"]["arms"] == ["task_off", "task_on"]
    assert config["training"]["seeds"] == [0, 1, 2]
    assert config["training"]["steps"] == 8000
    assert config["training"]["batch_size"] == 4
    assert config["training"]["width"] == 32
    assert config["training"]["amp_dtype"] == "bfloat16"
    assert cli_amp_dtype(config["training"]["amp_dtype"]) == "bf16"


def test_d0_parent_inventory_is_exact_and_excludes_eval_frame() -> None:
    config = load_config(CONFIG_PATH)
    parents = discover_parents(config)
    cam1 = [parent for parent in parents if parent["camera"] == "cam1"]
    cam2 = [parent for parent in parents if parent["camera"] == "cam2"]
    assert len(cam1) == 71
    assert len(cam2) == 70
    assert all(parent["path"].name != "frame_000102.png" for parent in cam1)
    assert {parent["camera"] for parent in parents} == {"cam1", "cam2"}


def test_crop_origins_are_fixed_for_native_frame() -> None:
    config = load_config(CONFIG_PATH)
    expected = {
        "cam2_upper_center_crack": (3872, 655),
        "cam2_upper_left_wall": (1414, 655),
        "cam1_lower_right_wall": (5920, 3549),
    }
    for rule in config["crop_rules"]:
        assert crop_origin(8192, 5460, 448, rule) == expected[rule["rule_id"]]


def test_parent_split_is_deterministic() -> None:
    config = load_config(CONFIG_PATH)
    assert [parent_split(index, config) for index in range(10)] == [
        "synthetic_holdout",
        "train",
        "train",
        "train",
        "train",
        "synthetic_holdout",
        "train",
        "train",
        "train",
        "train",
    ]


def test_same_seed_has_identical_initial_state() -> None:
    set_determinism(2)
    first = NAFNetUNet(width=4, encoder_blocks=(1,), middle_blocks=1, decoder_blocks=(1,))
    first_hash = state_dict_sha256(first.state_dict())
    set_determinism(2)
    second = NAFNetUNet(width=4, encoder_blocks=(1,), middle_blocks=1, decoder_blocks=(1,))
    assert state_dict_sha256(second.state_dict()) == first_hash


def test_model_forward_shape_and_range() -> None:
    set_determinism(0)
    model = NAFNetUNet(
        width=4,
        encoder_blocks=(1,),
        middle_blocks=1,
        decoder_blocks=(1,),
    )
    output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)
    detached = output.detach()
    assert float(detached.min()) >= 0.0
    assert float(detached.max()) <= 1.0


def test_training_stream_is_arm_independent_and_zero_prompt() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        training_config = json.load(handle)["training"]
    bank = np.arange(6 * 32 * 32, dtype=np.uint8).reshape(6, 32, 32)
    indices = np.arange(6, dtype=np.int64)
    first_stream = training_stream(bank, indices, 1, 2, training_config)
    second_stream = training_stream(bank, indices, 1, 2, training_config)
    first_input, first_target, first_metadata = next(first_stream)
    second_input, second_target, second_metadata = next(second_stream)
    np.testing.assert_array_equal(first_input, second_input)
    np.testing.assert_array_equal(first_target, second_target)
    assert first_metadata == second_metadata
    assert np.count_nonzero(first_input[:, 1:]) == 0
    assert first_input.dtype == np.float32
    assert first_target.dtype == np.float32
    assert torch.isfinite(torch.from_numpy(first_input)).all()


def test_checkpoint_verifier_allows_torch_version_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": {"global_step": 2, "torch": torch.__version__},
            "state": {"weight": torch.ones(1)},
        },
        checkpoint_path,
    )
    assert checkpoint_is_finite(checkpoint_path, expected_step=2)
