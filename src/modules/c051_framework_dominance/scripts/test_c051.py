"""Focused invariants for the C051 matched factorial implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c051_analyze as analysis  # noqa: E402
import c051_common as common  # noqa: E402


def test_paired_initialization_matches_shared_parameters() -> None:
    for backbone in common.BACKBONES:
        off, on = common.paired_initial_models(backbone, 0)
        off_state, on_state = off.state_dict(), on.state_dict()
        mismatches = []
        prompt_layer_seen = False
        for name, off_value in off_state.items():
            on_value = on_state[name]
            if off_value.shape == on_value.shape:
                if not torch.equal(off_value, on_value):
                    mismatches.append(name)
            elif (
                off_value.ndim == 4
                and off_value.shape[1] == 1
                and on_value.shape[1] == 3
            ):
                prompt_layer_seen = True
                assert torch.equal(off_value, on_value[:, :1])
                assert torch.count_nonzero(on_value[:, 1:]) == 0
            else:
                mismatches.append(name)
        assert prompt_layer_seen
        assert not mismatches


def test_effect_definition_uses_selected_maximum_backbone_contrast() -> None:
    cells = {
        ("nafnet", "off"): 0.40,
        ("nafnet", "on"): 0.50,
        ("plainunet", "off"): 0.42,
        ("plainunet", "on"): 0.52,
        ("restormer", "off"): 0.44,
        ("restormer", "on"): 0.54,
        ("mamba", "off"): 0.46,
        ("mamba", "on"): 0.56,
    }
    result = analysis.effects(cells)
    assert np.isclose(result["framework_main"], 0.10)
    assert np.isclose(result["max_backbone"], 0.06)
    assert np.isclose(result["dominance"], 0.04)
