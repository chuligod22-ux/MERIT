from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from merit_v2_model import CalibratedPromptBlend, MeritV2, PromptResidualCorrector


class DummyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.75))

    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1] * self.weight + 0.1


def build_model() -> MeritV2:
    torch.manual_seed(7)
    return MeritV2(
        DummyBase(),
        PromptResidualCorrector(hidden_channels=8, blocks=1, max_delta=0.05),
    )


def test_zero_prompt_is_exact_pixel_identity() -> None:
    model = build_model()
    source = torch.rand(2, 1, 16, 16)
    inputs = torch.cat([source, torch.zeros_like(source), torch.zeros_like(source)], dim=1)
    assert torch.equal(model.forward_preclip(inputs), source)


def test_positive_prompt_initially_matches_frozen_base() -> None:
    model = build_model()
    source = torch.rand(2, 1, 16, 16)
    inputs = torch.cat(
        [source, torch.full_like(source, 0.4), torch.full_like(source, 0.2)], dim=1
    )
    output, base, residual, blend = model.forward_components(inputs)
    assert torch.equal(output, base)
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(blend, torch.ones_like(blend))


def test_only_corrector_is_trainable_and_base_stays_in_eval_mode() -> None:
    model = build_model().train()
    assert not model.base.training
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.requires_grad for parameter in model.corrector.parameters())


def test_invalid_input_shape_is_rejected() -> None:
    model = build_model()
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand(1, 1, 8, 8))


def test_calibrated_blend_interpolates_and_keeps_zero_prompt_identity() -> None:
    model = CalibratedPromptBlend(
        DummyBase(), [0.2, 0.5, 0.9], [0.75, 1.0, 1.1]
    )
    source = torch.full((2, 1, 8, 8), 0.4)
    prompts = torch.full_like(source, 0.35)
    positive = torch.cat([source[:1], prompts[:1], prompts[:1]], dim=1)
    output, base, _, blend = model.forward_components(positive)
    assert blend.item() == pytest.approx(0.875)
    assert torch.allclose(output, source[:1] + blend * (base - source[:1]))
    zero = torch.cat([source[1:], torch.zeros_like(source[1:]).repeat(1, 2, 1, 1)], dim=1)
    assert torch.equal(model.forward_preclip(zero), source[1:])
