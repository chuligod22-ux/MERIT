from __future__ import annotations

import torch

import c012_explore as explore


class _Base(torch.nn.Module):
    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1] * 0.9 + 0.05


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = _Base()


def test_candidate_contract_shape_range_and_raw_identity() -> None:
    source = torch.linspace(0.0, 1.0, 2 * 1 * 32 * 32).reshape(2, 1, 32, 32)
    inputs = torch.cat([source, torch.zeros_like(source), torch.zeros_like(source)], dim=1)
    outputs = explore.make_candidates(inputs, _Model())
    assert tuple(outputs) == explore.CONDITIONS
    assert torch.equal(outputs["raw"], source)
    assert all(output.shape == source.shape for output in outputs.values())
    assert all(torch.isfinite(output).all() for output in outputs.values())
    assert all(float(output.min()) >= 0.0 and float(output.max()) <= 1.0 for output in outputs.values())


def test_inverse_residual_direction() -> None:
    source = torch.full((1, 1, 16, 16), 0.25)
    inputs = torch.cat([source, torch.zeros_like(source), torch.zeros_like(source)], dim=1)
    outputs = explore.make_candidates(inputs, _Model())
    assert torch.all(outputs["inv0250"] < source)
