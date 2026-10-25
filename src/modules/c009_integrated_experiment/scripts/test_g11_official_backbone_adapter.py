from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


MODULE_PATH = Path(__file__).with_name("g11_official_backbone_adapter.py")
SPEC = importlib.util.spec_from_file_location("g11_official_backbone_adapter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FirstChannelRestorationAdapter = MODULE.FirstChannelRestorationAdapter


class _DummyOfficialBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale


class _DummyMultiStageBackbone(nn.Module):
    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        return [inputs * 0.5, inputs * 0.25, inputs * 0.125]


def test_adapter_preserves_first_channel_and_gradient() -> None:
    backbone = _DummyOfficialBackbone()
    model = FirstChannelRestorationAdapter(backbone)
    inputs = (torch.rand(2, 3, 16, 16) * 0.4).requires_grad_(True)

    outputs = model(inputs)
    assert outputs.shape == (2, 1, 16, 16)
    assert torch.equal(outputs, inputs[:, :1] * backbone.scale)
    assert torch.equal(model.forward_preclip(inputs), inputs[:, :1] * backbone.scale)

    outputs.mean().backward()
    assert backbone.scale.grad is not None
    assert inputs.grad is not None


def test_adapter_clamps_but_retains_preclip_for_safety_audit() -> None:
    backbone = _DummyOfficialBackbone()
    model = FirstChannelRestorationAdapter(backbone)
    inputs = torch.ones(1, 3, 4, 4)
    assert torch.all(model.forward_preclip(inputs) == 2.0)
    assert torch.all(model(inputs) == 1.0)


def test_adapter_rejects_wrong_channel_count() -> None:
    model = FirstChannelRestorationAdapter(_DummyOfficialBackbone())
    wrong = torch.rand(1, 1, 16, 16)
    try:
        model(wrong)
    except ValueError as exc:
        assert "[batch, 3, height, width]" in str(exc)
    else:
        raise AssertionError("wrong-channel input was not rejected")


def test_adapter_selects_frozen_mprnet_final_output() -> None:
    model = FirstChannelRestorationAdapter(
        _DummyMultiStageBackbone(), output_selector="sequence_index_0"
    )
    inputs = torch.rand(2, 3, 8, 8)
    assert torch.equal(model.forward_preclip(inputs), inputs[:, :1] * 0.5)


def test_adapter_rejects_sequence_for_tensor_selector() -> None:
    model = FirstChannelRestorationAdapter(_DummyMultiStageBackbone())
    try:
        model(torch.rand(1, 3, 8, 8))
    except TypeError as error:
        assert "tensor selector" in str(error)
    else:
        raise AssertionError("Sequence output should require an explicit selector")
