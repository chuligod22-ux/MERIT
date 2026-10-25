from __future__ import annotations

import pytest
import torch

import c012_profiles as profiles


class _Base(torch.nn.Module):
    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1] * 0.8 + 0.1


def inputs() -> torch.Tensor:
    source = torch.linspace(0.0, 1.0, 2 * 32 * 32).reshape(2, 1, 32, 32)
    return torch.cat([source, torch.zeros_like(source), torch.zeros_like(source)], dim=1)


def test_zero_prompt_route_is_bit_exact_identity() -> None:
    value = inputs()
    output = profiles.route(value, _Base(), active=False, task="crack_detection")
    assert torch.equal(output, value[:, :1])


@pytest.mark.parametrize("detector", tuple(profiles.DETECTOR_PROFILE))
def test_detector_routes_are_finite_bounded(detector: str) -> None:
    output = profiles.route(inputs(), _Base(), active=True, task="crack_detection", detector_name=detector)
    assert output.shape == inputs()[:, :1].shape
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_width_route_is_inverse_base_residual() -> None:
    value = inputs()
    source = value[:, :1]
    base = _Base().forward_preclip(value)
    expected = torch.clamp(source - 0.5 * (base - source), 0.0, 1.0)
    actual = profiles.route(value, _Base(), active=True, task="printed_width")
    assert torch.equal(actual, expected)


def test_unknown_task_and_detector_fail_closed() -> None:
    with pytest.raises(ValueError):
        profiles.route(inputs(), _Base(), active=True, task="unknown")
    with pytest.raises(ValueError):
        profiles.route(inputs(), _Base(), active=True, task="crack_detection", detector_name="unknown")
