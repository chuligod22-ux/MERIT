from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


MODULE_PATH = Path(__file__).with_name("g11_official_smoke.py")
SPEC = importlib.util.spec_from_file_location("g11_official_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _StateFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.register_buffer("offset", torch.ones(1))


def test_state_hash_is_stable_and_detects_change() -> None:
    first = _StateFixture()
    second = _StateFixture()
    assert MODULE.state_dict_sha256(first) == MODULE.state_dict_sha256(second)
    with torch.no_grad():
        second.weight[0] += 1
    assert MODULE.state_dict_sha256(first) != MODULE.state_dict_sha256(second)


def test_flop_profile_reports_missing_dependency_or_result() -> None:
    model = nn.Conv2d(3, 3, 1)
    sample = torch.rand(1, 3, 8, 8)
    result = MODULE.profile_flops(model, sample)
    assert result["status"] in {"PASS", "PARTIAL_UNSUPPORTED_OPS", "BLOCKED_DEPENDENCY"}
