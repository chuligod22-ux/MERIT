from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
sys.path.insert(0, str(HERE))

import c017_evaluate as evaluation
import c017_models as models

SEGFORMER = C017 / "assets" / "nvidia_segformer_b0_ade512"


@pytest.mark.parametrize("model_name", models.MODEL_NAMES)
def test_model_output_contract(model_name: str) -> None:
    model = models.build_model(model_name, SEGFORMER).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_name", models.MODEL_NAMES)
def test_model_has_finite_gradients(model_name: str) -> None:
    model = models.build_model(model_name, SEGFORMER).train()
    inputs = torch.randn(1, 3, 64, 64)
    loss = model(inputs).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_unknown_model_fails_closed() -> None:
    with pytest.raises(models.C017ModelError):
        models.build_model("not-a-model", SEGFORMER)


def test_evaluation_condition_contract() -> None:
    assert len(evaluation.CONDITIONS) == 13
    assert evaluation.CONDITIONS["identity"] == {"profile": None, "weight": 0.0}
    for profile in evaluation.PROFILES:
        assert evaluation.CONDITIONS[f"{profile}__linear100"] == {
            "profile": profile,
            "weight": 1.0,
        }


def test_probability_fusion_endpoints() -> None:
    raw = torch.tensor([0.2, 0.8]).numpy()
    profile = torch.tensor([0.6, 0.4]).numpy()
    assert evaluation.fuse(raw, profile, 0.0) == pytest.approx(raw)
    assert evaluation.fuse(raw, profile, 1.0) == pytest.approx(profile)
    assert evaluation.fuse(raw, profile, 0.5) == pytest.approx([0.4, 0.6])


def test_payload_hash_ignores_its_own_field() -> None:
    value = {"a": 1}
    digest = evaluation.payload_sha256(value)
    assert evaluation.payload_sha256({**value, "payload_sha256": digest}) == digest
