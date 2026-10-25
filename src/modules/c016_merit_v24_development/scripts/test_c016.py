"""Contract tests for MERIT v2.4 cross-fitted development."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c016_pipeline as pipeline
import c016_profiles as profiles


class ConstantBase(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full_like(inputs[:, :1], self.value)


def test_identity_is_bit_exact_and_active_profile_is_bounded() -> None:
    inputs = torch.zeros((2, 3, 12, 12), dtype=torch.float32)
    inputs[:, 0] = 0.4
    outputs = profiles.make_candidates(
        inputs,
        ConstantBase(0.2),
        [profiles.IDENTITY, "inv0500", "combo_a050_g005"],
    )
    assert torch.equal(outputs[profiles.IDENTITY], inputs[:, :1])
    assert torch.allclose(outputs["inv0500"], torch.full_like(outputs["inv0500"], 0.5))
    assert float(outputs["combo_a050_g005"].min()) >= 0.0
    assert float(outputs["combo_a050_g005"].max()) <= 1.0


def test_width_safe_route_preserves_estimable_raw_measurement() -> None:
    raw = {"status": "ESTIMABLE", "estimate_px": 2.25}
    candidate = {"status": "NON_ESTIMABLE", "reason": "LOW_SNR"}
    effective, fallback = profiles.safe_width_measurement(raw, candidate)
    assert fallback is True
    assert effective == raw
    candidate_ok = {"status": "ESTIMABLE", "estimate_px": 2.0}
    effective_ok, fallback_ok = profiles.safe_width_measurement(raw, candidate_ok)
    assert fallback_ok is False
    assert effective_ok == candidate_ok


def _detection_row(tunnel: str, value: float) -> dict[str, object]:
    return {
        "tunnel": tunnel,
        "detector": "d2_resnet34_unet",
        "parent": f"p{tunnel}",
        "condition": "inv0500",
        "delta_auprc": value,
        "delta_f1": value,
        "delta_matched_fpr": -0.001,
    }


def test_detection_route_score_excludes_target_tunnel() -> None:
    rows = [
        _detection_row("A", -99.0),
        _detection_row("B", 0.01),
        _detection_row("C", 0.02),
        _detection_row("D", -0.001),
    ]
    score = pipeline._equal_tunnel_detection_score(
        rows, "A", "d2_resnet34_unet", "inv0500"
    )
    assert set(score["per_tunnel"]) == {"B", "C", "D"}
    assert abs(score["delta_auprc"] - (0.01 + 0.02 - 0.001) / 3.0) < 1e-12
    assert score["positive_tunnels"] == 2
    assert score["eligible"] is True


def test_width_route_score_excludes_target_and_accepts_safe_improvement() -> None:
    rows = []
    for tunnel, delta in (("A", 50.0), ("B", -0.2), ("C", -0.1), ("D", 0.05)):
        rows.append(
            {
                "roi_id": f"r{tunnel}",
                "image_id": f"i{tunnel}",
                "tunnel": tunnel,
                "parent": f"p{tunnel}",
                "condition": "inv0250",
                "raw_status": "ESTIMABLE",
                "effective_status": "ESTIMABLE",
                "delta_absolute_error_px": delta,
            }
        )
    score = pipeline._width_training_score(rows, "A", "inv0250")
    assert set(score["per_tunnel"]) == {"B", "C", "D"}
    assert score["improved_tunnels"] == 2
    assert score["eligible"] is True
