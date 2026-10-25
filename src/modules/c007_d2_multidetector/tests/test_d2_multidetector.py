from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("d2_multidetector_tested", ROOT / "d2_multidetector.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_parent_rows(delta_by_detector: dict[str, float]):
    rows = []
    parents = {
        "A": ["a1", "a2"],
        "B": ["b1"],
        "C": ["c1", "c2"],
        "D": ["d1", "d2"],
    }
    for detector, delta in delta_by_detector.items():
        for tunnel, names in parents.items():
            for parent in names:
                for condition, value in (
                    ("degraded", 0.20),
                    ("task_on", 0.20 + delta),
                    ("task_off", 0.19),
                    ("reference", 0.70),
                ):
                    rows.append(
                        {
                            "detector": detector,
                            "tunnel": tunnel,
                            "parent": parent,
                            "condition": condition,
                            "pixel_auprc": value,
                            "f1_at_0_5": value,
                        }
                    )
    return rows


def test_protocol_freezes_three_independent_detectors_and_legacy_reference():
    protocol = MODULE.load_protocol(ROOT / "protocol.json")
    assert tuple(protocol["detectors"]["independent_panel"]) == MODULE.INDEPENDENT
    assert protocol["detectors"]["coupled_reference"] == MODULE.LEGACY
    assert MODULE.LEGACY not in MODULE.INDEPENDENT


def test_model_adapters_emit_one_logit_at_input_size():
    protocol = MODULE.load_protocol(ROOT / "protocol.json")
    legacy = MODULE.import_module(
        Path(protocol["paths"]["legacy_architecture_code"]), "legacy_test_arch"
    )
    inputs = torch.zeros(2, 3, 128, 128)
    vgg, _ = MODULE.create_train_model(
        "d2_vgg16_unet", protocol, legacy, initialize_encoder=False
    )
    deep, _ = MODULE.create_train_model(
        "d2_deeplabv3_resnet50", protocol, legacy, initialize_encoder=False
    )
    vgg.eval()
    deep.eval()
    with torch.inference_mode():
        assert tuple(vgg(inputs).shape) == (2, 1, 128, 128)
        assert tuple(deep(inputs).shape) == (2, 1, 128, 128)


def test_panel_bootstrap_uses_independent_detectors_only():
    deltas = {detector: 0.05 for detector in MODULE.INDEPENDENT}
    deltas[MODULE.LEGACY] = -0.90
    result = MODULE.bootstrap_panel(
        synthetic_parent_rows(deltas),
        "pixel_auprc",
        "task_on",
        "degraded",
        resamples=100,
        seed=17,
    )
    assert abs(result["point"] - 0.05) < 1e-12
    assert result["ci_lower"] > 0
    assert result["positive_tunnels"] == 4


def test_detector_gate_requires_effect_ci_and_tunnel_direction():
    assert MODULE.detector_gate(
        {"point": 0.01, "ci_lower": 0.001, "positive_tunnels": 3}
    )
    assert not MODULE.detector_gate(
        {"point": 0.01, "ci_lower": -0.001, "positive_tunnels": 4}
    )
    assert not MODULE.detector_gate(
        {"point": 0.01, "ci_lower": 0.001, "positive_tunnels": 2}
    )
