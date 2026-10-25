from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wp3_parameter_matching_and_forward_contract() -> None:
    models = load_module("c009_wp23_models_test", HERE / "c009_wp23_models.py")
    for structure, expected in (("cascade", 16_296_154), ("parallel", 16_296_158)):
        torch.manual_seed(0)
        model, metadata = models.build_structure(structure)
        assert metadata["parameters"] == expected
        assert abs(float(metadata["relative_to_single"])) <= 0.05
        sample = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            preclip = model.forward_preclip(sample)
            clamped = model(sample)
        assert preclip.shape == (1, 1, 32, 32)
        assert clamped.shape == preclip.shape
        assert torch.isfinite(preclip).all()
        assert torch.min(clamped) >= 0.0
        assert torch.max(clamped) <= 1.0
        del model


def test_wp23_contract_has_exact_nine_run_matrix() -> None:
    protocol = json.loads(
        (C009_ROOT / "protocol" / "wp23_training_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = protocol["run_matrix"]
    assert len(matrix) == 9
    assert len(
        {(row["wp"], row["structure"], int(row["seed"])) for row in matrix}
    ) == 9
    assert protocol["wp3"]["cascade"]["parameters"] == 16_296_154
    assert protocol["wp3"]["parallel"]["parameters"] == 16_296_158
    assert protocol["wp3"]["parameter_gate"] == "PASS for cascade and parallel"
