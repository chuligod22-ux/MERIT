from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "c063_full737_battery.py"
SPEC = importlib.util.spec_from_file_location("c063_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_method_family_and_primary_merit() -> None:
    assert len(MODULE.METHODS) == 11
    assert MODULE.METHODS[0] == "degraded"
    assert MODULE.PRIMARY_MERIT == "merit_core_adapter"
    assert "nafnet_fidelity" in MODULE.METHODS
    assert "merit_routed_v3i" in MODULE.METHODS


def test_manifest_uses_all_frames_and_45_primary_cells() -> None:
    manifest = MODULE.build_manifest()
    assert len(manifest) == 737
    assert manifest["condition_cell"].nunique() == 50
    assert manifest.loc[manifest["primary_condition"] == 1, "condition_cell"].nunique() == 45
    assert manifest["analysis_crop"].nunique() == 1


def test_degradation_is_deterministic_and_hash_specific() -> None:
    image = np.arange(128 * 160, dtype=np.uint8).reshape(128, 160)
    first, first_seed = MODULE.deterministic_degrade(image, "01" * 32)
    second, second_seed = MODULE.deterministic_degrade(image, "01" * 32)
    third, third_seed = MODULE.deterministic_degrade(image, "02" * 32)
    assert first_seed == second_seed
    assert first_seed != third_seed
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_tile_starts_cover_without_gaps() -> None:
    for length in (512, 896, 1344, 2184, 8192):
        coverage = np.zeros(length, dtype=bool)
        for start in MODULE.tile_starts(length, MODULE.CLASSICAL_TILE, MODULE.CLASSICAL_OVERLAP):
            coverage[start:min(start + MODULE.CLASSICAL_TILE, length)] = True
        assert coverage.all()


def test_classical_methods_return_finite_uint8() -> None:
    image = np.random.default_rng(7).integers(0, 256, size=(96, 128), dtype=np.uint8)
    for method in MODULE.METHODS[1:8]:
        restored = MODULE.restore_classical(method, image)
        assert restored.shape == image.shape
        assert restored.dtype == np.uint8
        assert np.isfinite(restored).all()
