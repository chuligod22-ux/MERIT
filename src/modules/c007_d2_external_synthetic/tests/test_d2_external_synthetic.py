from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "d2_external_synthetic_under_test", ROOT / "d2_external_synthetic.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_protocol_contract() -> None:
    protocol = MODULE.load_protocol(ROOT / "protocol.json")
    assert protocol["data"]["selection"]["expected_tiles"] == 1703
    assert protocol["data"]["selection"]["expected_parents"] == 14
    assert protocol["statistics"]["conditions"] == list(MODULE.CONDITIONS)


def test_degradation_is_deterministic_and_non_identity() -> None:
    protocol = MODULE.load_protocol(ROOT / "protocol.json")
    clean = np.tile(np.arange(512, dtype=np.uint16) % 256, (512, 1)).astype(np.uint8)
    first = MODULE.degrade_gray(clean, protocol, "0036_0_0.png")
    second = MODULE.degrade_gray(clean, protocol, "0036_0_0.png")
    other = MODULE.degrade_gray(clean, protocol, "0036_0_4.png")
    assert np.array_equal(first, second)
    assert not np.array_equal(first, clean)
    assert not np.array_equal(first, other)


def test_paired_metrics_identity_and_change() -> None:
    clean = np.zeros((8, 8), dtype=np.uint8)
    changed = clean.copy()
    changed[:, 4:] = 10
    psnr, gradient = MODULE.paired_image_metrics(clean, changed)
    assert np.isfinite(psnr)
    assert gradient > 0.0


def test_stratified_bootstrap_uses_equal_tunnel_weight() -> None:
    rows = []
    for tunnel, parents, delta in (
        ("A", ("a1", "a2"), 0.1),
        ("B", ("b1",), 0.2),
        ("C", ("c1", "c2", "c3"), 0.3),
        ("D", ("d1",), 0.4),
    ):
        for parent in parents:
            rows.extend(
                [
                    {"tunnel": tunnel, "parent": parent, "condition": "task_on", "pixel_auprc": 0.5 + delta},
                    {"tunnel": tunnel, "parent": parent, "condition": "degraded", "pixel_auprc": 0.5},
                ]
            )
    result = MODULE.stratified_parent_bootstrap(
        rows,
        "pixel_auprc",
        "task_on",
        "degraded",
        resamples=100,
        seed=1,
    )
    assert abs(result["point"] - 0.25) < 1e-12
    assert abs(result["ci_lower"] - 0.25) < 1e-12
    assert abs(result["ci_upper"] - 0.25) < 1e-12
