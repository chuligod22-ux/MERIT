from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from c006_g09_e2e import projected_hours
from c006_operational_inference import (
    axis_starts,
    positive_hann_window,
    reflect_pad_to_multiple,
    tiled_detect_rgb,
    tiled_restore_gray,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class IdentityRestorer(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :1]


class ZeroDetector(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (value.shape[0], 1, value.shape[2], value.shape[3]),
            dtype=value.dtype,
            device=value.device,
        )


def test_axis_starts_covers_boundary_once_rule() -> None:
    assert axis_starts(17, 8, 2) == [0, 6, 9]
    assert axis_starts(16, 8, 0) == [0, 8]
    with pytest.raises(ValueError):
        axis_starts(7, 8, 2)


def test_positive_hann_has_no_zero_weight() -> None:
    window = positive_hann_window(8, 0.05)
    assert window.shape == (8, 8)
    assert np.all(window > 0.0)
    np.testing.assert_allclose(window, np.flip(window, axis=(0, 1)))


def test_overlap_blend_identity_is_pixel_exact_after_quantization() -> None:
    gray = np.arange(13 * 17, dtype=np.uint8).reshape(13, 17)
    output = tiled_restore_gray(
        gray,
        IdentityRestorer(),
        torch.device("cpu"),
        tile_size=8,
        overlap=2,
        batch_size=3,
        blend_floor=0.05,
        amp_dtype="none",
    )
    np.testing.assert_array_equal(
        np.rint(output * 255.0).astype(np.uint8), gray
    )


def test_detector_padding_and_crop_preserve_shape_and_range() -> None:
    image = np.zeros((13, 17, 3), dtype=np.uint8)
    padded, bottom, right = reflect_pad_to_multiple(image, 8)
    assert padded.shape == (16, 24, 3)
    assert (bottom, right) == (3, 7)
    probability = tiled_detect_rgb(
        image,
        ZeroDetector(),
        torch.device("cpu"),
        tile_size=8,
        batch_size=2,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    assert probability.shape == (13, 17)
    np.testing.assert_array_equal(probability, np.full((13, 17), 0.5, np.float32))


def test_projection_uses_one_raw_and_each_of_six_seed_conditions() -> None:
    results = [
        {"condition_seconds": {"raw": 1.0 + seed, "off": 10.0 + seed, "on": 20.0 + seed}}
        for seed in range(3)
    ]
    hours, components = projected_hours(results, cohort_frames=988)
    expected_seconds = 988 * (3.0 + 33.0 + 63.0)
    assert components["projected_total_seconds"] == expected_seconds
    assert hours == expected_seconds / 3600.0


def test_operational_protocol_binds_all_model_assets() -> None:
    protocol_path = ROOT / "c006_e2e_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    restoration = protocol["restoration"]
    assert len(restoration["checkpoints"]) == 6
    assert file_hash(Path(restoration["model_code_path"])) == restoration["model_code_sha256"]
    for record in restoration["checkpoints"].values():
        assert file_hash(Path(record["path"])) == record["sha256"]
    detector = protocol["detector"]
    for path_key, hash_key in (
        ("code_path", "code_sha256"),
        ("protocol_path", "protocol_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
    ):
        assert file_hash(Path(detector[path_key])) == detector[hash_key]


def test_g09_aggregate_is_hash_bound_and_fail_closed() -> None:
    protocol_path = ROOT / "c006_e2e_protocol.json"
    artifact_root = ROOT / "artifacts" / "g09_e2e_operational_v4"
    aggregate = json.loads(
        (artifact_root / "g09_e2e_aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["protocol_sha256"] == file_hash(protocol_path)
    assert aggregate["d3_read_or_used"] is False
    assert aggregate["no_oom"] is True
    assert aggregate["peak_gpu_memory_delta_mb"] == 22370
    assert aggregate["projection"]["projected_total_hours"] > 12.0
    assert aggregate["gate"]["pass"] is False
    for record in aggregate["results"]:
        assert file_hash(Path(record["path"])) == record["sha256"]
