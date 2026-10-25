from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c010_infer_v3 as inference


class IdentityPromptModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1]


def test_axis_starts_cover_end_without_duplicates() -> None:
    assert inference.axis_starts(448, 448, 112) == [0]
    assert inference.axis_starts(900, 448, 112) == [0, 336, 452]


def test_tiled_identity_preserves_small_odd_image() -> None:
    rng = np.random.RandomState(3)
    image = rng.rand(123, 217).astype(np.float32)
    restored = inference.tiled_restore(
        image,
        IdentityPromptModel(),
        torch.device("cpu"),
        prompt_blur=0.0,
        prompt_noise=0.0,
        tile_size=64,
        overlap=16,
        batch_size=3,
    )
    assert restored.shape == image.shape
    assert np.max(np.abs(restored - image)) < 3e-7


def test_unicode_safe_16bit_roundtrip(tmp_path: Path) -> None:
    source = np.arange(63, dtype=np.uint16).reshape(7, 9) * 1000
    path = tmp_path / "한글_출력.png"
    inference.write_gray(path, source.astype(np.float32) / 65535.0, source.dtype)
    decoded, dtype = inference.read_gray(path)
    assert dtype == np.uint16
    assert np.array_equal(np.rint(decoded * 65535.0).astype(np.uint16), source)
