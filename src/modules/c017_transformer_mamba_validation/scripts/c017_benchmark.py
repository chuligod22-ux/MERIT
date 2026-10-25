#!/usr/bin/env python
"""Benchmark one C017 training model at the frozen 512 px resolution."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
sys.path.insert(0, str(HERE))
import c017_models as models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=models.MODEL_NAMES, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = models.build_model(
        args.model, C017 / "assets" / "nvidia_segformer_b0_ade512"
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(args.batch, 3, 512, 512, device=device)
    targets = torch.rand(args.batch, 1, 512, 512, device=device).round()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "model": args.model,
                "batch": args.batch,
                "steps": args.steps,
                "seconds": elapsed,
                "images_per_second": args.batch * args.steps / elapsed,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "parameters": models.parameter_count(model),
                "loss": float(loss.detach().cpu()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
