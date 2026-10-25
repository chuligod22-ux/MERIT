#!/usr/bin/env python
"""Run one real D2 training batch through each C017 detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
C017 = HERE.parent
MODIFIED = C017.parent
CV_DIR = MODIFIED / "c005_sensitivity" / "d2_tunnel_cv"
FROZEN_DIR = MODIFIED / "c005_freeze" / "d2_detector"
for directory in (HERE, CV_DIR, FROZEN_DIR):
    sys.path.insert(0, str(directory))

import c017_models as models
import d2_detector as frozen
import d2_tunnel_cv as cv


def main() -> int:
    cv_protocol = cv.load_protocol(CV_DIR / "protocol.json")
    frozen_protocol = frozen.load_protocol(FROZEN_DIR / "protocol.json")
    d2_root = Path(r"<LOCAL_PATH_REDACTED>")
    pairs = cv.collect_parent_pairs(d2_root, cv_protocol, frozen_protocol)["0036"][:2]
    dataset = frozen.D2Tiles(pairs, tile_size=512, rotation_quadrants=(0, 1, 2, 3), seed=20260921)
    loader = frozen.make_loader(dataset, 2, 0, False, 20260921)
    images, masks, _, _ = next(iter(loader))
    device = frozen.require_cuda()
    results = {}
    for model_name in models.MODEL_NAMES:
        frozen.seed_everything(20260921)
        model = models.build_model(
            model_name, C017 / "assets" / "nvidia_segformer_b0_ade512"
        ).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images.to(device))
        loss = cv.loss_for_cv(logits, masks.to(device), frozen_protocol)
        loss.backward()
        optimizer.step()
        finite_gradients = all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        results[model_name] = {
            "shape": list(logits.shape),
            "loss": float(loss.detach().cpu()),
            "finite_logits": bool(torch.isfinite(logits).all()),
            "finite_gradients": bool(finite_gradients),
        }
        del model, optimizer, logits, loss
        torch.cuda.empty_cache()
    if not all(
        row["shape"] == [2, 1, 512, 512]
        and row["finite_logits"]
        and row["finite_gradients"]
        for row in results.values()
    ):
        raise RuntimeError(f"C017 smoke test failed: {results}")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
