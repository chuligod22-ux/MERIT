"""Diagnose MPRNet output saturation during the bounded G11 smoke sequence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
C006_ROOT = C009_ROOT.parent / "c006_execution"
MANIFEST = C009_ROOT / "protocol" / "g11_implementation_manifest_v2.json"
PREFLIGHT = C009_ROOT / "protocol" / "g11_source_preflight_v2.json"
CONTRACT = C009_ROOT / "protocol" / "g11_training_contract_v1.json"
OUTPUT = C009_ROOT / "smoke" / "g11_mprnet_stability_diagnostic_v1.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_stats(preclip: torch.Tensor, state: int) -> dict[str, Any]:
    value = preclip.float()
    below = value < 0.0
    above = value > 1.0
    clipped = torch.clamp(value, 0.0, 1.0)
    return {
        "state_after_optimizer_steps": state,
        "all_finite": bool(torch.all(torch.isfinite(value)).item()),
        "minimum": float(value.min().cpu()),
        "maximum": float(value.max().cpu()),
        "mean": float(value.mean().cpu()),
        "below_zero_fraction": float(below.float().mean().cpu()),
        "above_one_fraction": float(above.float().mean().cpu()),
        "out_of_range_fraction": float(torch.logical_or(below, above).float().mean().cpu()),
        "clipped_zero_fraction": float((clipped == 0.0).float().mean().cpu()),
        "clipped_one_fraction": float((clipped == 1.0).float().mean().cpu()),
    }


def run_arm(arm: str) -> dict[str, Any]:
    smoke = load_module(f"g11_smoke_diag_{arm}", HERE / "g11_official_smoke.py")
    factory = load_module(f"g11_factory_diag_{arm}", HERE / "g11_official_model_factory.py")
    loss_helpers = smoke.load_c006_loss_helpers()
    smoke.set_determinism(0)
    degraded, target, metadata, stream = smoke.frozen_batch(CONTRACT, 0)
    model, model_metadata = factory.build_official_adapter(
        "mprnet", MANIFEST, PREFLIGHT
    )
    initial_state_hash = smoke.state_dict_sha256(model)
    inputs = torch.from_numpy(stream.arm_inputs(degraded, metadata, arm)).cuda()
    target_tensor = torch.from_numpy(target).cuda()
    model = model.cuda().train()
    contract = stream.load_json(CONTRACT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["training"]["learning_rate"]),
        weight_decay=float(contract["training"]["weight_decay"]),
    )
    l1 = torch.nn.L1Loss()
    segmenter = None
    target_probability = None
    if arm == "merit_full":
        segmenter = loss_helpers.load_deepcrack(
            loss_helpers.load_json(C006_ROOT / "c006_protocol.json"),
            torch.device("cuda"),
        )
        with torch.no_grad():
            target_probability = loss_helpers.deepcrack_probability(
                segmenter, target_tensor
            )

    states: list[dict[str, Any]] = []
    losses: list[dict[str, float]] = []
    for state in range(8):
        with torch.no_grad(), smoke.amp_context(torch.device("cuda")):
            preclip = model.forward_preclip(inputs)
        states.append(output_stats(preclip, state))
        if state == 7:
            break

        optimizer.zero_grad(set_to_none=True)
        with smoke.amp_context(torch.device("cuda")):
            output = model(inputs)
        output_float = output.float()
        target_float = target_tensor.float()
        l1_value = l1(output_float, target_float)
        gradient_value = loss_helpers.grad_loss(output_float, target_float)
        fft_value = loss_helpers.fft_loss(output_float, target_float)
        task_value = torch.zeros((), device="cuda", dtype=torch.float32)
        loss = l1_value + gradient_value + 0.1 * fft_value
        if arm == "merit_full":
            assert segmenter is not None and target_probability is not None
            task_value = l1(
                loss_helpers.deepcrack_probability(segmenter, output_float),
                target_probability,
            )
            loss = loss + float(contract["training"]["task_lambda"]) * task_value
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
            for parameter in model.parameters()
        )
        losses.append(
            {
                "optimizer_step": state + 1,
                "total": float(loss.detach().cpu()),
                "l1": float(l1_value.detach().cpu()),
                "gradient": float(gradient_value.detach().cpu()),
                "fft": float(fft_value.detach().cpu()),
                "task": float(task_value.detach().cpu()),
                "finite_gradients": bool(finite_gradients),
            }
        )
        optimizer.step()
        torch.cuda.synchronize()

    first_complete = next(
        (
            row["state_after_optimizer_steps"]
            for row in states
            if row["out_of_range_fraction"] == 1.0
        ),
        None,
    )
    del optimizer, inputs, target_tensor, segmenter, target_probability, model
    torch.cuda.empty_cache()
    return {
        "arm": arm,
        "initial_state_sha256": initial_state_hash,
        "model": model_metadata,
        "states": states,
        "losses": losses,
        "first_state_with_complete_out_of_range": first_complete,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    arms = [run_arm("base"), run_arm("merit_full")]
    result = {
        "schema_version": "MERIT-C009-G11-MPRNET-STABILITY-DIAGNOSTIC-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_training": False,
        "full_training_allowed_by_this_artifact": False,
        "manifest_sha256": sha256_file(MANIFEST),
        "preflight_sha256": sha256_file(PREFLIGHT),
        "contract_sha256": sha256_file(CONTRACT),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "paired_initial_state_equal": (
            arms[0]["initial_state_sha256"] == arms[1]["initial_state_sha256"]
        ),
        "arms": arms,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
