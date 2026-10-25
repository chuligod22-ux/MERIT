"""Eager-mode supported-operation FLOP audit for one active G11 backbone."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.flop_counter import FlopCounterMode


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
MANIFEST = C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json"
PREFLIGHT = C009_ROOT / "protocol" / "g11_source_preflight_v3.json"


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


def serializable_counts(counts: dict[str, dict[Any, int]]) -> dict[str, dict[str, int]]:
    return {
        module: {
            str(operator): int(value)
            for operator, value in sorted(
                operators.items(), key=lambda item: str(item[0])
            )
        }
        for module, operators in sorted(counts.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("nafnet", "restormer", "mprnet"), required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 448x448 FLOP audit")

    factory = load_module(
        f"g11_factory_flops_{args.family}", HERE / "g11_official_model_factory.py"
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, metadata = factory.build_official_adapter(
        args.family, MANIFEST, PREFLIGHT
    )
    model = model.cuda().eval()
    sample = torch.zeros(1, 3, 448, 448, device="cuda", dtype=torch.float32)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with FlopCounterMode(model, depth=1, display=False) as counter:
            output = model.forward_preclip(sample)
    torch.cuda.synchronize()
    if output.shape != (1, 1, 448, 448) or not torch.all(torch.isfinite(output)):
        raise RuntimeError(f"Unexpected profiled output: {tuple(output.shape)}")

    counts = counter.get_flop_counts()
    result = {
        "schema_version": "MERIT-C009-G11-TORCH-FLOP-AUDIT-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_SUPPORTED_OPS",
        "family": args.family,
        "input_shape": [1, 3, 448, 448],
        "selected_output_shape": list(output.shape),
        "precision_during_eager_forward": "bfloat16 autocast",
        "counting_convention": "PyTorch FlopCounterMode; multiply-add is two FLOPs",
        "supported_operation_flops": int(counter.get_total_flops()),
        "supported_operation_gflops": float(counter.get_total_flops()) / 1.0e9,
        "counts": serializable_counts(counts),
        "coverage_limitation": "FlopCounterMode counts only operators with registered formulas; this is a supported-operation lower bound, not a complete total-compute claim.",
        "factory": metadata,
        "manifest_sha256": sha256_file(MANIFEST),
        "preflight_sha256": sha256_file(PREFLIGHT),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "full_training_allowed_by_this_artifact": False,
    }
    output_path = (
        args.output.resolve()
        if args.output
        else C009_ROOT / "smoke" / f"g11_{args.family}_torch_flops_v1.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
