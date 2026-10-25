"""Apply the fixed v2.2 calibration to the other NAFNet base seeds."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c010_pilot as pilot
from merit_v2_model import CalibratedPromptBlend


PROTOCOL = pilot.C010 / "protocol" / "pilot_v3.json"
OUTPUT = pilot.C010 / "runs" / "pilot_v3_seed_validation.json"


def load_base(seed: int, device: torch.device):
    base, metadata = pilot.factory.build_official_adapter("nafnet")
    checkpoint_path = (
        pilot.C009 / "runs" / "wp1" / "nafnet" / f"base_seed{seed}" / "final.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["model"], strict=True)
    base = base.to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base, {
        "seed": seed,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": pilot.sha256_file(checkpoint_path),
        "model": metadata,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise pilot.PilotError("CUDA is required")
    if OUTPUT.exists():
        print(OUTPUT.read_text(encoding="utf-8"))
        return
    protocol = pilot.load_json(PROTOCOL)
    calibration = protocol["calibration"]
    device = torch.device("cuda")
    results: dict[str, object] = {}
    started = time.perf_counter()
    for seed in (1, 2):
        pilot.set_determinism(20261011 + seed)
        base, provenance = load_base(seed, device)
        model = CalibratedPromptBlend(
            base,
            [float(value) for value in calibration["severity_anchors"]],
            [float(value) for value in calibration["blend_anchors"]],
        ).to(device).eval()
        run_dir = pilot.C010 / "runs" / f"pilot_v3_nafnet_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        development = pilot.evaluate(model, protocol, device, run_dir)
        results[f"seed{seed}"] = {
            "provenance": provenance,
            "development": development,
        }
        del model, base
        torch.cuda.empty_cache()
    all_pass = all(
        value["development"]["status"] == "PASS_DEVELOPMENT_TARGETS"
        for value in results.values()
    )
    output = {
        "schema_version": "MERIT-C010-V2-SEED-VALIDATION-v1",
        "status": "PASS" if all_pass else "ITERATE",
        "scientific_status": "EXPLORATORY_UNBLINDED_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": pilot.sha256_file(PROTOCOL),
        "runner_sha256": pilot.sha256_file(Path(__file__).resolve()),
        "fixed_calibration_from_seed0": calibration,
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    pilot.write_json_atomic(OUTPUT, output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
