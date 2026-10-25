"""Evaluate the bounded development-calibrated MERIT v2.2 blend."""

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
RUN_DIR = pilot.C010 / "runs" / "pilot_v3_nafnet_seed0"


def main() -> None:
    if not torch.cuda.is_available():
        raise pilot.PilotError("CUDA is required")
    if (RUN_DIR / "RUN_COMPLETE.json").exists():
        print((RUN_DIR / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        return
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    protocol = pilot.load_json(PROTOCOL)
    pilot.set_determinism(20261011)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    scaffold, provenance = pilot.build_model(
        {
            **pilot.load_json(pilot.DEFAULT_PROTOCOL),
            "base": protocol["base"],
        },
        device,
    )
    calibration = protocol["calibration"]
    model = CalibratedPromptBlend(
        scaffold.base,
        [float(value) for value in calibration["severity_anchors"]],
        [float(value) for value in calibration["blend_anchors"]],
    ).to(device).eval()
    started = time.perf_counter()
    development = pilot.evaluate(model, protocol, device, RUN_DIR)
    state_path = RUN_DIR / "calibration_state.pt"
    torch.save(
        {
            "severity_anchors": model.severity_anchors.cpu(),
            "blend_anchors": model.blend_anchors.cpu(),
            "protocol_sha256": pilot.sha256_file(PROTOCOL),
            "base_checkpoint_sha256": provenance["base_checkpoint_sha256"],
        },
        state_path,
    )
    result = {
        "schema_version": "MERIT-C010-V2-PILOT-COMPLETE-v3",
        "status": "PASS" if development["status"] == "PASS_DEVELOPMENT_TARGETS" else "ITERATE",
        "scientific_status": "EXPLORATORY_UNBLINDED_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": pilot.sha256_file(PROTOCOL),
        "runner_sha256": pilot.sha256_file(Path(__file__).resolve()),
        "model_sha256": pilot.sha256_file(HERE / "merit_v2_model.py"),
        "calibration_state_path": str(state_path.resolve()),
        "calibration_state_sha256": pilot.sha256_file(state_path),
        "provenance": provenance,
        "development": development,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    pilot.write_json_atomic(RUN_DIR / "RUN_COMPLETE.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
