"""Independent integrity audit for the completed C012 package."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
C012 = HERE.parent
MODIFIED = C012.parent
C011_SCRIPTS = MODIFIED / "c011_field_validation" / "scripts"
if str(C011_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(C011_SCRIPTS))

import c011_common as common
import c012_eval as evaluation


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot decode audit image: {path}")
    return image


def verify_payload(path: Path) -> None:
    value = common.load_json(path)
    expected = value.pop("payload_sha256", None)
    if expected is None:
        raise RuntimeError(f"Missing payload hash: {path}")
    actual = common.payload_sha256(value)
    if actual != expected:
        raise RuntimeError(f"Payload hash mismatch: {path}")


def main() -> None:
    protocol = evaluation.verify_holdout_protocol()
    protocol_copy = dict(protocol)
    expected_protocol_payload = protocol_copy.pop("payload_sha256")
    if common.payload_sha256(protocol_copy) != expected_protocol_payload:
        raise RuntimeError("Holdout protocol payload hash mismatch")
    groups = {
        "development_d2_v1": (C012 / "runs" / "development_v1" / "d2" / "progress", 6),
        "development_d2_v2": (C012 / "runs" / "development_v2" / "d2" / "progress", 6),
        "development_width": (C012 / "runs" / "development_v1" / "width" / "progress", 75),
        "holdout_d2": (C012 / "runs" / "holdout_v1" / "d2" / "progress", 6),
        "holdout_width": (C012 / "runs" / "holdout_v1" / "width" / "progress", 75),
    }
    counts = {}
    payload_files = []
    for name, (root, expected) in groups.items():
        files = sorted(root.glob("*.json"))
        if len(files) != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {len(files)}")
        counts[name] = len(files)
        payload_files.extend(files)
    auxiliary = [
        C012 / "runs" / "development_v1" / "d2" / "RUN_COMPLETE.json",
        C012 / "runs" / "development_v2" / "d2" / "RUN_COMPLETE.json",
        C012 / "runs" / "development_v1" / "width" / "RUN_COMPLETE.json",
        C012 / "runs" / "holdout_v1" / "d2" / "RUN_COMPLETE.json",
        C012 / "runs" / "holdout_v1" / "width" / "RUN_COMPLETE.json",
        C012 / "runs" / "development_v1" / "summary.json",
        C012 / "runs" / "development_v2" / "summary.json",
        C012 / "runs" / "holdout_v1" / "summary.json",
    ]
    for path in [*payload_files, *auxiliary]:
        verify_payload(path)
    freeze_mtime = evaluation.HOLDOUT_PROTOCOL.stat().st_mtime_ns
    holdout_files = [*groups["holdout_d2"][0].glob("*.json"), *groups["holdout_width"][0].glob("*.json")]
    if min(path.stat().st_mtime_ns for path in holdout_files) <= freeze_mtime:
        raise RuntimeError("A holdout output predates the frozen protocol")
    development = common.load_json(C012 / "runs" / "development_v2" / "summary.json")
    holdout = common.load_json(C012 / "runs" / "holdout_v1" / "summary.json")
    if development["global_gate_pass"] is not True or holdout["global_gate_pass"] is not True:
        raise RuntimeError("Final development or holdout gate is not PASS")
    smoke = C012 / "runs" / "inference_smoke"
    identity = read_image(smoke / "identity.png")
    resnet = read_image(smoke / "resnet.png")
    input_path = Path(common.load_json(smoke / "identity.png.json")["input_path"])
    source = read_image(input_path)
    if source.ndim == 3:
        conversion = cv2.COLOR_BGRA2GRAY if source.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        source = cv2.cvtColor(source, conversion)
    if not np.array_equal(source, identity) or not np.array_equal(source, resnet):
        raise RuntimeError("Identity or ResNet fail-closed output is not pixel exact")
    result: dict[str, Any] = {
        "schema_version": "MERIT-C012-INTEGRITY-AUDIT-v1",
        "status": "PASS",
        "protocol_payload_sha256": expected_protocol_payload,
        "protocol_file_sha256": common.sha256_file(evaluation.HOLDOUT_PROTOCOL),
        "progress_counts": counts,
        "verified_payload_files": len(payload_files) + len(auxiliary),
        "holdout_outputs_postdate_freeze": True,
        "development_global_gate_pass": True,
        "holdout_global_gate_pass": True,
        "identity_pixel_exact": True,
        "resnet_fail_closed_pixel_exact": True,
        "holdout_summary_payload_sha256": holdout["payload_sha256"],
    }
    result["payload_sha256"] = common.payload_sha256(result)
    output = C012 / "runs" / "c012_audit.json"
    common.write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
