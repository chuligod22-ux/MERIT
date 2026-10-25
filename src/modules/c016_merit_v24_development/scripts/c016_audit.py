"""Fail-closed integrity and release audit for MERIT C016/v2.4."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c016_cycle2 as cycle2  # noqa: E402
import c016_cycle3 as cycle3  # noqa: E402
import c016_cycle4 as cycle4  # noqa: E402
import c016_cycle5 as cycle5  # noqa: E402
import c016_pipeline as base  # noqa: E402
import c016_release as release  # noqa: E402


EXPECTED_DETECTION_PROFILES = {
    "d2_resnet34_unet": "sharp0025",
    "d2_vgg16_unet": "inv1250",
    "d2_deeplabv3_resnet50": "inv0125",
}
EXPECTED_DETECTION_FUSION_WEIGHTS = {
    "d2_resnet34_unet": 1.0,
    "d2_vgg16_unet": 0.75,
    "d2_deeplabv3_resnet50": 0.75,
}
EXPECTED_PROGRESS = {
    base.RUNS / "evaluation_v1" / "detection" / "progress": (
        12,
        "MERIT-C016-DETECTION-EVAL-UNIT-v1",
    ),
    base.RUNS / "evaluation_v1" / "width" / "progress": (
        42,
        "MERIT-C016-WIDTH-EVAL-UNIT-v1",
    ),
    cycle2.RUN_ROOT / "progress": (12, "MERIT-C016-DETECTION-EVAL-UNIT-v1"),
    cycle3.RUN_ROOT / "progress": (12, "MERIT-C016-DETECTION-EVAL-UNIT-v1"),
    cycle4.RUN_ROOT / "progress": (12, "MERIT-C016-DETECTION-CYCLE4-UNIT-v1"),
    cycle5.RUN_ROOT / "progress": (12, "MERIT-C016-DETECTION-CYCLE5-UNIT-v1"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise base.C016Error(message)


def verify_payload(path: Path, schema: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON artifact: {path}")
    payload = base.load_json(path)
    require(
        base.payload_sha256(payload) == payload.get("payload_sha256"),
        f"Payload SHA-256 mismatch: {path}",
    )
    if schema is not None:
        require(
            payload.get("schema_version") == schema,
            f"Unexpected schema in {path}: {payload.get('schema_version')}",
        )
    return payload


def audit_progress() -> dict[str, int]:
    counts: dict[str, int] = {}
    for directory, (expected_count, schema) in EXPECTED_PROGRESS.items():
        files = sorted(directory.glob("*.json"))
        require(
            len(files) == expected_count,
            f"Unexpected unit count in {directory}: {len(files)} != {expected_count}",
        )
        for path in files:
            verify_payload(path, schema)
        counts[str(directory.relative_to(base.C016))] = len(files)
    return counts


def audit_crossfit(selection: dict[str, Any]) -> None:
    expected_training = {
        target: sorted(set(base.TUNNELS) - {target}) for target in base.TUNNELS
    }
    require(set(selection["detection_routes"]) == set(base.TUNNELS), "Detection folds differ")
    require(set(selection["width_routes"]) == set(base.TUNNELS), "Width folds differ")
    for target in base.TUNNELS:
        for record in selection["detection_routes"][target].values():
            training = sorted(record["training_tunnels"])
            require(training == expected_training[target], f"Detection leakage in fold {target}")
            require(target not in training, f"Target tunnel appears in detection training: {target}")
        width_record = selection["width_routes"][target]
        training = sorted(width_record["training_tunnels"])
        require(training == expected_training[target], f"Width leakage in fold {target}")
        require(target not in training, f"Target tunnel appears in width training: {target}")


def audit_release() -> dict[str, Any]:
    release_path = base.RUNS / "c016_release_v24_summary.json"
    summary = verify_payload(release_path, "MERIT-C016-RELEASE-v2.4")
    require(summary["status"] == "SUPPORTED_WITH_LIMITS", "Unexpected release status")
    for label, record in summary["assets"].items():
        base.verify_asset(record, f"release:{label}")

    detection = summary["components"]["crack_detection"]
    width = summary["components"]["natural_crack_apparent_width"]
    require(release.VERSION == "MERIT-v2.4", "Release module version mismatch")
    require(release.DETECTION_VERSION == "MERIT-v2.4-cycle5", "Detection version drift")
    require(release.DETECTION_PROFILES == EXPECTED_DETECTION_PROFILES, "Detection profile drift")
    require(
        release.DETECTION_FUSION_WEIGHTS == EXPECTED_DETECTION_FUSION_WEIGHTS,
        "Detection fusion drift",
    )
    require(detection["profiles"] == EXPECTED_DETECTION_PROFILES, "Summary profile drift")
    require(
        detection["probability_fusion"]["weights"] == EXPECTED_DETECTION_FUSION_WEIGHTS,
        "Summary fusion drift",
    )
    require(detection["adoption"] == "ADOPTED", "Cycle-5 detection was not adopted")
    require(detection["gate_pass"] is True, "Detection gate failed")
    for analysis in (detection["all_seeds"], detection["held_replication_seeds_1_2"]):
        require(analysis["panel_gate_pass"] is True, "Detection panel gate failed")
        require(analysis["superiority_gate_pass"] is True, "Cycle-3 superiority gate failed")
        require(
            analysis["panel"]["delta_auprc"]["ci95_low"] > 0.0,
            "Detection efficacy CI failed",
        )
        require(
            analysis["selected_minus_cycle3_delta_auprc"]["ci95_low"] > 0.0,
            "Cycle-3 superiority CI failed",
        )
        require(
            analysis["selected_minus_cycle3_delta_auprc"]["positive_tunnels"] == 4,
            "Cycle-3 tunnel superiority failed",
        )
    require(
        detection["all_seeds"]["panel"]["delta_matched_fpr"]["mean"] <= 0.0,
        "Detection false-positive gate failed",
    )

    require(release.WIDTH_PROFILE == "inv1250", "Width route drift")
    require(width["profile"] == "inv1250", "Summary width route drift")
    require(width["safe_fallback"] == release.WIDTH_SAFE_FALLBACK, "Fallback route drift")
    require(width["gate_pass"] is True, "Width gate failed")
    require(width["paired_estimable_rate"] >= 0.70, "Width coverage gate failed")
    require(width["delta_absolute_error_px"]["ci95_high"] < 0.0, "Width efficacy CI failed")
    require(width["delta_absolute_error_px"]["improved_tunnels"] == 4, "Width tunnel gate failed")

    experiments = summary["experimental_detection_cycles"]
    require(experiments["cycle1"]["adopted"] is False, "Cycle 1 must remain rejected")
    require(experiments["cycle2"]["adopted"] is False, "Cycle 2 must remain rejected")
    require(experiments["cycle3"]["adopted"] is False, "Cycle 3 must be superseded")
    require(experiments["cycle4"]["adopted"] is False, "Cycle 4 must remain rejected")
    require(experiments["cycle4"]["efficacy_gate_pass"] is False, "Cycle 4 gate drift")
    require(experiments["cycle5"]["adopted"] is True, "Cycle 5 must be adopted")
    require(experiments["cycle5"]["efficacy_gate_pass"] is True, "Cycle 5 gate failed")
    return summary


def main() -> None:
    protocol = base.verify_protocol()
    selection = base.verify_selection()
    cycle2_protocol = cycle2.verify()
    cycle3_protocol = cycle3.verify()
    cycle4_protocol = cycle4.verify_protocol()
    cycle4_selection = cycle4.verify_selection()
    cycle5_protocol = cycle5.verify_protocol()
    cycle5_selection = cycle5.verify_selection()
    audit_crossfit(selection)
    counts = audit_progress()

    cycle1 = verify_payload(
        base.RUNS / "c016_final_summary.json",
        "MERIT-C016-V24-CROSSFIT-SUMMARY-v1",
    )
    cycle2_summary = verify_payload(
        cycle2.RUN_ROOT / "summary.json",
        "MERIT-C016-V24-CYCLE2-SUMMARY-v1",
    )
    cycle3_summary = verify_payload(
        cycle3.RUN_ROOT / "summary.json",
        "MERIT-C016-V24-CYCLE3-SUMMARY-v1",
    )
    cycle4_summary = verify_payload(
        cycle4.RUN_ROOT / "summary.json",
        "MERIT-C016-V24-CYCLE4-SUMMARY-v1",
    )
    cycle5_summary = verify_payload(
        cycle5.RUN_ROOT / "summary.json",
        "MERIT-C016-V24-CYCLE5-SUMMARY-v1",
    )
    released = audit_release()

    report = {
        "status": "PASS",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "selection_payload_sha256": selection["payload_sha256"],
        "cycle2_protocol_payload_sha256": cycle2_protocol["payload_sha256"],
        "cycle3_protocol_payload_sha256": cycle3_protocol["payload_sha256"],
        "cycle4_protocol_payload_sha256": cycle4_protocol["payload_sha256"],
        "cycle4_selection_payload_sha256": cycle4_selection["payload_sha256"],
        "cycle5_protocol_payload_sha256": cycle5_protocol["payload_sha256"],
        "cycle5_selection_payload_sha256": cycle5_selection["payload_sha256"],
        "release_payload_sha256": released["payload_sha256"],
        "verified_progress_units": counts,
        "cycle1_detection_gate_pass": cycle1["detection"]["gate"]["pass"],
        "cycle2_detection_gate_pass": cycle2_summary["detection"]["efficacy_gate_pass"],
        "cycle3_detection_gate_pass": cycle3_summary["efficacy_gate_pass"],
        "cycle4_detection_gate_pass": cycle4_summary["efficacy_gate_pass"],
        "cycle5_detection_gate_pass": cycle5_summary["efficacy_gate_pass"],
        "released_detection_gate_pass": released["components"]["crack_detection"]["gate_pass"],
        "released_width_gate_pass": released["components"]["natural_crack_apparent_width"][
            "gate_pass"
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
