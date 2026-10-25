"""Freeze the complete C009 post-training evaluation design before outcomes exist.

The document produced by this script fixes the model matrix, controlled
degradations, prompt policy, full-frame cohort, external detector panel, and
nominal-width assets.  It deliberately does not read any C009 trained
checkpoint or performance output.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
REVISION = C009.parent.parent
DEFAULT_OUTPUT = C009 / "protocol" / "full_evaluation_contract_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def asset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def wp1_matrix() -> list[dict[str, Any]]:
    return [
        {"family": family, "arm": arm, "seed": seed}
        for family in ("nafnet", "restormer", "mprnet")
        for seed in (0, 1, 2)
        for arm in ("base", "merit_full")
    ]


def wp23_matrix() -> list[dict[str, Any]]:
    return [
        {"wp": wp, "structure": structure, "seed": seed}
        for wp, structures in (
            ("wp2", ("zero_prompt_full",)),
            ("wp3", ("cascade", "parallel")),
        )
        for structure in structures
        for seed in (0, 1, 2)
    ]


def build() -> dict[str, Any]:
    c006 = REVISION / "40_수정본" / "c006_execution"
    c005 = REVISION / "40_수정본" / "c005_freeze"
    d2_multi = REVISION / "40_수정본" / "c007_d2_multidetector"
    d2_ext = REVISION / "40_수정본" / "c007_d2_external_synthetic"
    perimeter = REVISION / "40_수정본" / "c007_full_perimeter_harmonization"

    d0_manifest = c006 / "artifacts" / "d0_bank_manifest.csv"
    d2_cohort = d2_ext / "artifacts" / "external_v1" / "cohort.csv"
    d3_cohort = c005 / "artifacts" / "v1" / "d3_cohort.csv"
    d3_consensus = (
        c005
        / "runs"
        / "annotations_v2"
        / "consensus"
        / "frozen"
        / "d3_consensus_final.csv"
    )
    perimeter_cohort = (
        perimeter
        / "artifacts"
        / "full_perimeter_v1"
        / "full_perimeter_cohort.csv"
    )

    detector_checkpoints: list[dict[str, Any]] = []
    multi_root = d2_multi / "artifacts" / "multidetector_v2" / "training"
    for checkpoint in sorted(multi_root.glob("*/fold_*/seed_*/final_epoch.pt")):
        record = asset(checkpoint)
        record["detector"] = checkpoint.parents[2].name
        record["fold"] = checkpoint.parents[1].name.removeprefix("fold_")
        record["detector_seed"] = int(checkpoint.parent.name.removeprefix("seed_"))
        detector_checkpoints.append(record)
    cv_root = REVISION / "40_수정본" / "c005_sensitivity" / "d2_tunnel_cv"
    for checkpoint in sorted(
        (cv_root / "artifacts" / "cv_v1").glob("fold_*/seed_*/final_epoch.pt")
    ):
        record = asset(checkpoint)
        record["detector"] = "d2_resnet34_unet"
        record["fold"] = checkpoint.parents[1].name.removeprefix("fold_")
        record["detector_seed"] = int(checkpoint.parent.name.removeprefix("seed_"))
        detector_checkpoints.append(record)
    if len(detector_checkpoints) != 36:
        raise RuntimeError(
            f"Expected 36 frozen detector checkpoints, found {len(detector_checkpoints)}"
        )

    payload: dict[str, Any] = {
        "schema_version": "MERIT-C009-FULL-EVALUATION-CONTRACT-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_ANY_C009_FIXED_FINAL_OR_ENDPOINT_RESULT",
        "authorization": "FULL_SCOPE_AT_AUTHOR_RISK_PENDING_JOURNAL_EXTENSION",
        "result_access_at_freeze": (
            "WP1 was still training nafnet/base_seed0; no C009 fixed-final "
            "checkpoint, WP2/WP3 result, or C009 endpoint result existed."
        ),
        "global_rules": {
            "fixed_final_only": True,
            "no_checkpoint_selection": True,
            "no_external_data_for_model_or_threshold_selection": True,
            "all_predeclared_arms_seeds_families_conditions_reported": True,
            "partial_outputs_are_operational_progress_only": True,
            "performance_summary_release": (
                "Only after the relevant complete predeclared comparison block passes "
                "hash and coverage checks."
            ),
            "hard_gpu_budget_hours": 240.0,
            "journal_extension_state": "PENDING",
        },
        "model_matrix": {
            "wp1": wp1_matrix(),
            "wp1_count": 18,
            "wp23_additional": wp23_matrix(),
            "wp23_additional_count": 9,
            "reuse": {
                "wp2_informative_oracle": "wp1/nafnet/merit_full_seed{0,1,2}",
                "wp3_single": "wp1/nafnet/merit_full_seed{0,1,2}",
            },
        },
        "controlled_d0": {
            "cohort": "all 50 synthetic_holdout bank rows; no row selection",
            "in_distribution_conditions": [
                {"id": "id_low", "sigma_h_px": 0.7, "sigma_v_px": 0.6, "noise_std_8bit": 0.75},
                {"id": "id_mid", "sigma_h_px": 1.6, "sigma_v_px": 1.3, "noise_std_8bit": 1.5},
                {"id": "id_high", "sigma_h_px": 2.7, "sigma_v_px": 2.3, "noise_std_8bit": 2.75},
            ],
            "noise_seeds": [202609240, 202609241, 202609242],
            "input_conditions": ["degraded", "zero_degradation_identity"],
            "metrics": [
                "PSNR",
                "LPIPS-Alex",
                "gradient_MAE",
                "preclip_out_of_range_fraction",
                "mean_absolute_output_change",
                "p99_absolute_output_change",
                "saturation_fraction_change",
                "edge_normal_overshoot_p95",
            ],
            "prompt_policy": {
                "base": "zero maps",
                "merit_full": "oracle normalized blur/noise severity",
                "identity": "zero maps for every arm",
                "normalization": "p_b=(sigma_h+sigma_v)/(2*3); p_n=noise_std_8bit/3; clamp [0,1]",
            },
        },
        "wp2_prompt_perturbation": {
            "models": "three fixed-final WP1 NAFNet merit_full checkpoints",
            "cohort_conditions_seeds": "same controlled_d0 full matrix",
            "batch_order": "bank_index ascending, then condition id, then noise seed",
            "batch_size_for_shuffle": 10,
            "conditions": {
                "correct": "oracle pair",
                "shuffled": "cyclic +1 complete prompt pair within each fixed batch",
                "swapped": "exchange p_b and p_n",
                "constant": {"p_b": 0.48333333333333334, "p_n": 0.5},
                "zero": {"p_b": 0.0, "p_n": 0.0},
                "mismatch_multipliers": [0.5, 0.75, 1.25, 1.5],
            },
        },
        "wp3_architecture": {
            "models": ["single", "cascade", "parallel"],
            "seeds": [0, 1, 2],
            "cohort_conditions_seeds": "same controlled_d0 full matrix plus identity",
        },
        "wp4_ood": {
            "cohort": "all 50 D0 synthetic_holdout rows",
            "models": "all 18 WP1 fixed-final checkpoints; no post-result selection",
            "stochastic_seeds": [202609240, 202609241, 202609242],
            "families": {
                "poisson_gaussian": [
                    {"peak_photons": 160.0, "additive_std_8bit": 1.0},
                    {"peak_photons": 80.0, "additive_std_8bit": 2.0},
                    {"peak_photons": 40.0, "additive_std_8bit": 3.0},
                ],
                "linear_motion_psf": [
                    {"length_px": 3, "angle_deg": 0.0},
                    {"length_px": 7, "angle_deg": 5.0},
                    {"length_px": 13, "angle_deg": 10.0},
                ],
                "spatially_varying_blur": [
                    {"sigma_left_px": 0.4, "sigma_right_px": 1.4},
                    {"sigma_left_px": 0.5, "sigma_right_px": 2.8},
                    {"sigma_left_px": 0.6, "sigma_right_px": 4.2},
                ],
                "saturation_clipping": [
                    {"gain": 1.05, "clip_high": 0.98},
                    {"gain": 1.20, "clip_high": 0.94},
                    {"gain": 1.40, "clip_high": 0.88},
                ],
                "illumination_gradient_vignetting": [
                    {"edge_gain": 0.85, "vertical_gain": 0.95},
                    {"edge_gain": 0.65, "vertical_gain": 0.85},
                    {"edge_gain": 0.45, "vertical_gain": 0.70},
                ],
                "veiling_scattering": [
                    {"contrast": 0.90, "veil": 0.04},
                    {"contrast": 0.75, "veil": 0.10},
                    {"contrast": 0.55, "veil": 0.18},
                ],
            },
            "prompt_rule": (
                "OOD factors have no in-training oracle encoding; both prompt maps are zero. "
                "This tests unsupported-input behavior, not operational severity estimation."
            ),
        },
        "wp5_external_four_tunnel": {
            "cohort_rows": csv_rows(d2_cohort),
            "physical_parents": 14,
            "tunnels": ["A", "B", "C", "D"],
            "models": "all 18 WP1 fixed-final checkpoints",
            "synthetic_degradation": {
                "sigma_h_px": 2.6,
                "sigma_v_px": 2.4,
                "noise_std_8bit": 2.5,
                "base_seed": 20260923,
            },
            "prompts": {
                "base": "zero",
                "merit_full": {"p_b": 0.8333333333333334, "p_n": 0.8333333333333334},
            },
            "detectors": [
                "d2_resnet34_unet",
                "d2_vgg16_unet",
                "d2_deeplabv3_resnet50",
            ],
            "detector_seed_pairing": [
                {"restorer_seed": 0, "detector_seed": 20260921},
                {"restorer_seed": 1, "detector_seed": 20260922},
                {"restorer_seed": 2, "detector_seed": 20260923},
            ],
            "detector_threshold": 0.5,
            "histogram_bins": 4096,
            "inference_unit": "tile",
            "statistical_unit": "parent nested in tunnel",
        },
        "wp6_different_tunnel_same_system": {
            "cohort_rows_total": csv_rows(d3_cohort),
            "eligible_rows": 974,
            "unreadable_excluded": 14,
            "models": "all 18 WP1 fixed-final checkpoints",
            "input_prompt": "zero maps for every model; actual per-frame severity is unavailable",
            "tiling": {
                "tile_size": 448,
                "overlap": 112,
                "stride": 336,
                "blend": "separable Hann with floor 0.05",
                "batch_size": {
                    "nafnet": 32,
                    "restormer": 16,
                    "mprnet": 16,
                },
            },
            "output_storage": (
                "stream detector/descriptor endpoints; do not persist 18x974 full-resolution outputs"
            ),
            "claim_boundary": (
                "image-visible detector response and quality-descriptor harmonization only; "
                "no clean-reference restoration accuracy or physical false-positive claim"
            ),
        },
        "wp7_qc": {
            "field_reference_label": "least-degraded operational field reference",
            "field_reference_condition": "60 km/h, 2.5 m, ISO100",
            "metrics": [
                "brightness",
                "noise_proxy",
                "saturation_fraction",
                "local_contrast",
            ],
            "indoor_role": "ISO 12233 sensitivity only; not field lux calibration or clean truth",
        },
        "wp8_printed_width": {
            "models": "all 18 WP1 fixed-final checkpoints",
            "source_images": 50,
            "nominal_widths_mm": [round(index / 10, 1) for index in range(1, 11)],
            "predeclared_non_estimable_mm": [0.1, 0.2],
            "estimable_roi_count": 387,
            "estimable_roi_denominator": 400,
            "method": "orientation-aware RAW-only v2 frozen before C009 restoration output",
            "claim_boundary": "printed nominal apparent width only; not natural-crack physical width",
        },
        "assets": {
            "g10": asset(C009 / "protocol" / "g10_provenance_manifest_v1.json"),
            "g11_contract": asset(C009 / "protocol" / "g11_training_contract_v2.json"),
            "g12": asset(C009 / "protocol" / "g12_endpoint_gate_freeze_v1.json"),
            "g13": asset(C009 / "protocol" / "g13_feasibility_v2.json"),
            "wp1_freeze": asset(C009 / "protocol" / "wp1_execution_freeze_v1.json"),
            "wp23_freeze": asset(C009 / "protocol" / "wp23_execution_freeze_v1.json"),
            "d0_bank": asset(c006 / "artifacts" / "d0_bank_448.npy"),
            "d0_manifest": asset(d0_manifest),
            "d2_protocol": asset(d2_ext / "protocol.json"),
            "d2_cohort": asset(d2_cohort),
            "d2_multidetector_protocol": asset(d2_multi / "protocol.json"),
            "d2_detector_checkpoints": detector_checkpoints,
            "d3_protocol": asset(c006 / "c006_d3_stream_protocol.json"),
            "d3_cohort": asset(d3_cohort),
            "d3_consensus": asset(d3_consensus),
            "perimeter_protocol": asset(perimeter / "full_perimeter_protocol.json"),
            "perimeter_cohort": asset(perimeter_cohort),
            "width_roi": asset(C009 / "protocol" / "width_chart_roi_final_v1.csv"),
            "width_protocol": asset(C009 / "protocol" / "width_measurement_protocol_v2.json"),
            "width_raw_audit": asset(
                C009
                / "artifacts"
                / "width_raw_measurement_v2"
                / "width_chart_raw_audit_v2.json"
            ),
        },
        "hard_stops": [
            "Any frozen source, protocol, cohort, detector, or checkpoint hash mismatch.",
            "Any WP1 or WP2/WP3 run block is incomplete or uses a non-final checkpoint.",
            "Any post-result model, seed, family, severity, tunnel, parent, frame, or width exclusion.",
            "Any non-finite model output or incomplete predeclared comparison block.",
            "Any attempt to infer or tune actual-acquisition prompts from D2/D3 endpoint outcomes.",
            "Cumulative monitored GPU time reaches 240 hours before completion.",
        ],
    }
    payload["design_payload_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def main() -> None:
    payload = build()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen contract: {DEFAULT_OUTPUT}")
    with DEFAULT_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": payload["status"],
        "path": str(DEFAULT_OUTPUT),
        "sha256": sha256_file(DEFAULT_OUTPUT),
        "detector_checkpoints": len(payload["assets"]["d2_detector_checkpoints"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
