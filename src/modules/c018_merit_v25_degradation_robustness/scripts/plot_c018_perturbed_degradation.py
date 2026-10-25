#!/usr/bin/env python3
"""Build the auditable C018 perturbed-degradation summary figure.

The heatmaps use all 18 frozen family--severity cells.  The qualitative panel
uses one frozen confirmation crop selected from the source image only, before
any degraded or restored output is loaded.  It is illustrative and carries no
inferential weight.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from matplotlib.colors import Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402
import c018_evaluate_confirmation as confirmation  # noqa: E402


SEVERITY = 2
SELECTED_BANK_INDEX = 80
DISPLAY_GAMMA = 0.45
QUALITATIVE_ROI = (0, 224, 0, 224)  # y0, y1, x0, x1; fixed from the source crop only
OUTPUT_ROOT = common.C018 / "runs" / "figure15_v2"
MANUSCRIPT_R3 = common.REVISION / "manuscript_r3_6"
MANUSCRIPT_R2 = common.REVISION / "manuscript_r2_4"
SAMPLE_METRICS = common.C018 / "runs" / "input_reference_audit_v1" / "sample_metrics.jsonl"
C061 = common.REVISION / "c061_motion_deblur_development"
CELL_SUMMARY = C061 / "runs" / "integrated_r1_1_v1" / "cell_summary.csv"
C061_CONFIRMATION = C061 / "runs" / "confirmation_v1" / "RUN_COMPLETE.json"
C061_SAMPLE_METRICS = C061 / "runs" / "confirmation_v1" / "sample_metrics.jsonl"
C061_SCANNER = C061 / "scripts" / "c061_scan_motion.py"

FAMILIES = [
    ("illumination_gradient_vignetting", "Illumination / vignetting"),
    ("linear_motion_psf", "Linear-motion PSF"),
    ("poisson_gaussian", "Poisson--Gaussian noise"),
    ("saturation_clipping", "Saturation / clipping"),
    ("spatially_varying_blur", "Spatially varying blur"),
    ("veiling_scattering", "Veiling / scattering"),
]


def load_audit_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with SAMPLE_METRICS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["sample_id"])] = record
    if len(records) != 5292:
        raise RuntimeError(f"Expected 5,292 audit records, found {len(records)}")
    return records


def load_c061_scanner() -> Any:
    specification = importlib.util.spec_from_file_location(
        "c061_scanner_figure15", C061_SCANNER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import C061 scanner: {C061_SCANNER}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_c061_motion_records() -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    with C061_SAMPLE_METRICS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[(int(record["bank_index"]), int(record["severity"]))] = record
    if len(records) != 294:
        raise RuntimeError(f"Expected 294 C061 confirmation records, found {len(records)}")
    return records


def load_cell_arrays() -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    with CELL_SUMMARY.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["condition"], int(row["severity"])): row for row in rows}
    psnr = np.empty((len(FAMILIES), 3), dtype=np.float64)
    lpips_reduction = np.empty_like(psnr)
    ordered: list[dict[str, str]] = []
    for family_index, (family, _label) in enumerate(FAMILIES):
        for severity in (1, 2, 3):
            row = lookup[(family, severity)]
            ordered.append(row)
            psnr[family_index, severity - 1] = float(row["delta_psnr_vs_input_mean"])
            lpips_reduction[family_index, severity - 1] = -float(
                row["delta_lpips_vs_input_mean"]
            )
    return psnr, lpips_reduction, ordered


def build_selected_outputs() -> tuple[np.ndarray, dict[str, dict[str, Any]], dict[str, str]]:
    freeze = common.load_json(confirmation.FREEZE)
    physics = common.load_json(confirmation.PHYSICS_SCAN)
    evaluation = common.import_file(
        "c009_controlled_eval_c018_figure15",
        common.C009 / "scripts" / "c009_controlled_eval.py",
    )
    bank, rows = confirmation.load_source()
    row = rows[SELECTED_BANK_INDEX]
    if int(row["bank_index"]) != SELECTED_BANK_INDEX or row["role"] != "crack":
        raise RuntimeError("The frozen qualitative selection is not the expected crack-role crop")
    clean = bank[SELECTED_BANK_INDEX]
    seed = int(min(freeze["stochastic_seeds"]))
    family_order = list(freeze["families"].keys())
    audit = load_audit_records()
    c061_records = load_c061_motion_records()
    c061_result = common.load_json(C061_CONFIRMATION)
    c061_scanner = load_c061_scanner()

    samples: dict[str, dict[str, Any]] = {}
    for family, _label in FAMILIES:
        family_index = family_order.index(family)
        parameters = freeze["families"][family][SEVERITY - 1]
        rng = evaluation.deterministic_rng(
            seed, SELECTED_BANK_INDEX, 500 + family_index * 10 + SEVERITY
        )
        degraded = evaluation.ood_degrade(clean, family, parameters, rng)
        sample_id = (
            f"confirm__b{SELECTED_BANK_INDEX}__{family}__s{SEVERITY}__n{seed}"
        )
        samples[family] = {
            "sample_id": sample_id,
            "clean": clean,
            "degraded": degraded,
            "parameters": parameters,
            "audit": audit[sample_id],
        }

    motion_record = c061_records[(SELECTED_BANK_INDEX, SEVERITY)]
    samples["linear_motion_psf"]["audit"] = {
        "delta_psnr_candidate_minus_degraded": motion_record["delta_psnr_db"],
        "delta_lpips_candidate_minus_degraded": motion_record["delta_lpips"],
    }

    device = torch.device("cuda")
    model, _provenance = evaluation.common.build_and_load_model(
        {"kind": "wp1", "family": "nafnet", "arm": "base", "seed": 0}, device
    )
    checkpoint = torch.load(
        confirmation.POISSON_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    # Reproduce the original 16-sample inference batch.  BF16 convolution can
    # differ slightly with batch shape, so a one-image inference is not an
    # exact reconstruction of the frozen audit output.
    all_samples = confirmation.make_samples(bank, rows, freeze, evaluation)
    poisson_indices = [
        index
        for index, sample in enumerate(all_samples)
        if sample["condition"] == "poisson_gaussian"
    ]
    selected_poisson_id = samples["poisson_gaussian"]["sample_id"]
    selected_index = next(
        index for index in poisson_indices if all_samples[index]["sample_id"] == selected_poisson_id
    )
    position = poisson_indices.index(selected_index)
    batch_start = (position // 16) * 16
    batch_indices = poisson_indices[batch_start : batch_start + 16]
    poisson_output = confirmation.infer_model(
        model,
        all_samples,
        batch_indices,
        device,
    )[selected_index]

    for family, _label in FAMILIES:
        sample = samples[family]
        if family == "poisson_gaussian":
            candidate = poisson_output
        elif family == "linear_motion_psf":
            route = c061_result["cells"][str(SEVERITY)]
            kernel = common.motion_kernel(
                int(route["parameters"]["length_px"]),
                float(route["parameters"]["angle_deg"]),
            )
            estimate = c061_scanner.estimate(
                sample["degraded"], kernel, route["specification"]
            )
            candidate = c061_scanner.blend(
                sample["degraded"],
                estimate,
                float(route["specification"]["alpha"]),
            )
        else:
            scan = physics["results"][family][str(SEVERITY)]
            candidate = common.restore_physics(
                sample["degraded"],
                family,
                sample["parameters"],
                float(scan["selected_gain"]),
            )
        sample["candidate"] = candidate
        stored = sample["audit"]
        degraded_psnr = common.psnr(clean, sample["degraded"])
        candidate_psnr = common.psnr(clean, candidate)
        if family == "linear_motion_psf":
            delta_psnr = candidate_psnr - degraded_psnr
            expected = float(stored["delta_psnr_candidate_minus_degraded"])
            if abs(delta_psnr - expected) > 2e-5:
                raise RuntimeError(
                    f"C061 motion reconstruction mismatch: {delta_psnr:.9f} != {expected:.9f}"
                )
        else:
            if abs(degraded_psnr - float(stored["degraded_psnr_db"])) > 1e-6:
                raise RuntimeError(f"Degraded reconstruction mismatch for {family}")
            if abs(candidate_psnr - float(stored["candidate_psnr_db"])) > 2e-5:
                raise RuntimeError(
                    f"Candidate reconstruction mismatch for {family}: "
                    f"{candidate_psnr:.9f} != {float(stored['candidate_psnr_db']):.9f}"
                )
    motion = samples["linear_motion_psf"]
    if np.array_equal(motion["degraded"], motion["candidate"]):
        raise RuntimeError("C061 linear-motion panel unexpectedly remained an exact bypass")

    provenance = {
        "bank_index": row["bank_index"],
        "parent_sha256": row["parent_sha256"],
        "crop_sha256": row["crop_sha256"],
        "condition": row["condition"],
        "role": row["role"],
        "stochastic_seed": str(seed),
    }
    return clean, samples, provenance


def display_image(image: np.ndarray) -> np.ndarray:
    return np.power(np.clip(image, 0.0, 1.0), DISPLAY_GAMMA)


def qualitative_roi(image: np.ndarray) -> np.ndarray:
    """Return the fixed 2x crack-centred view used only for visual presentation."""
    y0, y1, x0, x1 = QUALITATIVE_ROI
    return image[y0:y1, x0:x1]


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    cmap: str,
    vmax: float,
    formatter: str,
    show_ylabels: bool = True,
) -> Any:
    image = ax.imshow(values, cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax), aspect="auto")
    ax.set_xticks(range(3), ["Severity 1", "Severity 2", "Severity 3"], fontsize=8.5)
    if show_ylabels:
        ax.set_yticks(
            range(len(FAMILIES)),
            [label for _family, label in FAMILIES],
            fontsize=8.2,
        )
    else:
        ax.set_yticks([])
    threshold = vmax * 0.52
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = format(value, formatter)
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value > threshold else "black",
                fontweight="normal",
            )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def create_figure(clean: np.ndarray, samples: dict[str, dict[str, Any]]) -> plt.Figure:
    psnr, lpips_reduction, _rows = load_cell_arrays()
    figure = plt.figure(figsize=(12.2, 13.5), constrained_layout=False)
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[1.72, 1.0, 1.0, 1.0],
        hspace=0.26,
        wspace=0.16,
        left=0.16,
        right=0.975,
        top=0.978,
        bottom=0.055,
    )

    psnr_grid = grid[0, 0].subgridspec(
        2, 1, height_ratios=[1.0, 0.13], hspace=0.14
    )
    lpips_grid = grid[0, 1].subgridspec(
        2, 1, height_ratios=[1.0, 0.13], hspace=0.14
    )
    psnr_ax = figure.add_subplot(psnr_grid[0, 0])
    lpips_ax = figure.add_subplot(lpips_grid[0, 0])
    psnr_image = annotate_heatmap(
        psnr_ax,
        psnr,
        "YlGnBu",
        float(np.max(psnr)),
        "+.1f",
    )
    lpips_image = annotate_heatmap(
        lpips_ax,
        lpips_reduction,
        "YlOrRd",
        float(np.max(lpips_reduction)),
        ".3f",
        show_ylabels=False,
    )
    cbar_psnr = figure.colorbar(psnr_image, ax=psnr_ax, fraction=0.035, pad=0.025)
    cbar_psnr.ax.tick_params(labelsize=7.5)
    cbar_lpips = figure.colorbar(lpips_image, ax=lpips_ax, fraction=0.035, pad=0.025)
    cbar_lpips.ax.tick_params(labelsize=7.5)
    for caption_grid, caption in (
        (psnr_grid, "(a) Mean PSNR change vs unrestored input (dB)"),
        (lpips_grid, "(b) Mean LPIPS reduction vs unrestored input"),
    ):
        caption_ax = figure.add_subplot(caption_grid[1, 0])
        caption_ax.axis("off")
        caption_ax.text(
            0.5,
            0.45,
            caption,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
        )

    panel_letters = ["(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for family_index, ((family, label), letter) in enumerate(zip(FAMILIES, panel_letters)):
        row = 1 + family_index // 2
        column = family_index % 2
        panel_grid = grid[row, column].subgridspec(
            2,
            3,
            height_ratios=[1.0, 0.25],
            hspace=0.06,
            wspace=0.10,
        )
        sample = samples[family]
        stored = sample["audit"]
        triplet = [
            qualitative_roi(clean),
            qualitative_roi(sample["degraded"]),
            qualitative_roi(sample["candidate"]),
        ]
        displayed_triplet = [display_image(image) for image in triplet]
        display_errors = [
            np.abs(displayed_triplet[1] - displayed_triplet[0]),
            np.abs(displayed_triplet[2] - displayed_triplet[0]),
        ]
        error_vmax = max(
            float(np.percentile(np.concatenate([error.ravel() for error in display_errors]), 99.0)),
            1e-6,
        )
        caption_ax = figure.add_subplot(panel_grid[1, :])
        caption_ax.axis("off")
        subcaption = f"{letter} {label} (severity {SEVERITY})"
        caption_ax.text(
            0.5,
            0.50,
            subcaption,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
        titles = ["Reference", "Degraded input", "MERIT output"]
        for offset, (image, title) in enumerate(zip(displayed_triplet, titles)):
            ax = figure.add_subplot(panel_grid[0, offset])
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=7.4, pad=3)
            border = "#1f77b4" if offset == 2 else ("#d97706" if offset == 1 else "#6b7280")
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)
                spine.set_edgecolor(border)
            if offset > 0:
                inset = ax.inset_axes([0.58, 0.04, 0.38, 0.38])
                inset.imshow(
                    display_errors[offset - 1],
                    cmap="magma",
                    vmin=0.0,
                    vmax=error_vmax,
                )
                inset.set_xticks([])
                inset.set_yticks([])
                inset.text(
                    0.04,
                    0.05,
                    r"$|\Delta_{ref}|$",
                    transform=inset.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=4.8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 0.8, "edgecolor": "none"},
                )
                for spine in inset.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor("white")

    figure.text(
        0.5,
        0.014,
        (
            "Qualitative panels show the same source-selected top-left 224 x 224 crack ROI at 2x magnification; "
            "insets show absolute display-domain difference from the reference on a shared within-family scale; "
            "stochastic families use one fixed seed; a display-only "
            f"gamma of {DISPLAY_GAMMA:.2f} is applied identically. Metrics use the full untransformed arrays."
        ),
        ha="center",
        fontsize=8.2,
        color="#374151",
    )
    return figure


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the Poisson--Gaussian MERIT panel")
    clean, samples, selection = build_selected_outputs()
    figure = create_figure(clean, samples)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_ROOT / "fig15_perturbed_degradation.png"
    pdf = OUTPUT_ROOT / "fig15_perturbed_degradation.pdf"
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)

    destinations = []
    for directory in (MANUSCRIPT_R3, MANUSCRIPT_R2):
        directory.mkdir(parents=True, exist_ok=True)
        for source, name in ((png, "fig15.png"), (pdf, "fig15.pdf")):
            target = directory / name
            shutil.copy2(source, target)
            destinations.append(str(target.resolve()))

    result = {
        "schema_version": "MERIT-C018-FIGURE15-v2",
        "selection_status": "SOURCE_ONLY_QUALITATIVE_SELECTION_NO_OUTPUT_SCREENING",
        "selection_rule": (
            "One clearly visible crack crop was selected from frozen role=crack source crops "
            "using the operational-source image only; no degraded or restored output was viewed. "
            "The same crop and severity are used for all six families; one seed is fixed for the stochastic families."
        ),
        "selection": selection,
        "severity": SEVERITY,
        "display_gamma": DISPLAY_GAMMA,
        "qualitative_roi_y0_y1_x0_x1": list(QUALITATIVE_ROI),
        "qualitative_zoom_factor": 2.0,
        "qualitative_roi_selected_from": "SOURCE_IMAGE_ONLY",
        "qualitative_error_insets": (
            "Absolute display-domain difference from the reference; degraded and restored "
            "insets share the same family-specific 99th-percentile scale."
        ),
        "metrics_use_untransformed_arrays": True,
        "families": [family for family, _label in FAMILIES],
        "cell_summary_sha256": common.sha256_file(CELL_SUMMARY),
        "sample_metrics_sha256": common.sha256_file(SAMPLE_METRICS),
        "c061_confirmation_sha256": common.sha256_file(C061_CONFIRMATION),
        "c061_sample_metrics_sha256": common.sha256_file(C061_SAMPLE_METRICS),
        "freeze_sha256": common.sha256_file(confirmation.FREEZE),
        "script_sha256": common.sha256_file(Path(__file__).resolve()),
        "png_sha256": common.sha256_file(png),
        "pdf_sha256": common.sha256_file(pdf),
        "destinations": destinations,
    }
    provenance = OUTPUT_ROOT / "fig15_provenance.json"
    common.write_json_atomic(provenance, result)
    shutil.copy2(provenance, MANUSCRIPT_R3 / "fig15_provenance.json")
    shutil.copy2(provenance, MANUSCRIPT_R2 / "fig15_provenance.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
