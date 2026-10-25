#!/usr/bin/env python3
"""Build Figures 11--13 from the frozen four-tunnel field evidence.

Figure 11 introduces the dataset with one outcome-blind full lining unfolding
per tunnel. Figure 12 visualises complete-path and held five-detector detection
effects. Figure 13 visualises apparent-width error, coverage, and prespecified
width-band heterogeneity. No example is selected using crack, restoration,
detector, or width outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parent
MODIFIED = ROOT.parent
TUNNELS = ("A", "B", "C", "D")
CROSS_SECTIONS = (
    MODIFIED
    / "c014_existing_four_tunnel_validation"
    / "protocol"
    / "natural_width_cross_sections_v1.csv"
)
COMPLETE_PATH = (
    MODIFIED
    / "c055_factorial_main_effect_confirmation"
    / "runs"
    / "r3_3_tunnel_summary.csv"
)
DETECTOR_SUMMARY = (
    MODIFIED
    / "c017_transformer_mamba_validation"
    / "runs"
    / "evaluation_v1"
    / "summary.json"
)
WIDTH_TUNNELS = (
    MODIFIED
    / "c015_reviewer_closure"
    / "r3"
    / "outputs"
    / "r3_6_width_tunnel_supplement.csv"
)
WIDTH_BANDS = (
    MODIFIED
    / "c015_reviewer_closure"
    / "r3"
    / "outputs"
    / "r3_6_width_band_supplement.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


SUBCAPTION_COLOR = "#101828"
SUBCAPTION_FONT_SIZE = 14.0


def add_bottom_subcaption(
    figure: plt.Figure,
    subplot_spec: Any,
    text: str,
    *,
    fontsize: float = SUBCAPTION_FONT_SIZE,
) -> plt.Axes:
    """Place a panel subcaption in a dedicated row below its panel."""
    axis = figure.add_subplot(subplot_spec)
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=SUBCAPTION_COLOR,
    )
    return axis


def dataset_root(rows: list[dict[str, str]]) -> Path:
    image_path = Path(rows[0]["image_path"])
    candidates = (image_path, *image_path.parents)
    return next(
        (candidate for candidate in candidates if candidate.name == "KICT_crack_dataset"),
        image_path.parents[4],
    )


def select_representative_unfoldings(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Select the central evaluated parent ID within each tunnel."""
    grouped: dict[str, set[str]] = {tunnel: set() for tunnel in TUNNELS}
    for row in rows:
        grouped[row["tunnel"]].add(row["parent"])
    root = dataset_root(rows)
    selected: list[dict[str, str]] = []
    for tunnel in TUNNELS:
        parents = sorted(grouped[tunnel])
        parent = parents[len(parents) // 2]
        path = root / "01.Bigeye_data" / "img" / f"{parent}.bmp"
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append(
            {
                "tunnel": tunnel,
                "parent": parent,
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "evaluated_parents": ",".join(parents),
            }
        )
    return selected


def load_unfolding_for_display(path: Path, target_width: int = 2400) -> tuple[np.ndarray, list[int]]:
    source = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Could not decode {path}")
    source_dimensions = [int(source.shape[1]), int(source.shape[0])]
    rotated = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
    scale = min(1.0, target_width / float(rotated.shape[1]))
    if scale < 1.0:
        rotated = cv2.resize(
            rotated,
            (target_width, int(round(rotated.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), source_dimensions


def plot_figure11() -> dict[str, Any]:
    rows = read_csv(CROSS_SECTIONS)
    selected = select_representative_unfoldings(rows)
    figure = plt.figure(figsize=(13.4, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 0.12, 1.0, 0.12],
        hspace=0.03,
        wspace=0.06,
    )
    unfoldings: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        panel_row = (index // 2) * 2
        panel_column = index % 2
        axis = figure.add_subplot(grid[panel_row, panel_column])
        image_path = Path(row["path"])
        display, source_dimensions = load_unfolding_for_display(image_path)
        axis.imshow(display, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#667085")
            spine.set_linewidth(0.8)
        add_bottom_subcaption(
            figure,
            grid[panel_row + 1, panel_column],
            f"({chr(ord('a') + index)}) Tunnel {row['tunnel']} · representative span {row['parent']}",
        )
        unfoldings.append(
            {
                "tunnel": row["tunnel"],
                "parent": row["parent"],
                "evaluated_parent_ids": row["evaluated_parents"].split(","),
                "source_relative_path": row["relative_path"],
                "source_sha256": sha256_file(image_path),
                "source_dimensions_px": source_dimensions,
                "display_rotation_degrees_clockwise": 90,
                "display_crop": "none",
            }
        )

    png = ROOT / "fig11.png"
    pdf = ROOT / "fig11.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    payload = {
        "schema_version": "MERIT-R3-6-FIG11-FOUR-TUNNEL-v4",
        "selection_rule": (
            "Within each tunnel, sort the evaluated physical parent identifiers and select "
            "the central identifier (upper median for an even count). Display the complete "
            "parent-level lining unfolding with a 90-degree clockwise display rotation and "
            "no spatial crop. No restoration, detector, crack, or width outcome is used."
        ),
        "source": {"path": str(CROSS_SECTIONS.relative_to(MODIFIED)), "sha256": sha256_file(CROSS_SECTIONS)},
        "selected_unfoldings": unfoldings,
        "claim_boundary": "Descriptive dataset context only; quantitative performance and expert annotations are reported in Figures 12--14 and Tables 5--6.",
        "artifacts": {"fig11.png": sha256_file(png), "fig11.pdf": sha256_file(pdf)},
    }
    write_json(ROOT / "fig11_provenance.json", payload)
    return payload


def add_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    cmap: str,
    limit: float,
    value_format: str,
) -> None:
    image = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(5), ["A", "B", "C", "D", "Mean"])
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_xlabel("Tunnel")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=7.6,
                color="white" if abs(value) > limit * 0.48 else "#101828",
            )
    axis.axvline(3.5, color="#101828", linewidth=1.1)
    colorbar = plt.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.ax.tick_params(labelsize=7)


def plot_figure12() -> dict[str, Any]:
    complete_rows = read_csv(COMPLETE_PATH)
    complete = {row["tunnel"]: row for row in complete_rows}
    detector_payload = json.loads(DETECTOR_SUMMARY.read_text(encoding="utf-8"))
    held = detector_payload["held_replication_seeds_1_2"]
    results = held["extended_five_detector_results"]
    detector_order = [
        "d2_resnet34_unet",
        "d2_vgg16_unet",
        "d2_deeplabv3_resnet50",
        "segformer_b0",
        "mamba2_unet",
    ]
    detector_labels = ["ResNet34 U-Net", "VGG16 U-Net", "DeepLabV3", "SegFormer-B0", "Mamba2 U-Net"]
    auprc = np.array(
        [
            [float(results[name]["delta_auprc"]["per_tunnel"][tunnel]) for tunnel in TUNNELS]
            + [float(results[name]["delta_auprc"]["mean"])]
            for name in detector_order
        ]
    ) * 100.0
    fpr = np.array(
        [
            [float(results[name]["delta_matched_fpr"]["per_tunnel"][tunnel]) for tunnel in TUNNELS]
            + [float(results[name]["delta_matched_fpr"]["mean"])]
            for name in detector_order
        ]
    ) * 1.0e4

    figure = plt.figure(figsize=(13.4, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[0.92, 0.11, 1.08, 0.11],
        hspace=0.04,
        wspace=0.10,
    )
    slope = figure.add_subplot(grid[0, :])
    x = np.arange(4, dtype=float)
    raw = np.array([float(complete[t]["raw_auprc"]) for t in TUNNELS])
    merit = np.array([float(complete[t]["complete_path_auprc"]) for t in TUNNELS])
    for index in range(4):
        slope.plot(
            [x[index] - 0.12, x[index] + 0.12],
            [raw[index], merit[index]],
            color="#98A2B3",
            linewidth=1.6,
        )
        slope.text(
            x[index],
            max(raw[index], merit[index]) + 0.018,
            f"+{(merit[index] - raw[index]) * 100:.2f} pp",
            ha="center",
            fontsize=8.5,
            color="#0057B8",
            fontweight="bold",
        )
    slope.scatter(x - 0.12, raw, s=55, color="#667085", label="Raw bypass", zorder=3)
    slope.scatter(x + 0.12, merit, s=55, color="#0057B8", label="Complete MERIT path", zorder=3)
    slope.set_xticks(x, [f"Tunnel {tunnel}" for tunnel in TUNNELS])
    slope.set_ylabel("AUPRC")
    slope.set_ylim(0, max(0.68, float(merit.max()) + 0.09))
    slope.grid(axis="y", color="#E4E7EC", linewidth=0.7)
    slope.legend(frameon=False, ncol=2, loc="upper right")
    slope.spines[["top", "right"]].set_visible(False)
    add_bottom_subcaption(
        figure,
        grid[1, :],
        "(a) Complete-path natural-crack detection",
        fontsize=SUBCAPTION_FONT_SIZE,
    )

    auprc_axis = figure.add_subplot(grid[2, 0])
    add_heatmap(
        auprc_axis,
        auprc,
        detector_labels,
        "RdBu",
        4.2,
        ".3f",
    )
    fpr_axis = figure.add_subplot(grid[2, 1])
    add_heatmap(
        fpr_axis,
        fpr,
        detector_labels,
        "RdBu_r",
        6.0,
        ".2f",
    )
    add_bottom_subcaption(
        figure,
        grid[3, 0],
        "(b) Frozen-confirmation $\\Delta$AUPRC (percentage points)",
        fontsize=SUBCAPTION_FONT_SIZE,
    )
    add_bottom_subcaption(
        figure,
        grid[3, 1],
        "(c) Frozen matched-sensitivity $\\Delta$FPR ($\\times10^{-4}$)",
        fontsize=SUBCAPTION_FONT_SIZE,
    )
    png = ROOT / "fig12.png"
    pdf = ROOT / "fig12.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    payload = {
        "schema_version": "MERIT-R3-6-FIG12-FOUR-TUNNEL-v3",
        "sources": {
            "complete_path": {"path": str(COMPLETE_PATH.relative_to(MODIFIED)), "sha256": sha256_file(COMPLETE_PATH)},
            "detector_summary": {"path": str(DETECTOR_SUMMARY.relative_to(MODIFIED)), "sha256": sha256_file(DETECTOR_SUMMARY)},
        },
        "coverage": {
            "tiles": 1703,
            "physical_parents": 14,
            "span_units_per_tunnel": {
                tunnel: int(complete[tunnel]["parent_count"]) for tunnel in TUNNELS
            },
            "tunnels": 4,
            "held_seeds": [1, 2],
        },
        "complete_path": {
            "raw": {tunnel: float(raw[index]) for index, tunnel in enumerate(TUNNELS)},
            "merit": {tunnel: float(merit[index]) for index, tunnel in enumerate(TUNNELS)},
            "equal_tunnel_delta": float(np.mean(merit - raw)),
        },
        "held_detector_order": detector_order,
        "held_delta_auprc_percentage_points": auprc.tolist(),
        "held_delta_matched_fpr_times_1e4": fpr.tolist(),
        "claim_boundary": "Four-tunnel revision-stage field evidence; detector families and seeds do not create additional independent sites.",
        "artifacts": {"fig12.png": sha256_file(png), "fig12.pdf": sha256_file(pdf)},
    }
    write_json(ROOT / "fig12_provenance.json", payload)
    return payload


def plot_figure13() -> dict[str, Any]:
    tunnel_rows = read_csv(WIDTH_TUNNELS)
    band_rows = read_csv(WIDTH_BANDS)
    tunnels = {row["tunnel"]: row for row in tunnel_rows}
    tunnel_delta = np.array([float(tunnels[t]["hierarchical_delta_absolute_error_px"]) for t in TUNNELS])
    coverage = np.array([float(tunnels[t]["paired_estimable_rate"]) * 100.0 for t in TUNNELS])
    band_labels = [row["expert_mask_width_band"].replace(">=", "≥") for row in band_rows]
    band_delta = np.array([float(row["paired_roi_mean_delta_absolute_error_px"]) for row in band_rows])
    pooled_delta = -0.30722220269698347
    pooled_coverage = 70.10406811731315

    figure = plt.figure(figsize=(13.4, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.12],
        hspace=0.04,
        wspace=0.12,
    )
    axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    x = np.arange(4)
    colors = ["#2E90FA" if value < 0 else "#D92D20" for value in tunnel_delta]
    axes[0].bar(x, tunnel_delta, color=colors, width=0.62)
    axes[0].axhline(0, color="#344054", linewidth=0.9)
    axes[0].axhline(pooled_delta, color="#B54708", linestyle="--", linewidth=1.1, label=f"Pooled {pooled_delta:.3f} px")
    for index, value in enumerate(tunnel_delta):
        axes[0].text(index, value - 0.025, f"{value:.3f}", ha="center", va="top", fontsize=8, color="white" if value < -0.1 else "#101828")
    axes[0].set_xticks(x, [f"Tunnel {t}" for t in TUNNELS], rotation=15)
    axes[0].set_ylabel("$\\Delta$ absolute width error (px)\nMERIT $-$ raw")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    axes[1].bar(x, coverage, color="#7F56D9", width=0.62)
    axes[1].axhline(pooled_coverage, color="#B54708", linestyle="--", linewidth=1.1, label=f"Pooled {pooled_coverage:.1f}%")
    for index, value in enumerate(coverage):
        axes[1].text(index, value + 1.5, f"{value:.1f}%", ha="center", fontsize=8)
    axes[1].set_xticks(x, [f"Tunnel {t}" for t in TUNNELS], rotation=15)
    axes[1].set_ylim(0, 90)
    axes[1].set_ylabel("Paired estimable regions (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    band_x = np.arange(len(band_labels))
    band_colors = ["#2E90FA" if value < 0 else "#D92D20" for value in band_delta]
    axes[2].bar(band_x, band_delta, color=band_colors, width=0.62)
    axes[2].axhline(0, color="#344054", linewidth=0.9)
    for index, value in enumerate(band_delta):
        axes[2].text(index, value - 0.05 if value < 0 else value + 0.05, f"{value:+.3f}", ha="center", va="top" if value < 0 else "bottom", fontsize=8, color="white" if value < -0.25 else "#101828")
    axes[2].set_xticks(band_x, band_labels, rotation=15)
    axes[2].set_ylabel("Mean $\\Delta$ absolute error (px)")
    for column, caption in enumerate(
        (
            "(a) Paired apparent-width error",
            "(b) Measurement coverage",
            "(c) Expert-mask width-band heterogeneity",
        )
    ):
        add_bottom_subcaption(
            figure,
            grid[1, column],
            caption,
            fontsize=SUBCAPTION_FONT_SIZE,
        )

    for axis in axes:
        axis.grid(axis="y", color="#E4E7EC", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    png = ROOT / "fig13.png"
    pdf = ROOT / "fig13.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    payload = {
        "schema_version": "MERIT-R3-6-FIG13-FOUR-TUNNEL-v3",
        "sources": {
            "tunnel_summary": {"path": str(WIDTH_TUNNELS.relative_to(MODIFIED)), "sha256": sha256_file(WIDTH_TUNNELS)},
            "width_band_summary": {"path": str(WIDTH_BANDS.relative_to(MODIFIED)), "sha256": sha256_file(WIDTH_BANDS)},
        },
        "coverage": {"mask_defined_regions": 1057, "paired_estimable_regions": 741, "paired_estimable_rate": pooled_coverage / 100.0, "tunnels": 4},
        "per_tunnel_delta_absolute_error_px": {tunnel: float(tunnel_delta[index]) for index, tunnel in enumerate(TUNNELS)},
        "per_tunnel_paired_estimable_rate": {tunnel: float(coverage[index] / 100.0) for index, tunnel in enumerate(TUNNELS)},
        "pooled_delta_absolute_error_px": pooled_delta,
        "width_band_delta_absolute_error_px": {band_labels[index]: float(band_delta[index]) for index in range(len(band_labels))},
        "claim_boundary": "Apparent width in expert-mask-referenced pixels on the paired estimable subset; not in-situ physical width in millimetres or a whole-dataset guarantee.",
        "artifacts": {"fig13.png": sha256_file(png), "fig13.pdf": sha256_file(pdf)},
    }
    write_json(ROOT / "fig13_provenance.json", payload)
    return payload


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#344054",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    outputs = [plot_figure11(), plot_figure12(), plot_figure13()]
    print(json.dumps({payload["schema_version"]: payload["artifacts"] for payload in outputs}, indent=2))


if __name__ == "__main__":
    main()
