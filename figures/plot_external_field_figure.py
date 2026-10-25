#!/usr/bin/env python3
"""Build the four-tunnel field-evaluation figure.

The example image for each tunnel is selected without using any model outcome:
among the unique images in the frozen natural-width manifest, choose the image
whose recorded SHA-256 is lexicographically smallest.  The figure combines
those auditably selected images with the frozen tunnel-level detection and
apparent-width endpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parent
MODIFIED = ROOT.parent
CROSS_SECTIONS = (
    MODIFIED
    / "c014_existing_four_tunnel_validation"
    / "protocol"
    / "natural_width_cross_sections_v1.csv"
)
DETECTION_TUNNELS = (
    MODIFIED
    / "c055_factorial_main_effect_confirmation"
    / "runs"
    / "r3_3_tunnel_summary.csv"
)
WIDTH_RELEASE = (
    MODIFIED
    / "c016_merit_v24_development"
    / "runs"
    / "c016_release_v24_summary.json"
)
OUTPUT_PNG = ROOT / "fig14.png"
OUTPUT_PDF = ROOT / "fig14.pdf"
OUTPUT_PROVENANCE = ROOT / "fig14_provenance.json"
TUNNELS = ("A", "B", "C", "D")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def select_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_tunnel_and_hash: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_tunnel_and_hash[row["tunnel"]][row["image_sha256"]].append(row)

    selected: list[dict[str, str]] = []
    for tunnel in TUNNELS:
        image_hash = min(by_tunnel_and_hash[tunnel])
        image_rows = sorted(
            by_tunnel_and_hash[tunnel][image_hash], key=lambda row: row["roi_id"]
        )
        selected.append(image_rows[0])
    return selected


def verify_example(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])
    if sha256_file(image_path) != row["image_sha256"]:
        raise RuntimeError(f"Image hash mismatch: {image_path}")
    if sha256_file(mask_path) != row["mask_sha256"]:
        raise RuntimeError(f"Mask hash mismatch: {mask_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape != mask.shape:
        raise RuntimeError(f"Invalid image/mask pair: {image_path}")
    return image, mask


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


def plot_examples(
    axes: list[plt.Axes], selected: list[dict[str, str]]
) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    for axis, row in zip(axes, selected):
        image, mask = verify_example(row)
        center_x = float(row["center_x_px"])
        center_y = float(row["center_y_px"])
        # A tighter, outcome-independent crop keeps the frozen width chord and
        # mask boundary legible after the four panels are reduced to two-column width.
        crop_size = 192
        left = min(max(int(round(center_x)) - crop_size // 2, 0), image.shape[1] - crop_size)
        top = min(max(int(round(center_y)) - crop_size // 2, 0), image.shape[0] - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        image_crop = image[top:bottom, left:right]
        mask_crop = mask[top:bottom, left:right]
        axis.imshow(image_crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        if np.any(mask_crop > 0):
            axis.contour(
                mask_crop > 0,
                levels=[0.5],
                colors=["#00D4FF"],
                linewidths=1.55,
            )
        x1, y1, x2, y2 = (float(row[key]) for key in ("x1_px", "y1_px", "x2_px", "y2_px"))
        axis.plot(
            [x1 - left, x2 - left],
            [y1 - top, y2 - top],
            color="#FFD400",
            linewidth=2.0,
        )
        axis.scatter(
            [center_x - left],
            [center_y - top],
            s=20,
            color="#FFD400",
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#667085")
            spine.set_linewidth(1.2)
        provenance.append(
            {
                "tunnel": row["tunnel"],
                "parent": row["parent"],
                "image_id": row["image_id"],
                "image_sha256": row["image_sha256"],
                "mask_sha256": row["mask_sha256"],
                "displayed_roi_id": row["roi_id"],
                "reference_width_px": float(row["reference_width_px"]),
                "display_crop_xyxy_px": [left, top, right, bottom],
            }
        )
    return provenance


def plot_detection(axis: plt.Axes, rows: list[dict[str, str]]) -> dict[str, Any]:
    by_tunnel = {row["tunnel"]: row for row in rows}
    x = np.arange(len(TUNNELS), dtype=float)
    raw = np.array([float(by_tunnel[t]["raw_auprc"]) for t in TUNNELS])
    merit = np.array([float(by_tunnel[t]["complete_path_auprc"]) for t in TUNNELS])
    for position, raw_value, merit_value in zip(x, raw, merit):
        axis.plot(
            [position - 0.10, position + 0.10],
            [raw_value, merit_value],
            color="#98A2B3",
            linewidth=2.1,
            zorder=1,
        )
    axis.scatter(x - 0.10, raw, color="#667085", s=78, label="Raw", zorder=2)
    axis.scatter(x + 0.10, merit, color="#0057B8", s=78, label="MERIT", zorder=2)
    for position, raw_value, merit_value in zip(x, raw, merit):
        axis.text(
            position,
            max(raw_value, merit_value) + 0.022,
            f"+{merit_value - raw_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12.5,
            color="#0057B8",
        )
    axis.set_xticks(x, [f"Tunnel {t}" for t in TUNNELS])
    axis.set_ylabel("AUPRC")
    axis.set_ylim(0, max(0.68, float(merit.max()) + 0.10))
    axis.grid(axis="y", color="#E4E7EC", linewidth=1.0)
    axis.legend(frameon=False, ncol=2, loc="upper right", fontsize=12)
    axis.spines[["top", "right"]].set_visible(False)
    return {
        "raw_equal_tunnel_mean": float(raw.mean()),
        "merit_equal_tunnel_mean": float(merit.mean()),
        "delta_equal_tunnel_mean": float((merit - raw).mean()),
        "per_tunnel_delta": {
            tunnel: float(merit[index] - raw[index]) for index, tunnel in enumerate(TUNNELS)
        },
    }


def plot_width(axis: plt.Axes, release: dict[str, Any]) -> dict[str, Any]:
    endpoint = release["components"]["natural_crack_apparent_width"]
    summary = endpoint["delta_absolute_error_px"]
    values = np.array([float(summary["per_tunnel"][tunnel]) for tunnel in TUNNELS])
    x = np.arange(len(TUNNELS), dtype=float)
    bars = axis.bar(x, values, width=0.58, color="#3B82C4")
    axis.axhline(0, color="#344054", linewidth=1.35)
    axis.axhline(
        float(summary["mean"]),
        color="#B54708",
        linewidth=1.5,
        linestyle="--",
        label=f"Pooled: {float(summary['mean']):.3f} px",
    )
    for bar, value in zip(bars, values):
        if value < -0.10:
            label_y = value + 0.025
            label_va = "bottom"
            label_color = "white"
        else:
            label_y = value - 0.018
            label_va = "top"
            label_color = "#101828"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.3f}",
            ha="center",
            va=label_va,
            fontsize=12.5,
            color=label_color,
        )
    axis.set_xticks(x, [f"Tunnel {t}" for t in TUNNELS])
    axis.set_ylabel("$\\Delta$ absolute width error (px)\nMERIT $-$ raw")
    axis.set_ylim(min(-0.58, float(values.min()) - 0.08), 0.10)
    axis.grid(axis="y", color="#E4E7EC", linewidth=1.0)
    axis.legend(frameon=False, loc="upper right", fontsize=12)
    axis.spines[["top", "right"]].set_visible(False)
    return {
        "pooled_delta_absolute_error_px": float(summary["mean"]),
        "ci95": [float(summary["ci95_low"]), float(summary["ci95_high"])],
        "per_tunnel": {tunnel: float(summary["per_tunnel"][tunnel]) for tunnel in TUNNELS},
        "paired_estimable_rate": float(endpoint["paired_estimable_rate"]),
    }


def main() -> None:
    cross_sections = read_csv(CROSS_SECTIONS)
    selected = select_examples(cross_sections)
    detection_rows = read_csv(DETECTION_TUNNELS)
    release = json.loads(WIDTH_RELEASE.read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelcolor": "#344054",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.6, 8.1), constrained_layout=True)
    grid = figure.add_gridspec(
        4,
        4,
        height_ratios=[1.0, 0.14, 0.95, 0.14],
        hspace=0.04,
        wspace=0.12,
    )
    example_axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    provenance_examples = plot_examples(example_axes, selected)
    for index, row in enumerate(selected):
        add_bottom_subcaption(
            figure,
            grid[1, index],
            f"({chr(ord('a') + index)}) Tunnel {row['tunnel']}",
        )
    detection = plot_detection(figure.add_subplot(grid[2, :2]), detection_rows)
    width = plot_width(figure.add_subplot(grid[2, 2:]), release)
    add_bottom_subcaption(
        figure,
        grid[3, :2],
        "(e) Complete-path detection",
    )
    add_bottom_subcaption(
        figure,
        grid[3, 2:],
        "(f) Apparent-width absolute-error change",
    )
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    provenance = {
        "schema_version": "MERIT-R3-6-FIG14-v4",
        "selection_rule": (
            "For each tunnel, select the unique frozen-manifest image with the "
            "lexicographically smallest recorded image SHA-256; display the first ROI by roi_id. "
            "No image, detector, restoration, or width outcome is used for selection."
        ),
        "source_assets": {
            "cross_sections": {
                "path": str(CROSS_SECTIONS.relative_to(MODIFIED)),
                "sha256": sha256_file(CROSS_SECTIONS),
            },
            "detection_tunnels": {
                "path": str(DETECTION_TUNNELS.relative_to(MODIFIED)),
                "sha256": sha256_file(DETECTION_TUNNELS),
            },
            "width_release": {
                "path": str(WIDTH_RELEASE.relative_to(MODIFIED)),
                "sha256": sha256_file(WIDTH_RELEASE),
            },
        },
        "selected_examples": provenance_examples,
        "detection": detection,
        "apparent_width": width,
        "artifacts": {
            "fig14.png": sha256_file(OUTPUT_PNG),
            "fig14.pdf": sha256_file(OUTPUT_PDF),
        },
    }
    OUTPUT_PROVENANCE.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
