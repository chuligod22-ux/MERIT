#!/usr/bin/env python3
"""Build outcome-blind qualitative Figures 8 and 9 from the controlled dataset.

The frozen file ordering remains in provenance but is not shown inside the
publication figures.  Each displayed zoom is selected deterministically from
the operational source alone and then reused unchanged for every method.
Residual insets use a common scale within each figure so subtle departures
from the operational source remain visible without outcome-based cropping.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
BATTERY_SCRIPT = BASE / "scripts" / "c063_full737_battery.py"
MANIFEST = BASE / "protocol" / "full737_battery_manifest_v1.csv"
MANUSCRIPT = ROOT / "05_MERIT_revision" / "40_수정본" / "manuscript_r3_6"
FIG8_PNG = MANUSCRIPT / "fig8.png"
FIG8_PDF = MANUSCRIPT / "fig8.pdf"
FIG8_PROVENANCE = MANUSCRIPT / "fig8_provenance.json"
FIG9_PNG = MANUSCRIPT / "fig9.png"
FIG9_PDF = MANUSCRIPT / "fig9.pdf"
FIG9_PROVENANCE = MANUSCRIPT / "fig9_provenance.json"

ANCHOR_CONDITIONS = (
    "crack_d25_ISO400_V60",
    "crack_d25_ISO800_V60",
    "crack_d45_ISO400_V60",
    "crack_d65_ISO1600_V80",
)

DISPLAY_NAMES = {
    "source": "Operational source",
    "degraded": "Common degraded input",
    "wiener_oracle": "Wiener (oracle PSF)",
    "richardson_lucy_oracle": "Richardson–Lucy",
    "unsupervised_wiener_oracle": "Unsupervised Wiener",
    "wiener_wrong_psf": "Wiener (wrong PSF)",
    "tv_denoise": "TV denoise",
    "nl_means": "NL-means",
    "unsharp": "Unsharp mask",
    "nafnet_fidelity": "Fidelity NAFNet",
    "merit_core_adapter": "MERIT core",
    "merit_routed_v3i": "Final routed path",
}

SUBCAPTION_COLOR = "#101828"
FIG8_SUBCAPTION_FONT_SIZE = 14.0
FIG9_SUBCAPTION_FONT_SIZE = 11.5


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recreate(row: pd.Series, battery, runtime: tuple[Any, ...]) -> dict[str, Any]:
    models, inference_module, adapter_module, candidate, gate, device = runtime
    image_path = Path(str(row["image_path"]))
    if sha256_file(image_path) != str(row["image_sha256"]):
        raise RuntimeError(f"input hash mismatch: {image_path}")
    bgr = battery.read_bgr(image_path)
    crop_rows = int(np.floor(bgr.shape[0] * battery.TOP_FRACTION))
    crop_y0 = (crop_rows - battery.CROP_HEIGHT) // 2
    crop_x0 = (bgr.shape[1] - battery.CROP_WIDTH) // 2
    source = cv2.cvtColor(
        bgr[
            crop_y0 : crop_y0 + battery.CROP_HEIGHT,
            crop_x0 : crop_x0 + battery.CROP_WIDTH,
        ],
        cv2.COLOR_BGR2GRAY,
    )
    degraded, degradation_seed = battery.deterministic_degrade(
        source, str(row["image_sha256"])
    )
    outputs: dict[str, np.ndarray] = {"source": source, "degraded": degraded}
    for method in battery.METHODS[1:8]:
        outputs[method] = battery.restore_classical(method, degraded)
    outputs["nafnet_fidelity"] = battery.deep_restore(
        degraded, models["nafnet_fidelity"], inference_module, device
    )
    merit_raw = battery.deep_restore(degraded, models["merit"], inference_module, device)
    merit_active, _diagnostics = adapter_module.apply_safe_adapter(
        degraded, merit_raw, candidate
    )
    outputs["merit_core_adapter"] = merit_active

    gate_key = (int(row["speed_kmh"]), float(row["distance_m"]), int(row["iso"]))
    gate_active = bool(gate.loc[gate_key, "route_active"])
    manifest_active = bool(int(row["route_active"]))
    if gate_active != manifest_active:
        raise RuntimeError(f"route mismatch for {gate_key}")
    outputs["merit_routed_v3i"] = merit_active if gate_active else degraded.copy()

    progress_path = battery.progress_path(
        int(row["ordinal"]), str(row["condition_cell"]), str(row["frame"])
    )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    hash_matches: dict[str, bool] = {}
    for method, output in outputs.items():
        actual = battery.sha256_array(output)
        expected = progress["output_hashes"][method]
        hash_matches[method] = actual == expected
        if actual != expected and method != "unsupervised_wiener_oracle":
            raise RuntimeError(
                f"regenerated output hash mismatch: {row['condition_cell']} {method}"
            )
    return {
        "row": row,
        "outputs": outputs,
        "metrics": progress["metrics"],
        "progress_path": progress_path,
        "progress_sha256": sha256_file(progress_path),
        "degradation_seed": degradation_seed,
        "route_active": gate_active,
        "hash_matches": hash_matches,
    }


def error_text(record: dict[str, Any], method: str) -> str:
    if not record["hash_matches"].get(method, False):
        return "same-parameter stochastic rerun"
    source_mean = float(record["metrics"]["source"]["mean_probability"])
    method_mean = float(record["metrics"][method]["mean_probability"])
    if method == "source":
        return "response reference"
    return rf"$|\Delta\mathrm{{response}}|={abs(method_mean-source_mean)*1e4:.2f}\times10^{{-4}}$"


def style_axis(axis, image: np.ndarray, border: str) -> None:
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(border)
        spine.set_linewidth(1.8)


def source_defined_zoom_box(source: np.ndarray) -> tuple[int, int, int, int]:
    """Select a reproducible high-structure crop using the source image only."""
    height, width = source.shape
    zoom_height = min(384, height)
    zoom_width = min(576, width)
    y_candidates = np.linspace(0, height - zoom_height, 3, dtype=int)
    x_candidates = np.linspace(0, width - zoom_width, 3, dtype=int)
    source_float = source.astype(np.float32)
    gradient_x = cv2.Sobel(source_float, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(source_float, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.hypot(gradient_x, gradient_y)
    candidates: list[tuple[float, int, int]] = []
    for y0 in y_candidates:
        for x0 in x_candidates:
            score = float(energy[y0 : y0 + zoom_height, x0 : x0 + zoom_width].mean())
            candidates.append((score, int(y0), int(x0)))
    _score, y0, x0 = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
    return y0, x0, zoom_height, zoom_width


def crop_to_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, height, width = box
    return image[y0 : y0 + height, x0 : x0 + width]


def common_residual_scale(
    records: list[dict[str, Any]], methods: tuple[str, ...]
) -> float:
    residuals: list[np.ndarray] = []
    for record in records:
        box = record["zoom_box"]
        source = crop_to_box(record["outputs"]["source"], box).astype(np.float32)
        for method in methods:
            if method == "source":
                continue
            output = crop_to_box(record["outputs"][method], box).astype(np.float32)
            residuals.append(np.abs(output - source).ravel())
    combined = np.concatenate(residuals)
    return max(1.0, float(np.percentile(combined, 99.0)))


def add_residual_inset(
    axis,
    source: np.ndarray,
    output: np.ndarray,
    residual_vmax: float,
) -> None:
    inset = axis.inset_axes([0.675, 0.055, 0.30, 0.30])
    residual = np.abs(output.astype(np.float32) - source.astype(np.float32))
    inset.imshow(
        residual,
        cmap="magma",
        vmin=0.0,
        vmax=residual_vmax,
        interpolation="nearest",
    )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.9)
    inset.set_title(
        r"$|I-I_{src}|$",
        fontsize=5.8,
        color="#111827",
        pad=1.0,
        bbox={"facecolor": "white", "alpha": 0.78, "pad": 0.8, "edgecolor": "none"},
    )


def condition_label(row: pd.Series, route_active: bool) -> str:
    route = "active restoration" if route_active else "exact preservation"
    distance = f"{float(row['distance_m']):g} m"
    return f"{distance} | ISO {int(row['iso'])} | {int(row['speed_kmh'])} km/h\n{route}"


def flatten_png(path: Path) -> None:
    """Store an RGB PNG so Markdown viewers and review tools decode it consistently."""
    with Image.open(path) as image:
        if image.mode == "RGB":
            return
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        background.convert("RGB").save(path, optimize=True)


def plot_figure8(records: list[dict[str, Any]]) -> None:
    methods = ("source", "degraded", "merit_core_adapter", "merit_routed_v3i")
    colours = ("#374151", "#B42318", "#1769AA", "#087F5B")
    subcaptions = (
        "(a) Operational source",
        "(b) Degraded input",
        "(c) MERIT core",
        "(d) Preservation-gated path",
    )
    for record in records:
        record["zoom_box"] = source_defined_zoom_box(record["outputs"]["source"])
    residual_vmax = common_residual_scale(records, methods)
    for record in records:
        record["display_residual_vmax_dn"] = residual_vmax
    fig, axes = plt.subplots(len(records), len(methods), figsize=(15.6, 10.8))
    for row_index, record in enumerate(records):
        row = record["row"]
        box = record["zoom_box"]
        source_zoom = crop_to_box(record["outputs"]["source"], box)
        for column_index, (method, colour) in enumerate(zip(methods, colours)):
            axis = axes[row_index, column_index]
            output_zoom = crop_to_box(record["outputs"][method], box)
            style_axis(axis, output_zoom, colour)
            if method != "source":
                add_residual_inset(axis, source_zoom, output_zoom, residual_vmax)
            axis.text(
                0.02,
                0.97,
                error_text(record, method),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.9,
                color="white",
                bbox={"facecolor": "#111827", "alpha": 0.80, "pad": 2.3, "edgecolor": "none"},
            )
            if column_index == 0:
                axis.set_ylabel(
                    condition_label(row, record["route_active"]),
                    fontsize=9.2,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=58,
                )
    fig.subplots_adjust(left=0.205, right=0.995, top=0.985, bottom=0.085, wspace=0.035, hspace=0.12)
    for column_index, caption in enumerate(subcaptions):
        position = axes[-1, column_index].get_position()
        colour = colours[column_index]
        fig.text(
            (position.x0 + position.x1) / 2,
            0.052,
            caption,
            ha="center",
            va="center",
            fontsize=FIG8_SUBCAPTION_FONT_SIZE,
            color=SUBCAPTION_COLOR,
            fontweight="bold",
        )
    fig.savefig(FIG8_PNG, dpi=300, facecolor="white")
    fig.savefig(FIG8_PDF, facecolor="white")
    plt.close(fig)
    flatten_png(FIG8_PNG)


def plot_figure9(record: dict[str, Any]) -> None:
    methods = (
        "source",
        "degraded",
        "wiener_oracle",
        "richardson_lucy_oracle",
        "unsupervised_wiener_oracle",
        "wiener_wrong_psf",
        "tv_denoise",
        "nl_means",
        "unsharp",
        "nafnet_fidelity",
        "merit_core_adapter",
        "merit_routed_v3i",
    )
    record["zoom_box"] = source_defined_zoom_box(record["outputs"]["source"])
    residual_vmax = common_residual_scale([record], methods)
    record["display_residual_vmax_dn"] = residual_vmax
    box = record["zoom_box"]
    source_zoom = crop_to_box(record["outputs"]["source"], box)
    fig, axes = plt.subplots(3, 4, figsize=(15.8, 10.1))
    for panel_index, (axis, method) in enumerate(zip(axes.ravel(), methods)):
        if method == "source":
            border = "#374151"
        elif method == "degraded":
            border = "#B42318"
        elif method.startswith("merit"):
            border = "#087F5B"
        else:
            border = "#6B7280"
        output_zoom = crop_to_box(record["outputs"][method], box)
        style_axis(axis, output_zoom, border)
        if method != "source":
            add_residual_inset(axis, source_zoom, output_zoom, residual_vmax)
        axis.text(
            0.5,
            -0.075,
            f"({chr(ord('a') + panel_index)}) {DISPLAY_NAMES[method]}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=FIG9_SUBCAPTION_FONT_SIZE,
            color=SUBCAPTION_COLOR,
            fontweight="bold",
            clip_on=False,
        )
    row = record["row"]
    fig.subplots_adjust(left=0.015, right=0.995, top=0.985, bottom=0.055, wspace=0.035, hspace=0.24)
    # Keep the Markdown preview light enough for common renderers while the
    # accompanying PDF remains the publication-resolution vector artifact.
    fig.savefig(FIG9_PNG, dpi=200, facecolor="white")
    fig.savefig(FIG9_PDF, facecolor="white")
    plt.close(fig)
    flatten_png(FIG9_PNG)


def provenance_payload(
    figure: str,
    selection_rule: str,
    records: list[dict[str, Any]],
    artifacts: tuple[Path, Path],
) -> dict[str, Any]:
    return {
        "schema_version": f"MERIT-R3-6-{figure.upper()}-C063-v3",
        "source_analysis": "c063_full737_restoration_battery/full737_battery_v1",
        "selection_rule": selection_rule,
        "selection_uses_model_outcomes": False,
        "redundant_footer_removed": True,
        "regeneration": (
            "Every deterministic plotted array matched the hash frozen during the original "
            "737-frame run. The unsupervised-Wiener panel is a same-parameter stochastic rerun "
            "because its original internal random state was not persisted; no original per-frame "
            "metric is attached to that rerun."
        ),
        "records": [
            {
                "condition_cell": str(record["row"]["condition_cell"]),
                "ordinal": int(record["row"]["ordinal"]),
                "frame": str(record["row"]["frame"]),
                "input_sha256": str(record["row"]["image_sha256"]),
                "primary_condition": bool(int(record["row"]["primary_condition"])),
                "route_active": bool(record["route_active"]),
                "degradation_seed": int(record["degradation_seed"]),
                "progress_json": str(record["progress_path"].relative_to(ROOT)).replace("\\", "/"),
                "progress_sha256": record["progress_sha256"],
                "verified_output_hashes": sum(record["hash_matches"].values()),
                "non_exact_regenerations": [
                    method for method, matched in record["hash_matches"].items() if not matched
                ],
                "display_zoom_box_yxhw": [int(value) for value in record["zoom_box"]],
                "display_zoom_selection": "maximum mean Sobel magnitude over a fixed 3x3 source-only candidate grid",
                "display_residual_vmax_dn": float(record["display_residual_vmax_dn"]),
            }
            for record in records
        ],
        "claim_boundary": (
            "Qualitative, outcome-blind examples from the controlled C063 dataset. "
            "Full-dataset analysis remains in Table 4 and Figure 10; these panels are not "
            "human-mask detection accuracy or natural-field confirmation."
        ),
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the frozen deep-restoration outputs")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    battery = load_module("c063_qualitative_battery", BATTERY_SCRIPT)
    manifest = pd.read_csv(MANIFEST, dtype={"frame": str})
    if len(manifest) != 737 or manifest["condition_cell"].nunique() != 50:
        raise RuntimeError("frozen manifest coverage changed")
    anchor_rows = []
    for condition in ANCHOR_CONDITIONS:
        group = manifest.loc[manifest["condition_cell"] == condition]
        if group.empty:
            raise RuntimeError(f"missing anchor condition: {condition}")
        anchor_rows.append(group.sort_values("image_sha256").iloc[0])
    primary = manifest.loc[manifest["primary_condition"] == 1]
    if len(primary) != 667 or primary["condition_cell"].nunique() != 45:
        raise RuntimeError("primary dataset coverage changed")
    battery_row = primary.sort_values("image_sha256").iloc[0]

    device = torch.device("cuda")
    models, _detector, inference_module, adapter_module, candidate, gate = battery.load_models(device)
    runtime = (models, inference_module, adapter_module, candidate, gate, device)
    with torch.inference_mode():
        anchor_records = [recreate(row, battery, runtime) for row in anchor_rows]
        battery_record = recreate(battery_row, battery, runtime)

    plot_figure8(anchor_records)
    plot_figure9(battery_record)
    FIG8_PROVENANCE.write_text(
        json.dumps(
            provenance_payload(
                "fig8",
                "Four fixed acquisition-grid anchors selected before outcome inspection; source-only high-structure display zoom.",
                anchor_records,
                (FIG8_PNG, FIG8_PDF),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    FIG9_PROVENANCE.write_text(
        json.dumps(
            provenance_payload(
                "fig9",
                "One prespecified primary-analysis frame selected before outcome inspection; source-only high-structure display zoom.",
                [battery_record],
                (FIG9_PNG, FIG9_PDF),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "figure8_records": len(anchor_records),
                "figure9_records": 1,
                "verified_arrays": sum(
                    sum(record["hash_matches"].values()) for record in anchor_records
                )
                + sum(battery_record["hash_matches"].values()),
                "stochastic_reruns": sum(
                    sum(not matched for matched in record["hash_matches"].values())
                    for record in [*anchor_records, battery_record]
                ),
                "figure8_sha256": sha256_file(FIG8_PNG),
                "figure9_sha256": sha256_file(FIG9_PNG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
