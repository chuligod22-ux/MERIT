"""Create the C063 interpretation figure from frozen exported tables."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "artifacts" / "full737_battery_v1"
PNG = OUTPUT / "c063_interpretation_forest.png"
PDF = OUTPUT / "c063_interpretation_forest.pdf"
PROVENANCE = OUTPUT / "c063_interpretation_forest_provenance.json"

METHODS = [
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
]
LABELS = {
    "wiener_oracle": "Wiener (oracle PSF)",
    "richardson_lucy_oracle": "Richardson–Lucy",
    "unsupervised_wiener_oracle": "Unsupervised Wiener",
    "wiener_wrong_psf": "Wiener (wrong PSF)",
    "tv_denoise": "TV denoise",
    "nl_means": "NL-means",
    "unsharp": "Unsharp mask",
    "nafnet_fidelity": "Fidelity NAFNet",
    "merit_core_adapter": "MERIT core",
    "merit_routed_v3i": "MERIT + gate",
}

SUBCAPTION_COLOR = "#101828"
SUBCAPTION_FONT_SIZE = 14.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def values(text: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(text), dtype=float)


def add_text_row(
    figure: plt.Figure,
    subplot_spec: object,
    text: str,
    *,
    fontsize: float,
    weight: str = "normal",
) -> plt.Axes:
    """Add centred text in a dedicated row beneath a data panel."""
    axis = figure.add_subplot(subplot_spec)
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=SUBCAPTION_COLOR,
    )
    return axis


def main() -> None:
    summary_path = OUTPUT / "method_summary.csv"
    dominance_path = OUTPUT / "merit_dominance.csv"
    condition_path = OUTPUT / "condition_results.csv"
    summary = pd.read_csv(summary_path)
    dominance = pd.read_csv(dominance_path)
    condition = pd.read_csv(condition_path)
    summary = summary[summary["metric"] == "mean_probability"].set_index("method").loc[METHODS]
    dominance = dominance[dominance["metric"] == "mean_probability"].set_index("baseline")
    primary = condition[condition["primary_condition"] == True].copy()  # noqa: E712

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(12.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.05, 0.14, 1.0, 0.14],
        hspace=0.04,
        wspace=0.16,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[2, :])

    y = np.arange(len(METHODS))[::-1]
    centers = summary["mean_delta_error"].to_numpy(float) * 1e4
    intervals = np.vstack(summary["delta_error_95pct_ci"].map(values).to_numpy()) * 1e4
    colors = ["#6B7280"] * 8 + ["#1565C0", "#00897B"]
    for index, method in enumerate(METHODS):
        ax_a.errorbar(
            centers[index],
            y[index],
            xerr=[[centers[index] - intervals[index, 0]], [intervals[index, 1] - centers[index]]],
            fmt="o",
            ms=5.5,
            capsize=2.8,
            color=colors[index],
            ecolor=colors[index],
            lw=1.35,
        )
    ax_a.axvline(0, color="black", lw=0.9, ls="--")
    ax_a.set_yticks(y, [LABELS[item] for item in METHODS])
    ax_a.set_xlabel(r"Method error − degraded error ($\times 10^{-4}$)")
    ax_a.grid(axis="x", color="#D1D5DB", lw=0.6, alpha=0.8)

    baseline_methods = METHODS[:8]
    yb = np.arange(len(baseline_methods))[::-1]
    dom_center = dominance.loc[baseline_methods, "mean_baseline_minus_merit_error"].to_numpy(float) * 1e4
    dom_ci = np.vstack(
        dominance.loc[baseline_methods, "condition_bootstrap_95pct_ci"].map(values).to_numpy()
    ) * 1e4
    dom_p = dominance.loc[baseline_methods, "holm_adjusted_p"].to_numpy(float)
    for index, method in enumerate(baseline_methods):
        colour = "#1565C0" if dom_p[index] < 0.05 else "#78909C"
        ax_b.errorbar(
            dom_center[index],
            yb[index],
            xerr=[[dom_center[index] - dom_ci[index, 0]], [dom_ci[index, 1] - dom_center[index]]],
            fmt="o",
            ms=5.5,
            capsize=2.8,
            color=colour,
            ecolor=colour,
            lw=1.35,
        )
        ax_b.text(
            dom_ci[index, 1] + 0.035,
            yb[index],
            f"Holm p={dom_p[index]:.3f}",
            va="center",
            fontsize=7.5,
        )
    ax_b.axvline(0, color="black", lw=0.9, ls="--")
    ax_b.set_xlim(-0.06, float(dom_ci[:, 1].max() + 0.62))
    ax_b.set_yticks(yb, [LABELS[item] for item in baseline_methods])
    ax_b.set_xlabel(r"Baseline error − MERIT core error ($\times 10^{-4}$)")
    ax_b.grid(axis="x", color="#D1D5DB", lw=0.6, alpha=0.8)
    add_text_row(
        fig,
        grid[1, 0],
        "(a) Recovery relative to degraded input",
        fontsize=SUBCAPTION_FONT_SIZE,
        weight="bold",
    )
    add_text_row(
        fig,
        grid[1, 1],
        "(b) Direct MERIT comparisons",
        fontsize=SUBCAPTION_FONT_SIZE,
        weight="bold",
    )

    degraded = primary["mean_probability_degraded_abs_source_error"].to_numpy(float)
    core_delta = (
        primary["mean_probability_merit_core_adapter_abs_source_error"].to_numpy(float) - degraded
    ) * 1e4
    routed_delta = (
        primary["mean_probability_merit_routed_v3i_abs_source_error"].to_numpy(float) - degraded
    ) * 1e4
    order = np.argsort(core_delta)
    x = np.arange(len(primary))
    ax_c.axhline(0, color="black", lw=0.9, ls="--")
    ax_c.scatter(x - 0.13, core_delta[order], s=23, color="#1565C0", label="MERIT core", zorder=3)
    ax_c.scatter(x + 0.13, routed_delta[order], s=23, color="#00897B", label="MERIT + gate", zorder=3)
    condition_names = primary["condition_cell"].to_numpy()[order]
    influential = int(np.where(condition_names == "crack_d25_ISO400_V80")[0][0])
    ax_c.annotate(
        "High-influence condition\n2.5 m · ISO400 · 80 km/h",
        xy=(influential - 0.13, core_delta[order][influential]),
        xytext=(influential + 5.5, core_delta[order][influential] + 0.85),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=8.0,
    )
    ax_c.set_xlim(-1, len(primary))
    ax_c.set_xlabel("Prespecified acquisition conditions, ordered by MERIT-core effect")
    ax_c.set_ylabel(r"Method error − degraded error ($\times 10^{-4}$)")
    ax_c.grid(axis="y", color="#D1D5DB", lw=0.6, alpha=0.8)
    ax_c.legend(frameon=False, loc="upper left", ncol=2)
    add_text_row(
        fig,
        grid[3, :],
        "(c) Condition-level heterogeneity",
        fontsize=SUBCAPTION_FONT_SIZE,
        weight="bold",
    )
    fig.savefig(PNG, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "schema_version": "MERIT-C063-INTERPRETATION-FIGURE-v2",
        "inputs": {
            str(summary_path.relative_to(BASE)): sha256(summary_path),
            str(dominance_path.relative_to(BASE)): sha256(dominance_path),
            str(condition_path.relative_to(BASE)): sha256(condition_path),
        },
        "script_sha256": sha256(Path(__file__)),
        "outputs": {PNG.name: sha256(PNG), PDF.name: sha256(PDF)},
        "claim_boundary": (
            "Controlled detector-response comparison; not human-mask detection accuracy. "
            "Frames are repeated observations within acquisition conditions."
        ),
    }
    PROVENANCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
