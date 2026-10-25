from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE.parent
    / "c015_reviewer_closure"
    / "r3"
    / "outputs"
    / "r3_5_training_curve_seed_mean.csv"
)


def load_objective() -> dict[str, list[tuple[int, float, float]]]:
    curves: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["metric"] != "reconstructed_total_objective":
                continue
            curves[row["arm"]].append(
                (int(row["step"]), float(row["seed_mean"]), float(row["seed_sd"]))
            )
    for values in curves.values():
        values.sort(key=lambda item: item[0])
    return dict(curves)


def main() -> None:
    curves = load_objective()
    labels = {
        "single": "Single conditioned",
        "cascade": "Noise→optics cascade",
        "parallel": "Parallel+fusion",
    }
    colours = {"single": "#1f77b4", "cascade": "#d62728", "parallel": "#2ca02c"}

    fig = Figure(figsize=(8.3, 3.25), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    for arm in ("single", "cascade", "parallel"):
        values = curves[arm]
        steps = np.asarray([item[0] for item in values], dtype=float)
        means = np.asarray([item[1] for item in values], dtype=float)
        sds = np.asarray([item[2] for item in values], dtype=float)
        for axis in axes:
            axis.plot(steps, means, color=colours[arm], linewidth=1.6, label=labels[arm])
            axis.fill_between(
                steps,
                np.maximum(means - sds, np.finfo(float).tiny),
                means + sds,
                color=colours[arm],
                alpha=0.14,
                linewidth=0,
            )

    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 8000)
    axes[0].set_title("Full fixed-budget training")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Reconstructed objective (log scale)")

    axes[1].set_xlim(7200, 8000)
    axes[1].set_ylim(0.0145, 0.0180)
    axes[1].set_title("Final 10% of steps")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Reconstructed objective")

    for axis in axes:
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)

    fig.savefig(HERE / "figS1.pdf", bbox_inches="tight")
    fig.savefig(HERE / "figS1.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
