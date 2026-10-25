"""Regenerate Figure 10 with descriptive, non-causal detector-response wording."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = ROOT / "fig10.pdf"
OUTPUT_PNG = ROOT / "fig10.png"
OUTPUT_PROVENANCE = ROOT / "fig10_provenance.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw() -> None:
    labels = [
        "unsharp", "SD-x4", "NL-means", "unsup.\nWiener", "TV-\ndenoise",
        "wrong-PSF\nWiener", "Wiener", "Richardson-\nLucy", "NAFNet-\nbig", "MERIT", "DiffBIR",
    ]
    means = np.array([-1, 1, 2, 2, 3, 5, 9, 13, 23, 65, 1126], dtype=float)
    stds = np.array([1, 3, 1, 4, 1, 8, 10, 8, 24, 26, 1129], dtype=float)
    clipped = np.minimum(means, 98)
    x = np.arange(len(labels))
    colors = [
        "#8E6BAF", "#9A77C7", "#4D8DB3", "#C77732", "#2E86B7",
        "#A95F19", "#EA8A0A", "#D9A648", "#95BF95", "#2CA02C", "#D65343",
    ]

    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    bars = ax.bar(x, clipped, color=colors, edgecolor="#172033", linewidth=0.9, zorder=3)
    bars[-1].set_hatch("//")
    ax.errorbar(
        x[:-1], means[:-1], yerr=stds[:-1], fmt="none", ecolor="#111111",
        elinewidth=1.8, capsize=5, capthick=1.8, zorder=4,
    )
    ax.axhline(0, color="#6B7280", linewidth=1.1)
    ax.set_ylim(-6, 105)
    ax.set_ylabel("Recovery of selected detector response\n(% of recorded loss; N=12)", fontsize=11.5)
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for idx, mean in enumerate(means[:-1]):
        y = max(mean + stds[idx] + 2.0, 2.0)
        ax.text(idx, y, f"{mean:+.0f}%", ha="center", va="bottom", fontsize=9.5)
    ax.text(10, 101, "+1126% (off-scale)", ha="center", va="bottom", fontsize=10,
            color="#B83B2E", fontweight="bold")
    ax.text(10, 50, "off-scale\nresponse increase", ha="center", va="center",
            rotation=90, fontsize=9.5, color="white", fontweight="bold")

    ax.set_title(
        "Archival detector-response change in 12 detector-selected crops",
        fontsize=13.5, fontweight="bold", pad=16,
    )
    ax.text(
        0.5, 1.015,
        "Descriptive only: selection and response calculation use the same frozen DeepCrack model",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#58677A",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    provenance = {
        "schema_version": "MERIT-R3-6-FIG10-v1",
        "source": "main.tex Table 4; integer-rounded selected-crop mean and SD",
        "selection_limitation": "Same DeepCrack response used for selection and response calculation; descriptive only.",
        "means_percent": dict(zip(labels, means.tolist())),
        "selected_crop_sd_percent": dict(zip(labels, stds.tolist())),
        "artifacts": {
            OUTPUT_PNG.name: sha256_file(OUTPUT_PNG),
            OUTPUT_PDF.name: sha256_file(OUTPUT_PDF),
        },
    }
    OUTPUT_PROVENANCE.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    draw()
