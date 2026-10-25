"""Publication-quality overview of the revised MERIT framework.

The figure deliberately separates measurement-informed model development,
condition-level optical preservation, and measure-specific evaluation.
Controlled-chart evidence is not presented as a proxy for field performance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(__file__).resolve().parent
OUTPUT_PNG = ROOT / "fig1.png"
OUTPUT_PDF = ROOT / "fig1.pdf"
OUTPUT_PROVENANCE = ROOT / "fig1_provenance.json"
SOURCE_CHART = ROOT / "fig5.png"
SOURCE_FIELD = ROOT / "fig14.png"

COL = {
    "ink": "#111827", "muted": "#526174", "line": "#334155",
    "orange": "#D97706", "orange_fill": "#FFF7E8",
    "red": "#C62828", "red_fill": "#FFF1F0",
    "violet": "#6B3FA0", "violet_fill": "#F5F0FB",
    "blue": "#174E96", "blue_fill": "#EEF5FF",
    "teal": "#087F8C", "teal_fill": "#EAF8F8",
    "green": "#1B7F35", "green_fill": "#EEF8EF",
    "slate_fill": "#F5F7FA", "cyan": "#00C8F0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_fraction(path: Path, box: tuple[float, float, float, float], *,
                  contrast=1.0, grayscale=False) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    left, top, right, bottom = box
    crop = image.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))
    if grayscale:
        crop = ImageOps.grayscale(crop).convert("RGB")
        crop = ImageOps.autocontrast(crop, cutoff=0.5)
    if contrast != 1.0:
        crop = ImageEnhance.Contrast(crop).enhance(contrast)
    return np.asarray(crop)


def arrow(ax, start, end, *, color=COL["line"], lw=1.25, ms=10, style="-|>", zorder=8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=ms,
                                linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                                zorder=zorder))


def card(ax, x, y, w, h, color, fill, number, title, *, title_size=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.18",
                                facecolor="white", edgecolor=color, linewidth=1.5, zorder=1))
    ax.add_patch(FancyBboxPatch((x + 0.015, y + h - 0.72), w - 0.03, 0.70,
                                boxstyle="round,pad=0.01,rounding_size=0.16",
                                facecolor=fill, edgecolor="none", zorder=2))
    ax.add_patch(Circle((x + 0.30, y + h - 0.36), 0.19, facecolor=color,
                        edgecolor="white", lw=1.0, zorder=5))
    ax.text(x + 0.30, y + h - 0.36, str(number), color="white", fontsize=11,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(x + 0.58, y + h - 0.36, title, color=color, fontsize=title_size,
            fontweight="bold", ha="left", va="center", zorder=6)


def framed_image(ax, image, extent, *, edge=COL["line"], lw=0.8, label=None,
                 label_color=COL["ink"]):
    x0, x1, y0, y1 = extent
    ax.imshow(image, extent=extent, aspect="auto", zorder=3)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=edge, linewidth=lw, zorder=4))
    if label:
        ax.text((x0 + x1) / 2, y1 + 0.08, label, ha="center", va="bottom",
                fontsize=7.1, color=label_color, zorder=6)


def small_caption(ax, x, y, text, *, color=COL["muted"], weight="normal",
                  size=7.3, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, linespacing=1.12, zorder=8)


def draw_mtf(ax, x, y, w, h):
    ax.plot([x, x, x + w], [y + h, y, y], color=COL["ink"], lw=0.75, zorder=5)
    bars = np.linspace(0.92, 0.16, 11)
    xs = np.linspace(x + 0.13, x + w - 0.10, len(bars))
    for xx, value in zip(xs, bars):
        ax.add_patch(Rectangle((xx - 0.035, y), 0.07, h * value,
                               facecolor=COL["ink"], edgecolor="none", zorder=4))
    ax.plot([x, x + w], [y + h * 0.5, y + h * 0.5], color="#9CA3AF",
            lw=0.55, ls="--", zorder=3)
    small_caption(ax, x - 0.03, y + h + 0.06, "MTF", size=6.4, ha="left", color=COL["ink"])
    small_caption(ax, x + w / 2, y - 0.18, "spatial frequency", size=5.7)


def draw_psf_noise(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#FAFAFA", edgecolor="#9CA3AF",
                                lw=0.75, ls="--", zorder=3))
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(x + 0.38 * w + 0.27 * w * np.cos(theta),
            y + 0.68 * h + 0.10 * h * np.sin(theta), color="#B8BEC7", zorder=4)
    ax.plot(x + 0.38 * w + 0.27 * w * np.cos(theta),
            y + 0.68 * h + 0.10 * h * np.sin(theta), color=COL["ink"], lw=0.6, zorder=5)
    rng = np.random.default_rng(7)
    ax.scatter(x + rng.uniform(0.18, 0.82, 26) * w,
               y + rng.uniform(0.12, 0.45, 26) * h,
               s=2.5, color=COL["ink"], zorder=5)
    small_caption(ax, x + w / 2, y + h / 2, "+", color=COL["ink"], weight="bold", size=11)
    small_caption(ax, x + w / 2, y - 0.14, "anisotropic PSF + sensor noise", size=5.8)


def draw_prompt_stack(ax, image, x, y, w, h):
    for dx, dy, face, label in [(0.34, 0.28, "#DFF0E4", r"$p_n$"),
                                (0.20, 0.16, "#DDEBFA", r"$p_b$")]:
        ax.add_patch(Rectangle((x + dx, y + dy), w, h, facecolor=face,
                               edgecolor=COL["line"], lw=0.7, zorder=2))
        small_caption(ax, x + dx + w - 0.08, y + dy + h - 0.10,
                      label, size=6.2, weight="bold")
    framed_image(ax, image, (x, x + w, y, y + h), edge=COL["violet"], lw=0.9)
    small_caption(ax, x + w / 2 + 0.19, y - 0.23, r"$[y,\ p_b,\ p_n]$",
                  color=COL["violet"], size=7.4, weight="bold")


def draw_restorer(ax, x, y, w, h):
    mid = y + h / 2
    left = Polygon([(x, y + h), (x + 0.72, y + 0.70 * h),
                    (x + 0.72, y + 0.30 * h), (x, y)], closed=True,
                   facecolor="#C7D9F2", edgecolor=COL["blue"], lw=0.9, zorder=4)
    right = Polygon([(x + w, y + h), (x + w - 0.72, y + 0.70 * h),
                     (x + w - 0.72, y + 0.30 * h), (x + w, y)], closed=True,
                    facecolor="#C7D9F2", edgecolor=COL["blue"], lw=0.9, zorder=4)
    ax.add_patch(left)
    ax.add_patch(right)
    ax.add_patch(Rectangle((x + 0.72, mid - 0.16), w - 1.44, 0.32,
                           facecolor="white", edgecolor=COL["blue"], lw=0.8, zorder=4))
    for offset in (-0.36, 0.0, 0.36):
        arrow(ax, (x + 0.74, mid + offset), (x + w - 0.74, mid + offset),
              color=COL["blue"], lw=0.7, ms=6)
    for bend, yy in [(0.20, mid + 0.62)]:
        ax.add_patch(FancyArrowPatch((x + 0.35, yy), (x + w - 0.35, yy),
                                    arrowstyle="-|>", mutation_scale=6,
                                    connectionstyle=f"arc3,rad={-bend if yy > mid else bend}",
                                    color=COL["blue"], lw=0.7, ls="--", zorder=3))


def draw_gate_panel(ax, x, y, w, h):
    card(ax, x, y, w, h, COL["green"], COL["green_fill"], 5,
         "Optical-preservation gate", title_size=9.0)
    dx, dy, dw, dh = x + 0.34, y + 1.03, 1.30, 1.22
    diamond = Polygon([(dx + dw / 2, dy + dh), (dx + dw, dy + dh / 2),
                       (dx + dw / 2, dy), (dx, dy + dh / 2)], closed=True,
                      facecolor=COL["green_fill"], edgecolor=COL["green"],
                      lw=1.05, zorder=4)
    ax.add_patch(diamond)
    small_caption(ax, dx + dw / 2, dy + 0.74, "RAW optical\nmargin",
                  color=COL["green"], weight="bold", size=6.0)
    small_caption(ax, dx + dw / 2, dy + 0.39, r"$g_c > 0.001$",
                  color=COL["ink"], weight="bold", size=6.1)
    small_caption(ax, dx + dw / 2, dy + 0.20, "[cy/px]",
                  color=COL["muted"], size=5.6)

    bx, bw, bh = x + 2.10, 1.35, 0.66
    for by, face, edge, label in [
        (y + 1.61, COL["green_fill"], COL["green"], "bounded\ncorrection"),
        (y + 0.76, COL["slate_fill"], COL["line"], "byte-exact\nRAW bypass"),
    ]:
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                    boxstyle="round,pad=0.02,rounding_size=0.07",
                                    facecolor=face, edgecolor=edge, lw=0.75, zorder=4))
        small_caption(ax, bx + bw / 2, by + bh / 2, label,
                      color=edge, weight="bold", size=6.2)
    arrow(ax, (dx + dw, dy + dh / 2 + 0.08), (bx - 0.06, y + 1.93),
          color=COL["green"], lw=0.9, ms=7)
    arrow(ax, (dx + dw, dy + dh / 2 - 0.08), (bx - 0.06, y + 1.11),
          color=COL["line"], lw=0.9, ms=7)
    small_caption(ax, bx - 0.16, y + 2.16, "yes", color=COL["green"], size=5.8)
    small_caption(ax, bx - 0.16, y + 0.91, "no", color=COL["line"], size=5.8)
    small_caption(ax, x + w / 2, y + 0.35,
                  "condition-wise correction · exact RAW fallback",
                  color=COL["green"], weight="bold", size=6.3)


def schematic_crack_tile(ax, x, y, w, h, label, *, response=False, phase=0.0):
    """Draw a clearly schematic image/response pair without implying evidence data."""
    face = "#111827" if response else "#D7DCE2"
    edge = "#64748B"
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                           linewidth=0.6, zorder=4))
    t = np.linspace(0.06, 0.94, 80)
    curve = 0.18 + 0.62 * t + 0.065 * np.sin(7.0 * t + phase)
    ax.plot(x + w * t, y + h * curve,
            color=COL["cyan"] if response else "#374151",
            lw=1.15 if response else 0.85, zorder=6)
    if not response:
        ax.plot([x + 0.08 * w, x + 0.90 * w], [y + 0.25 * h, y + 0.31 * h],
                color="#B6BEC8", lw=0.45, zorder=5)
        ax.plot([x + 0.12 * w, x + 0.86 * w], [y + 0.70 * h, y + 0.65 * h],
                color="#C1C7CF", lw=0.45, zorder=5)
    ax.text(x + 0.06 * w, y + 0.84 * h, label,
            ha="left", va="center", fontsize=5.4, fontweight="bold",
            color="white" if response else COL["ink"], zorder=7)


def draw_loss_panel(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.16",
                                facecolor="white", edgecolor=COL["teal"], lw=1.35, zorder=1))
    ax.add_patch(FancyBboxPatch((x + 0.015, y + h - 0.68), w - 0.03, 0.66,
                                boxstyle="round,pad=0.01,rounding_size=0.14",
                                facecolor=COL["teal_fill"], edgecolor="none", zorder=2))
    ax.add_patch(Circle((x + 0.32, y + h - 0.34), 0.18, facecolor=COL["teal"],
                        edgecolor="white", lw=0.9, zorder=5))
    ax.text(x + 0.32, y + h - 0.34, "T", color="white", fontsize=9.0,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(x + 0.62, y + h - 0.34,
            "Training supervision  ·  model-development path only",
            color=COL["teal"], fontsize=9.8, fontweight="bold", va="center", zorder=6)

    sep1, sep2 = x + 4.35, x + 8.75
    for sep in (sep1, sep2):
        ax.plot([sep, sep], [y + 0.22, y + h - 0.82],
                color="#A6C9CC", lw=0.75, ls="--", zorder=3)

    # 1) Image-domain fidelity.
    small_caption(ax, x + 2.18, y + h - 0.96, "Image-domain fidelity",
                  color=COL["teal"], weight="bold", size=7.6)
    px0, py0, pw, ph = x + 0.60, y + 0.58, 3.10, 1.52
    ax.plot([px0, px0, px0 + pw], [py0 + ph, py0, py0], color=COL["ink"], lw=0.7, zorder=4)
    xx = np.linspace(0, 1, 100)
    target = 0.10 + 0.72 / (1 + np.exp(-(xx - 0.42) * 18)) - 0.18 * np.clip(xx - 0.70, 0, None)
    pred = target + 0.05 * np.sin(7 * xx) * np.exp(-((xx - 0.52) / 0.30) ** 2)
    ax.plot(px0 + xx * pw, py0 + target * ph, color=COL["teal"], lw=1.15, zorder=5)
    ax.plot(px0 + xx * pw, py0 + pred * ph, color=COL["teal"], lw=0.9, ls="--", zorder=5)
    small_caption(ax, px0 + pw / 2, y + 0.30,
                  r"pre-clamp $\bar{x}$: $L_1$ + gradient + focal-frequency",
                  color=COL["ink"], size=6.25)

    # 2) Frozen-segmenter response matching. The paired icon is schematic and
    # does not reuse the expert-mask field-validation graphic as a task target.
    small_caption(ax, x + 6.55, y + h - 0.96, "Frozen-segmenter response matching",
                  color=COL["teal"], weight="bold", size=7.6)
    ix0, ox0, iw, ih = x + 4.62, x + 7.42, 1.02, 0.48
    upper_y, lower_y = y + 1.38, y + 0.76
    schematic_crack_tile(ax, ix0, upper_y, iw, ih, r"restored $\hat{x}$", phase=0.35)
    schematic_crack_tile(ax, ix0, lower_y, iw, ih, r"source ref. $\tilde{x}$", phase=0.0)
    schematic_crack_tile(ax, ox0, upper_y, iw, ih,
                         r"$\mathrm{seg}(\hat{x})$", response=True, phase=0.35)
    schematic_crack_tile(ax, ox0, lower_y, iw, ih,
                         r"$\mathrm{seg}(\tilde{x})$", response=True, phase=0.0)
    bx, by, bw, bh = x + 6.00, y + 0.96, 1.05, 0.76
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="white", edgecolor=COL["line"], lw=0.7, zorder=5))
    small_caption(ax, bx + bw / 2, by + bh / 2, "shared frozen\nsegmenter",
                  color=COL["ink"], size=6.0, weight="bold")
    upper_c, lower_c = upper_y + ih / 2, lower_y + ih / 2
    arrow(ax, (ix0 + iw + 0.06, upper_c), (bx - 0.06, by + 0.55),
          color=COL["teal"], lw=0.8, ms=6)
    arrow(ax, (ix0 + iw + 0.06, lower_c), (bx - 0.06, by + 0.21),
          color=COL["teal"], lw=0.8, ms=6)
    arrow(ax, (bx + bw + 0.06, by + 0.55), (ox0 - 0.06, upper_c),
          color=COL["line"], lw=0.8, ms=6)
    arrow(ax, (bx + bw + 0.06, by + 0.21), (ox0 - 0.06, lower_c),
          color=COL["line"], lw=0.8, ms=6)
    small_caption(ax, x + 6.55, y + 0.30,
                  r"$\mathcal{L}_{\mathrm{task}}=\mathrm{mean}|\mathrm{seg}(\hat{x})-\mathrm{seg}(\tilde{x})|$",
                  color=COL["muted"], size=6.3)

    # 3) Optimisation contract.
    cx0, cx1 = x + 9.15, x + w - 0.38
    small_caption(ax, (cx0 + cx1) / 2, y + h - 0.96, "Optimisation contract",
                  color=COL["teal"], weight="bold", size=7.6)
    ax.add_patch(FancyBboxPatch((cx0, y + 1.16), cx1 - cx0, 0.90,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=COL["teal_fill"], edgecolor=COL["teal"],
                                lw=0.7, zorder=3))
    small_caption(ax, (cx0 + cx1) / 2, y + 1.77,
                  r"$\mathcal{L}_{\mathrm{fid}}=\mathcal{L}_1+\mathcal{L}_{\mathrm{grad}}+0.1\,\mathcal{L}_{\mathrm{ff}}$",
                  color=COL["ink"], weight="bold", size=8.0)
    small_caption(ax, (cx0 + cx1) / 2, y + 1.43,
                  r"$\mathcal{L}=\mathcal{L}_{\mathrm{fid}}+5.0\,\mathcal{L}_{\mathrm{task}}$",
                  color=COL["ink"], weight="bold", size=8.2)
    small_caption(ax, cx0 + 0.08, y + 0.79, "• prompt-conditioned restoration",
                  color=COL["ink"], size=6.4, ha="left")
    small_caption(ax, cx0 + 0.08, y + 0.51, "• global residual output",
                  color=COL["ink"], size=6.4, ha="left")
    small_caption(ax, cx0 + 0.08, y + 0.23, "• frozen segmenter weights",
                  color=COL["ink"], size=6.4, ha="left")


def draw_validation(ax, x, y, w, h, field_image):
    card(ax, x, y, w, h, COL["green"], COL["green_fill"], 6,
         "Measure-specific validation", title_size=8.6)
    row_x, row_w, row_h = x + 0.16, w - 0.32, 0.72
    rows = [y + 2.10, y + 1.25, y + 0.40]

    ax.add_patch(FancyBboxPatch((row_x, rows[0]), row_w, row_h,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                facecolor=COL["orange_fill"], edgecolor=COL["orange"], lw=0.65, zorder=3))
    small_caption(ax, row_x + 0.18, rows[0] + 0.50, "REFERENCE-BASED",
                  color=COL["orange"], weight="bold", size=6.2, ha="left")
    small_caption(ax, row_x + 0.18, rows[0] + 0.23, "MTF50 · paired PSNR / LPIPS",
                  color=COL["ink"], weight="bold", size=6.3, ha="left")

    ax.add_patch(FancyBboxPatch((row_x, rows[1]), row_w, row_h,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                facecolor=COL["teal_fill"], edgecolor=COL["teal"], lw=0.65, zorder=3))
    framed_image(ax, field_image, (row_x + 0.12, row_x + 0.72,
                                   rows[1] + 0.10, rows[1] + 0.62),
                 edge=COL["teal"], lw=0.5)
    small_caption(ax, row_x + 0.86, rows[1] + 0.50, "FOUR TUNNELS",
                  color=COL["teal"], weight="bold", size=6.1, ha="left")
    small_caption(ax, row_x + 0.86, rows[1] + 0.23, "AUPRC · matched-sensitivity FPR",
                  color=COL["ink"], weight="bold", size=6.2, ha="left")

    ax.add_patch(FancyBboxPatch((row_x, rows[2]), row_w, row_h,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                facecolor=COL["blue_fill"], edgecolor=COL["blue"], lw=0.65, zorder=3))
    xx = np.linspace(0, 1, 60)
    profile = np.exp(-((xx - 0.50) / 0.12) ** 2)
    ax.plot(row_x + 0.16 + 0.62 * xx, rows[2] + 0.12 + 0.42 * profile,
            color=COL["blue"], lw=0.9, zorder=5)
    ax.plot([row_x + 0.38, row_x + 0.55], [rows[2] + 0.47, rows[2] + 0.47],
            color=COL["orange"], lw=1.1, zorder=5)
    small_caption(ax, row_x + 0.90, rows[2] + 0.50, "APPARENT WIDTH",
                  color=COL["blue"], weight="bold", size=6.1, ha="left")
    small_caption(ax, row_x + 0.90, rows[2] + 0.23, "expert-mask error · paired coverage",
                  color=COL["ink"], weight="bold", size=6.2, ha="left")


def draw() -> None:
    if not SOURCE_CHART.exists() or not SOURCE_FIELD.exists():
        raise FileNotFoundError("Run render_markdown_figure_previews.py before plot_merit_pipeline.py")

    # Figure 5 contains compact metric badges near the bottom of each SFR crop.
    # They become illegible after the thumbnails are reduced for Figure 1, so the
    # overview uses clean image-only crops and leaves the numerical values in Fig. 5.
    source = crop_fraction(SOURCE_CHART, (0.015, 0.030, 0.246, 0.260), contrast=1.05, grayscale=True)
    degraded = crop_fraction(SOURCE_CHART, (0.257, 0.030, 0.489, 0.260), contrast=1.05, grayscale=True)
    restored = crop_fraction(SOURCE_CHART, (0.744, 0.716, 0.975, 0.925), contrast=1.05, grayscale=True)
    field = crop_fraction(SOURCE_FIELD, (0.045, 0.085, 0.247, 0.420), contrast=1.05)

    fig, ax = plt.subplots(figsize=(15.6, 8.32))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    # The former full-width footer repeated the manuscript prose and became
    # unreadable after journal-width scaling.  Start the canvas immediately
    # below the training/validation lane instead.
    ax.set_ylim(0.75, 9.6)
    ax.axis("off")

    ax.text(9, 9.28, "MERIT framework: measurement-informed restoration with optical preservation",
            ha="center", va="center", fontsize=17.5, fontweight="bold", color=COL["ink"])
    ax.text(6.93, 8.86, "MODEL DEVELOPMENT", ha="center", va="center",
            fontsize=7.8, color=COL["muted"], fontweight="bold")
    ax.text(15.80, 8.86, "INFERENCE CONTROL & VALIDATION", ha="center", va="center",
            fontsize=7.0, color=COL["green"], fontweight="bold")
    ax.plot([0.30, 13.55], [8.72, 8.72], color="#D9E0E8", lw=0.8)
    ax.plot([13.90, 17.70], [8.72, 8.72], color="#B9D8BF", lw=0.8)

    # Top row: four aligned development stages and one deployment gate.
    card(ax, 0.30, 5.25, 2.55, 3.20, COL["orange"], COL["orange_fill"],
         1, "Optical\ncharacterisation", title_size=8.8)
    draw_mtf(ax, 0.68, 6.36, 1.62, 0.95)
    small_caption(ax, 1.575, 5.93, "MTF50 + ISO-noise", color=COL["orange"],
                  weight="bold", size=7.0)
    small_caption(ax, 1.575, 5.64, "measured operating envelope", size=6.6)
    small_caption(ax, 1.575, 5.38, "operational source reference",
                  color=COL["muted"], size=6.3)

    card(ax, 3.05, 5.25, 3.55, 3.20, COL["red"], COL["red_fill"],
         2, "Factorised\ndegradation model", title_size=9.0)
    framed_image(ax, source, (3.35, 4.12, 6.15, 7.22), edge=COL["line"], label="source-ref. x")
    draw_psf_noise(ax, 4.42, 6.17, 0.82, 0.96)
    framed_image(ax, degraded, (5.52, 6.29, 6.15, 7.22), edge=COL["red"],
                 lw=1.0, label="degraded y", label_color=COL["red"])
    arrow(ax, (4.16, 6.65), (4.36, 6.65), color=COL["line"], lw=0.9, ms=7)
    arrow(ax, (5.29, 6.65), (5.47, 6.65), color=COL["line"], lw=0.9, ms=7)
    small_caption(ax, 4.825, 5.79, "blur + noise sampled from\nthe measured envelope",
                  color=COL["red"], weight="bold", size=6.4)
    small_caption(ax, 4.825, 5.38, "known parameters retained as prompts", size=6.1)

    card(ax, 6.80, 5.25, 2.50, 3.20, COL["violet"], COL["violet_fill"],
         3, "Prompt\nconditioning", title_size=8.8)
    draw_prompt_stack(ax, degraded, 7.45, 6.12, 1.10, 1.18)
    small_caption(ax, 8.00, 5.76, "global blur / noise maps", color=COL["violet"],
                  weight="bold", size=6.7)
    small_caption(ax, 8.00, 5.42, "explicit condition input", size=6.4)

    card(ax, 9.50, 5.25, 4.05, 3.20, COL["blue"], COL["blue_fill"],
         4, "MERIT restorer", title_size=9.6)
    draw_restorer(ax, 9.92, 6.17, 2.18, 1.05)
    framed_image(ax, restored, (12.48, 13.18, 6.10, 7.24), edge=COL["green"],
                 lw=1.0, label=r"network output $\hat{x}$", label_color=COL["green"])
    arrow(ax, (12.16, 6.69), (12.42, 6.69), color=COL["blue"], lw=0.9, ms=7)
    small_caption(ax, 11.525, 5.80, "encoder–decoder backbone + skip connections",
                  color=COL["blue"], weight="bold", size=6.6)
    small_caption(ax, 11.525, 5.43, "prompt conditioning + global residual", size=6.4)

    draw_gate_panel(ax, 13.90, 5.25, 3.80, 3.20)

    for start, end, color in [
        ((2.85, 6.66), (3.01, 6.66), COL["orange"]),
        ((6.60, 6.66), (6.76, 6.66), COL["red"]),
        ((9.30, 6.66), (9.46, 6.66), COL["violet"]),
    ]:
        arrow(ax, start, end, color=color, lw=1.3, ms=9)
    arrow(ax, (13.22, 6.69), (14.19, 6.89), color=COL["blue"], lw=1.15, ms=8)
    small_caption(ax, 13.72, 7.13, r"raw $y$ + $\hat{x}$",
                  color=COL["blue"], weight="bold", size=5.7)

    # Bottom row: training contract and validation evidence occupy a common baseline.
    draw_loss_panel(ax, 0.30, 1.05, 13.25, 3.65)
    draw_validation(ax, 13.90, 1.05, 3.80, 3.65, field)

    arrow(ax, (11.52, 4.70), (11.52, 5.21), color=COL["teal"], lw=1.0, ms=8)
    small_caption(ax, 11.78, 4.95, "optimises", color=COL["teal"],
                  weight="bold", size=6.2, ha="left")

    # Aligned lane separator distinguishes development from inference control.
    ax.plot([13.72, 13.72], [0.96, 8.52], color="#A8CDB0", lw=0.85, ls=(0, (3, 2)))

    fig.savefig(OUTPUT_PNG, dpi=350, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)

    provenance = {
        "schema_version": "MERIT-R3-6-FIG1-v12",
        "purpose": "Grid-aligned, print-readable framework overview with separated development, inference-control, and validation lanes.",
        "claim_boundary": "Controlled chart, natural detection, and width measures are reported separately.",
        "visual_sources": {SOURCE_CHART.name: sha256_file(SOURCE_CHART),
                           SOURCE_FIELD.name: sha256_file(SOURCE_FIELD)},
        "artifacts": {OUTPUT_PNG.name: sha256_file(OUTPUT_PNG),
                      OUTPUT_PDF.name: sha256_file(OUTPUT_PDF)},
    }
    OUTPUT_PROVENANCE.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    draw()
