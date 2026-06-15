# -*- coding: utf-8 -*-
"""Fig.1 (F2_pipeline) — MERIT framework schematic, publication-grade.
Each stage carries a small method-illustrating sub-glyph (no result numbers):
 (1) measurement calibration  (2) calibrated degradation: edge -> anisotropic
 PSF+noise -> degraded edge  (3) measurement prompt: 3-channel stack
 (4) NAFNet restorer: U-Net  (5) training: fidelity (edge-profile match) +
 frozen segmenter-in-loop task (crack patch fires / wall patch stays silent)
 (6) MPD validation: triangle + MTF-recovery / detection / false-crack icons.
The chart anchors both calibration and validation (the measurement loop)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle, Ellipse
from scipy.ndimage import gaussian_filter
FIG = os.environ.get("MERIT_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
plt.rcParams.update({"font.family": "DejaVu Sans"})

AMB = ("#cf8a00", "#fdf1da"); RED = ("#c0392b", "#fbe5e1")
PUR = ("#7d3fb0", "#efe6f8"); BLU = ("#2667b3", "#e3edf9")
TEA = ("#1d8a72", "#e0f3ed"); GRN = ("#2c7d2c", "#e7f3e2")

fig, ax = plt.subplots(figsize=(16.8, 8.0)); ax.axis("off")
ax.set_xlim(0, 17.0); ax.set_ylim(0, 8.0)

def module(x, y, w, h, num, title, hc, bc, title_fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.13",
                                fc=bc, ec=hc, lw=1.8, zorder=3))
    if num:
        ax.add_patch(Circle((x + 0.36, y + h - 0.36), 0.24, fc=hc, ec="none", zorder=5))
        ax.text(x + 0.36, y + h - 0.36, num, ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold", zorder=6)
    ax.text(x + 0.72, y + h - 0.37, title, ha="left", va="center",
            color=hc, fontsize=title_fs, fontweight="bold", zorder=6)

def txt(x, y, s, fs=8.5, **kw):
    ax.text(x, y, s, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), fontsize=fs, zorder=8, **kw)

def eqtag(x, y, s, c):
    ax.text(x, y, s, ha="center", va="bottom", fontsize=7.4, style="italic", color=c, zorder=8)

def arrow(p, q, c="k", lw=1.9, ls="-", ms=15, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=2, shrinkB=2))

# ---------- image sub-panels ----------
def edge_arr(blur=0.0, noise=0.0, seed=0):
    a = np.zeros((34, 34)); a[:, 17:] = 1.0
    if blur > 0: a = gaussian_filter(a, (blur * 0.4, blur))
    if noise > 0: a = a + np.random.RandomState(seed).randn(34, 34) * noise
    return np.clip(a, 0, 1)

def crack_arr(seed=3):
    rng = np.random.RandomState(seed); a = 0.55 + 0.05 * rng.randn(34, 34)
    for t in range(34):
        c = int(3 + 0.8 * t)
        if 0 <= c < 34: a[t, max(0, c - 1):c + 1] = 0.12
    return np.clip(a, 0, 1)

def wall_arr(seed=5):
    rng = np.random.RandomState(seed); a = 0.56 + 0.04 * rng.randn(34, 34)
    a[::9, :] = 0.44; a[:, ::11] = 0.44
    return np.clip(a, 0, 1)

def show(arr, x, y, w, h, ec="#333", lab=None, labfs=7.6, seg=None):
    ax.imshow(arr, cmap="gray", extent=(x, x + w, y, y + h), zorder=6, aspect="auto", vmin=0, vmax=1)
    if seg is not None:
        ax.imshow(np.ma.masked_less(seg, 0.5), cmap="autumn", extent=(x, x + w, y, y + h),
                  zorder=7, aspect="auto", alpha=0.85, vmin=0, vmax=1)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=1.1, zorder=8))
    if lab: txt(x + w / 2, y - 0.2, lab, fs=labfs, fontweight="bold")

def miniplot(bx, by, bw, bh, series, frame=True, fc="white"):
    if frame: ax.add_patch(Rectangle((bx, by), bw, bh, fc=fc, ec="#aaa", lw=0.9, zorder=6))
    for xs, ys, kw in series:
        ax.plot(bx + np.asarray(xs) * bw, by + np.asarray(ys) * bh, zorder=8, **kw)

def unet(cx, cy, w=1.9, H=1.05, h=0.4, c=BLU[0]):
    ax.add_patch(Polygon([(cx-w/2, cy+H/2), (cx-w/2, cy-H/2), (cx-0.13, cy-h/2), (cx-0.13, cy+h/2)],
                         closed=True, fc="#cfe0f5", ec=c, lw=1.3, zorder=7))
    ax.add_patch(Polygon([(cx+w/2, cy+H/2), (cx+w/2, cy-H/2), (cx+0.13, cy-h/2), (cx+0.13, cy+h/2)],
                         closed=True, fc="#cfe0f5", ec=c, lw=1.3, zorder=7))
    for dy in (0.32, 0.0, -0.32):
        ax.add_patch(FancyArrowPatch((cx-w/2+0.13, cy+dy*H), (cx+w/2-0.13, cy+dy*H), arrowstyle="-",
                     lw=0.8, color=c, ls=(0, (2, 2)),
                     connectionstyle=f"arc3,rad={-0.5 if dy>0 else (0.5 if dy<0 else 0)}", zorder=6))
    txt(cx, cy + H/2 + 0.18, "encoder – decoder", fs=6.6, color=c, style="italic")

def chart_glyph(cx, cy, s=0.30):
    for dx, dy, fc in [(-1, 0, "k"), (0, 0, "w"), (-1, -1, "w"), (0, -1, "k")]:
        ax.add_patch(Rectangle((cx+dx*s, cy+dy*s), s, s, fc=fc, ec="k", lw=0.7, zorder=7))

def mpd_triangle(cx, cy, s=0.8):
    P = np.array([[cx, cy+s], [cx-s*0.92, cy-s*0.72], [cx+s*0.92, cy-s*0.72]])
    ax.add_patch(Polygon(P, closed=True, fc="#f3f9ee", ec=GRN[0], lw=1.7, zorder=6))
    for p, lab, col, dy in [(P[0], "M", "#2c7d2c", 0.17), (P[1], "P", "#7d3fb0", -0.04), (P[2], "D", "#2667b3", -0.04)]:
        ax.text(p[0], p[1]+dy, lab, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col, zorder=9)
    ax.scatter([cx], [cy-0.05], marker="*", s=230, color="#d62728", edgecolor="k", lw=0.5, zorder=9)
    txt(cx, cy-0.42, "MERIT", fs=7.0, fontweight="bold", color="#d62728")

# ===================== (1) Measurement calibration =====================
module(1.45, 5.62, 3.05, 1.62, "1", "Measurement calibration", *AMB, title_fs=9.5)
chart_glyph(1.86, 6.36)
txt(3.05, 6.46, "ISO 12233\n+ e-SFR", fs=7.8)
txt(2.97, 5.92, r"measured  $\mathrm{MTF}_{50}(d)\,\cdot\,\sigma_{\mathrm{motion}}(v)\,\cdot\,\sigma_n(\mathrm{ISO})$", fs=7.9)

# clean x
show(crack_arr(2), 0.20, 3.55, 1.18, 1.18, lab=r"clean $x$")

# ===================== (2) Calibrated degradation =====================
module(1.45, 2.40, 3.10, 3.05, "2", "Calibrated degradation", *RED)
show(edge_arr(0, 0), 1.62, 3.95, 0.78, 0.95)
ax.add_patch(Ellipse((2.95, 4.42), 0.62, 0.34, fc="#f6d6d0", ec=RED[0], lw=1.3, zorder=7))
txt(2.95, 4.42, r"$G_{\sigma_h,\sigma_v}$", fs=7.6)
txt(2.95, 4.04, "+ noise", fs=6.8, color=RED[0])
for (sx, sy) in [(2.78, 4.74), (3.02, 4.80), (3.16, 4.68), (2.88, 4.66)]:
    ax.add_patch(Circle((sx, sy), 0.022, fc="#999", ec="none", zorder=8))
show(edge_arr(2.2, 0.07, 1), 3.55, 3.95, 0.78, 0.95)
arrow((2.42, 4.42), (2.63, 4.42), lw=1.4, ms=11)
arrow((3.28, 4.42), (3.53, 4.42), lw=1.4, ms=11)
txt(3.0, 3.55, r"$y = G_{\sigma_h,\sigma_v}\!*x + n,\ \ \sigma=K/\mathrm{MTF}_{50}$", fs=8.0)
txt(3.0, 3.20, "anisotropic blur (σ_h ≥ σ_v) + sensor noise", fs=6.9, color=RED[0])
eqtag(3.0, 2.55, "Eq. 1–4", RED[0])

# ===================== (3) Measurement prompt =====================
module(4.75, 2.40, 2.55, 3.05, "3", "Measurement prompt", *PUR)
# 3-channel stack (offset squares): degraded img + p_b + p_n
bx, by, sz, off, dz = 5.20, 3.78, 0.80, 0.15, 0.37
show(edge_arr(2.2, 0.07, 2), bx, by, sz, sz, ec=PUR[0])
ax.add_patch(Rectangle((bx+off, by+dz), sz, sz*0.42, fc="#cdbce6", ec=PUR[0], lw=1.1, zorder=8))
ax.add_patch(Rectangle((bx+2*off, by+2*dz), sz, sz*0.42, fc="#b79ad9", ec=PUR[0], lw=1.1, zorder=9))
txt(bx+sz+0.48, by+0.16, r"$\tilde{y}$", fs=8.0)
txt(bx+sz+0.56, by+dz+0.16, r"$p_b$", fs=8.0, color=PUR[0])
txt(bx+sz+0.64, by+2*dz+0.16, r"$p_n$", fs=8.0, color=PUR[0])
txt(6.02, 3.38, r"$\mathbf{x}=[\,\tilde{y},\,p_b,\,p_n\,]$", fs=8.6)
txt(6.02, 3.06, "blur / noise as constant-map channels", fs=6.7, color=PUR[0])
eqtag(6.02, 2.55, "Eq. 5–6", PUR[0])

# ===================== (4) MERIT restorer =====================
module(7.50, 2.40, 3.10, 3.05, "4", "MERIT restorer", *BLU)
unet(9.05, 4.42, w=1.95, H=1.0, h=0.4)
txt(9.05, 3.50, r"NAFNet $\mathcal{F}_\theta$ + global residual", fs=8.0)
txt(9.05, 3.16, "backbone-agnostic:  CNN / Transformer / SSM", fs=6.8, color=BLU[0], fontweight="bold")
eqtag(9.05, 2.55, "Eq. 7 · §4.7", BLU[0])

# restored
show(crack_arr(2), 10.80, 3.55, 1.18, 1.18, lab=r"restored $\hat{x}$")

# ===================== (6) MPD validation =====================
module(12.15, 2.40, 4.70, 3.05, "6", "MPD validation", *GRN, title_fs=9.5)
mpd_triangle(13.05, 3.92, s=0.70)
# icon column
ix = 14.28
# (a) MTF recovery curve
xs = np.linspace(0, 1, 40)
clean = 1 / (1 + (xs / 0.55) ** 4); deg = 0.45 / (1 + (xs / 0.32) ** 4)
miniplot(ix, 4.02, 1.00, 0.58, [(xs, clean, dict(color="#2c7d2c", lw=1.4)),
                                (xs, deg, dict(color="#999", lw=1.1, ls=":")),
                                (xs, clean * 0.99, dict(color="#d62728", lw=1.2, ls=(0, (3, 2))))])
txt(15.42, 4.32, "Measurement:\nchart MTF50\n(restored ≈ clean)", fs=6.5, ha="left")
# (b) detection (crack fires)
show(crack_arr(2), ix, 3.05, 0.60, 0.60, ec=GRN[0], seg=(crack_arr(2) < 0.25).astype(float))
ax.text(ix + 0.30, 2.94, "✓", ha="center", va="top", fontsize=10, color="#2c7d2c", zorder=9, fontweight="bold")
txt(15.08, 3.35, "Task:\ndetection fitness", fs=6.5, ha="left")
# (c) false-crack (wall stays clean)
show(wall_arr(5), 15.92, 3.05, 0.60, 0.60, ec=GRN[0])
ax.text(16.22, 3.35, "∅", ha="center", va="center", fontsize=12, color="#c0392b", zorder=9)
txt(16.22, 2.88, "false-crack:\nno fabrication", fs=6.3)

# ===================== (5) Training objective =====================
module(4.75, 0.35, 5.85, 1.85, "5", "Training objective", *TEA)
# Fidelity (left): edge-profile match
txt(5.95, 1.62, "Fidelity", fs=8.6, fontweight="bold", color=TEA[0])
xs = np.linspace(0, 1, 40); prof = 1 / (1 + np.exp(-(xs - 0.5) * 14))
miniplot(5.30, 0.70, 1.15, 0.62, [(xs, prof, dict(color="#1d8a72", lw=1.5)),
                                  (xs, prof * 0.98 + 0.01, dict(color="#d62728", lw=1.2, ls=(0, (3, 2))))])
txt(6.78, 1.00, r"$\mathcal{L}_1+\mathcal{L}_{\mathrm{grad}}+\lambda_{\mathrm{ff}}\mathcal{L}_{\mathrm{ff}}$", fs=8.0, ha="left")
ax.plot([7.95, 7.95], [0.55, 1.95], color="#cdddd7", lw=1.0, zorder=5)
# Task (right): crack fires / wall silent + frozen segmenter
txt(9.35, 1.62, "Task  (hallucination-suppressing)", fs=8.4, fontweight="bold", color=TEA[0])
show(crack_arr(2), 8.15, 0.62, 0.66, 0.66, ec=TEA[0], seg=(crack_arr(2) < 0.25).astype(float))
txt(8.48, 0.46, "crack → recover", fs=6.3)
show(wall_arr(5), 9.00, 0.62, 0.66, 0.66, ec=TEA[0])
txt(9.33, 0.46, "wall → suppress", fs=6.3)
ax.add_patch(FancyBboxPatch((9.95, 0.72), 0.78, 0.5, boxstyle="round,pad=0.02", fc="#d7eee7", ec=TEA[0], lw=1.0, zorder=7))
txt(10.34, 0.97, "seg(·)", fs=7.0, fontweight="bold")
ax.text(10.02, 1.17, "❄", fontsize=8.5, color="#3a7", zorder=8, ha="center", va="center")
txt(10.34, 1.40, r"$\lambda_{\mathrm{task}}\|\mathrm{seg}(\hat{x})-\mathrm{seg}(x)\|$", fs=7.4)
eqtag(7.7, 0.40, "Eq. 8–11 · frozen segmenter", TEA[0])

# ===================== arrows / flow =====================
arrow((1.40, 4.13), (1.45, 4.13))                       # clean -> 2
arrow((4.55, 4.05), (4.75, 4.05)); txt(4.65, 4.30, r"$y$", fs=8.0)   # 2 -> 3
arrow((7.30, 4.05), (7.50, 4.05))                        # 3 -> 4
arrow((10.60, 4.13), (10.80, 4.13))                      # 4 -> restored
arrow((11.98, 4.13), (12.15, 4.13))                      # restored -> 6
# (1) -> (2) measured envelope
arrow((2.97, 5.62), (2.97, 5.46), c=AMB[0], lw=1.7, ls=(0, (5, 3)))
ax.text(3.22, 5.54, "measured\nenvelope", ha="left", va="center", fontsize=7.0, color=AMB[0], style="italic", zorder=8)
# (5) -> (4) training gradient
arrow((9.05, 2.20), (9.05, 2.40), c=TEA[0], lw=1.7, ls=(0, (4, 3)))
ax.text(9.24, 2.30, r"$\nabla_\theta$", fontsize=8.5, color=TEA[0], zorder=8, ha="left", va="center")
# (1) -> (6) measurement loop (re-measure restored chart)
ax.add_patch(FancyArrowPatch((2.97, 7.30), (14.0, 5.46), arrowstyle="-|>", mutation_scale=15, lw=1.7,
             color=AMB[0], ls=(0, (5, 3)), connectionstyle="arc3,rad=-0.22", zorder=2, shrinkA=4, shrinkB=4))
ax.text(9.4, 7.86, "re-measure restored chart  →  objective hallucination test", ha="center",
        va="center", fontsize=7.6, color=AMB[0], style="italic", zorder=8)

ax.text(8.5, 0.04, "The chart anchors both calibration and validation: if the restorer is honest with the chart, it is honest with the crack.",
        ha="center", va="bottom", fontsize=7.8, style="italic", color="#444", zorder=8)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"F2_pipeline.{ext}"), dpi=200 if ext == "png" else None, bbox_inches="tight")
plt.close(fig); print("saved F2_pipeline (redesign + sub-glyphs)")
