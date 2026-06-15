# -*- coding: utf-8 -*-
"""SP03 E4 — publication figures from measured results (F1 MPD triangle, F4 detection, F8 classical battery)."""
import os, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "font.family": "DejaVu Sans"})
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
FIG = os.path.join(DATA, "figs"); os.makedirs(FIG, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(DATA, "mpd_axes.csv"))))
M = {r["method"]: dict(PSNR=float(r["PSNR"]), LPIPS=float(r["LPIPS"]), MTFerr=float(r["MTF_err"])) for r in rows}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig); print("saved", name)


# ---------- F1 — Measurement-Perception-Distortion (the conceptual centerpiece) ----------
# x = Perception (LPIPS, lower better), y = Measurement (MTF error, lower better), size ~ Distortion (PSNR)
fig, ax = plt.subplots(figsize=(6.2, 4.6))
groups = {
    "denoise-only": (["TV denoise", "NL-means"], "#1f77b4"),
    "oracle deconv": (["Wiener (oracle)", "Richardson-Lucy"], "#ff7f0e"),
    "naive/degraded": (["degraded", "unsharp"], "#7f7f7f"),
    "generative (diffusion)": (["SD-x4 diffusion", "DiffBIR v2.1"], "#9467bd"),
    "learned (ours)": (["NAFNet-big", "MERIT (ours)"], "#2ca02c"),
}
F1_OFF = {"degraded": (8, -12, "left"), "DiffBIR v2.1": (2, 8, "center"),
          "NL-means": (6, -11, "left"), "TV denoise": (6, 4, "left")}
pmin, pmax = min(v["PSNR"] for v in M.values()), max(v["PSNR"] for v in M.values())
for label, (names, col) in groups.items():
    xs = [M[n]["LPIPS"] for n in names]; ys = [M[n]["MTFerr"] for n in names]
    ss = [60 + 340 * (M[n]["PSNR"] - pmin) / (pmax - pmin) for n in names]
    ax.scatter(xs, ys, s=ss, c=col, alpha=0.8, edgecolors="k", linewidths=0.6, label=label, zorder=3)
    for n in names:
        tag = "MERIT" if "MERIT" in n else n.split(" (")[0]
        dx, dy, ha = F1_OFF.get(tag, (5, 4, "left"))
        ax.annotate(tag, (M[n]["LPIPS"], M[n]["MTFerr"]),
                    xytext=(dx, dy), ha=ha, textcoords="offset points", fontsize=8,
                    fontweight="bold" if "MERIT" in n else "normal")
ax.set_xlabel("Perception  —  LPIPS to clean  (lower better)")
ax.set_ylabel("Measurement  —  |MTF50 error|  (lower better)")
ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.60), ncol=1,
          framealpha=0.9, markerscale=0.45, handletextpad=0.4, labelspacing=0.6)
ax.set_ylim(-0.003, max(v["MTFerr"] for v in M.values()) * 1.15)
save(fig, "F1_mpd_triangle")

# ---------- F4 — detection-fitness recovery (vs oracle Wiener) ----------
det = [("degraded", 1, 0), ("Wiener\n(oracle)", 9, 10), ("NAFNet-big", 23, 24), ("MERIT\n(ours)", 65, 26)]
fig, ax = plt.subplots(figsize=(5.2, 4.0))
xs = np.arange(len(det))
cols = ["#7f7f7f", "#ff7f0e", "#8fbf8f", "#2ca02c"]
ax.bar(xs, [d[1] for d in det], yerr=[d[2] for d in det], capsize=4, color=cols, edgecolor="k", linewidth=0.6)
ax.set_xticks(xs); ax.set_xticklabels([d[0] for d in det])
ax.set_ylabel("Detection-fitness recovery (% of lost, N=12)")
for i, d in enumerate(det):
    ax.text(i, d[1] + d[2] + 1.5, f"+{d[1]}%", ha="center", fontsize=9, fontweight="bold" if d[0].startswith("MERIT") else "normal")
ax.set_ylim(0, 100)
save(fig, "F4_detection_recovery")

# ---------- F8 — classical-baseline battery (MTF recovery vs fidelity) ----------
fig, ax = plt.subplots(figsize=(6.0, 4.4))
order = ["degraded", "unsharp", "Wiener (oracle)", "Richardson-Lucy", "TV denoise", "NL-means",
         "SD-x4 diffusion", "DiffBIR v2.1", "NAFNet-big", "MERIT (ours)"]
LBL_OFF = {"DiffBIR v2.1": (-3, 9, "center"), "degraded": (7, -13, "left"),
           "NL-means": (6, -11, "left")}  # split the crowded mid/right cluster
for n in order:
    is_ours = "MERIT" in n
    ax.scatter(M[n]["PSNR"], M[n]["MTFerr"], s=150 if is_ours else 80,
               c="#2ca02c" if is_ours else "#1f77b4", marker="*" if is_ours else "o",
               edgecolors="k", linewidths=0.6, zorder=3)
    tag = "MERIT" if is_ours else n.split(" (")[0]
    dx, dy, ha = LBL_OFF.get(tag, (4, 3, "left"))
    ax.annotate(tag, (M[n]["PSNR"], M[n]["MTFerr"]), xytext=(dx, dy), ha=ha,
                textcoords="offset points", fontsize=7.5,
                fontweight="bold" if is_ours else "normal")
ax.set_xlabel("Distortion fidelity  —  PSNR to clean (dB, higher better)")
ax.set_ylabel("Measurement  —  |MTF50 error| (lower better)")
ax.set_ylim(-0.003, max(M[n]["MTFerr"] for n in order) * 1.15)
save(fig, "F8_classical_battery")

# ---------- F3 — MTF50 recovery across degradation severity ----------
sw = list(csv.reader(open(os.path.join(DATA, "sweep.csv"))))[1:]
clean_mtf = float([r for r in sw if r[0] == "clean_mtf"][0][1])
sw = [[float(c) for c in r] for r in sw if r[0] != "clean_mtf"]
bl = [r[0] for r in sw]
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.axhline(clean_mtf, ls="--", c="k", lw=1, label=f"clean ({clean_mtf:.3f})")
ax.plot(bl, [r[1] for r in sw], "o-", c="#7f7f7f", label="degraded")
ax.plot(bl, [r[2] for r in sw], "s-", c="#ff7f0e", label="Wiener (oracle)")
ax.plot(bl, [r[3] for r in sw], "*-", c="#2ca02c", ms=11, label="MERIT (ours)")
ax.set_xlabel("added blur σ (px)"); ax.set_ylabel("recovered MTF50 (cy/px)")
ax.set_title("MTF50 recovery vs degradation severity\n(MERIT ≈ clean across all levels; Wiener unstable)")
ax.legend(fontsize=8); ax.set_ylim(0, clean_mtf * 1.25)
save(fig, "F3_mtf_sweep")

# ---------- F7 — ablation: single MERIT vs explicit disentanglement vs no-task ----------
abl = [("MERIT", 98, 65, "#2ca02c"),
       ("no task", 107, 23, "#8fbf8f"),
       ("cascade\n(B)", 77, 49, "#d62728"),
       ("parallel+\nfusion (B)", 76, 49, "#e377c2")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
xs = np.arange(len(abl)); cols = [a[3] for a in abl]
b1 = a1.bar(xs, [a[1] for a in abl], color=cols, edgecolor="k", lw=0.6)
a1.axhline(100, ls=":", c="k", lw=0.8); a1.set_ylabel("MTF50 (% of clean)")
a1.bar_label(b1, fmt="%d%%", padding=2, fontsize=9)
b2 = a2.bar(xs, [a[2] for a in abl], color=cols, edgecolor="k", lw=0.6)
a2.set_ylabel("Detection recovery (%)")
a2.bar_label(b2, fmt="+%d%%", padding=2, fontsize=9)
for ax in (a1, a2):
    ax.set_xticks(xs); ax.set_xticklabels([a[0] for a in abl], fontsize=8.5)
    ax.margins(y=0.12)
fig.subplots_adjust(bottom=0.20, top=0.96, wspace=0.30)
for ax, txt in zip([a1, a2], ["(a) Measurement (MTF50, % of clean)", "(b) Task (crack-detection recovery, %)"]):
    p = ax.get_position(); fig.text((p.x0+p.x1)/2, 0.015, txt, ha="center", va="bottom", fontsize=10)
save(fig, "F7_ablation")

# ---------- F2 — MERIT framework pipeline schematic ----------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(10.6, 3.1)); ax.axis("off"); ax.set_xlim(0, 11.6); ax.set_ylim(0, 3.2)
W = 2.25
boxes = [(0.3, "High-quality\nframe (clean)", "#eaf3ea"),
         (3.0, "Degraded +\nmeasurement prompt\n(blur, noise)", "#fde3d0"),
         (5.75, "MERIT restorer\n(NAFNet +\nmeasurement prompt)", "#e3edf7"),
         (8.5, "Restored\nframe", "#eaf3ea")]
for x, txt, col in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.25), W, 1.05, boxstyle="round,pad=0.05", fc=col, ec="k", lw=1.0))
    ax.text(x + W / 2, 1.78, txt, ha="center", va="center", fontsize=7.8)
for xa, xb in [(2.55, 3.0), (5.30, 5.75), (8.05, 8.5)]:
    ax.add_patch(FancyArrowPatch((xa, 1.78), (xb, 1.78), arrowstyle="-|>", mutation_scale=13, lw=1.3, color="k"))
ax.text(2.78, 2.85, "calibrated degradation  (measured MTF·σ_motion·σ_n)", ha="center", fontsize=7, color="#b35900")
ax.text(5.9, 0.72, "loss: L1 + frequency + segmenter-in-loop task", ha="center", fontsize=7.2, color="#1f5fa8")
ax.text(9.62, 0.55, "validation: chart MTF50 (measurement)\n+ detection (task) + false-crack (no hallucination)",
        ha="center", va="center", fontsize=6.6, color="#2a7a2a")
save(fig, "F2_pipeline")

print("\nfigures ->", FIG)
