# -*- coding: utf-8 -*-
"""Standalone regen of Fig.14 (F7_ablation) only — avoids clobbering F3/F4 which are
produced by separate scripts. Values match Table 4 / tables.md exactly."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = os.environ.get("MERIT_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

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
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"F7_ablation.{ext}"))
plt.close(fig); print("saved F7_ablation (98% MTF, no titles, bottom subcaptions)")
