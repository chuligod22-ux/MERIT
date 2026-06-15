# -*- coding: utf-8 -*-
"""Regenerate ONLY F4_detection_recovery (no title). Others untouched."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl, numpy as np
mpl.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.25,
                     "font.family":"DejaVu Sans"})
FIG=os.environ.get("MERIT_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
det=[("degraded",1,0),("Wiener\n(oracle)",9,10),("NAFNet-big",23,24),("MERIT\n(ours)",65,26)]
fig,ax=plt.subplots(figsize=(5.2,4.0)); xs=np.arange(len(det))
cols=["#7f7f7f","#ff7f0e","#8fbf8f","#2ca02c"]
ax.bar(xs,[d[1] for d in det],yerr=[d[2] for d in det],capsize=4,color=cols,edgecolor="k",linewidth=0.6)
ax.set_xticks(xs); ax.set_xticklabels([d[0] for d in det])
ax.set_ylabel("Detection-fitness recovery (% of lost, N=12)")
for i,d in enumerate(det):
    ax.text(i,d[1]+d[2]+1.5,f"+{d[1]}%",ha="center",fontsize=9,fontweight="bold" if d[0].startswith("MERIT") else "normal")
ax.set_ylim(0,100)
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F4_detection_recovery.{ext}"))
print("saved F4_detection_recovery (no title)")
