# -*- coding: utf-8 -*-
"""Regenerate ONLY F6b_falsecrack_maxprob (hardcoded paper values; no GPU). Shorter note."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl, numpy as np
mpl.rcParams.update({"font.size":10,"figure.dpi":200,"savefig.bbox":"tight","font.family":"DejaVu Sans"})
FIG=os.environ.get("MERIT_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
fig,axb=plt.subplots(figsize=(6.4,3.9))
labels=["clean","MERIT\n(ours)","NAFNet-big","SD-x4","DiffBIR v2.1"]
maxv=[0.092,0.099,0.099,0.139,0.173]; cols=["#7f7f7f","#2ca02c","#8fbf8f","#b07fd0","#7d3fb0"]
xs=np.arange(len(labels))
axb.bar(xs,maxv,color=cols,edgecolor="k",linewidth=0.6,width=0.6)
for i,v in enumerate(maxv): axb.text(i,v+0.004,f"{v:.3f}",ha="center",fontsize=9,fontweight="bold" if i==1 else "normal")
axb.set_xticks(xs); axb.set_xticklabels(labels,fontsize=9)
axb.set_ylabel("max crack-prob (crack-free region)",fontsize=9.5)
axb.set_ylim(0,0.205); axb.grid(axis="y",alpha=0.25)
axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
axb.set_title("all methods 0% above the 0.5 detection threshold",fontsize=9,color="0.3",pad=6)
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F6b_falsecrack_maxprob.{ext}"))
print("saved F6b_falsecrack_maxprob (short note)")
