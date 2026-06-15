# -*- coding: utf-8 -*-
"""Fig.10 detection-recovery bar — FULL battery incl diffusion. Genuine-recovery methods on a 0-100 axis;
DiffBIR shown off-scale because its +1126% is fabricated (hallucinated) detections, not recovery.
Values are the established per-method means/stds (see full_battery_downstream.csv + diffusion_det)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl, numpy as np
mpl.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})
FIG=os.environ.get("MERIT_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
# (label, mean, std, color)  — ascending genuine recovery; DiffBIR handled off-scale separately
GEN=[("unsharp",-1,1,"#9467bd"),("SD-x4",1,3,"#7d3fb0"),("NL-means",2,1,"#5fa3d6"),
     ("unsup.\nWiener",2,4,"#c2691a"),("TV-\ndenoise",3,1,"#1f77b4"),("wrong-PSF\nWiener",5,8,"#a85b16"),
     ("Wiener",9,10,"#e08214"),("Richardson-\nLucy",13,8,"#d6a14a"),("NAFNet-\nbig",23,24,"#8fbf8f"),
     ("MERIT",65,26,"#2ca02c")]
DB=("DiffBIR",1126,1129,"#c0392b")
labels=[g[0] for g in GEN]+[DB[0]]; xs=np.arange(len(labels))
fig,ax=plt.subplots(figsize=(9.6,4.4))
for i,(lab,m,s,c) in enumerate(GEN):
    ax.bar(i,m,yerr=s,capsize=4,color=c,edgecolor="k",linewidth=0.6)
    ax.text(i,m+s+1.5,f"{m:+.0f}%",ha="center",fontsize=8.5,fontweight="bold" if lab=="MERIT" else "normal")
# DiffBIR off-scale: clipped hatched bar
ax.bar(len(GEN),98,color=DB[3],edgecolor="k",linewidth=0.8,hatch="///",alpha=0.85)
ax.text(len(GEN),99.5,"+1126%",ha="center",va="bottom",fontsize=9,color=DB[3],fontweight="bold")
ax.text(len(GEN),49,"fabricated cracks",ha="center",va="center",fontsize=9,color="white",fontweight="bold",rotation=90)
# axis-break marks on the DiffBIR bar top
ax.set_xticks(xs); ax.set_xticklabels(labels,fontsize=8.5)
ax.set_ylabel("Detection-fitness recovery (% of lost, N=12)"); ax.set_ylim(min(-5,-3),105)
ax.axhline(0,color="0.6",lw=0.8)
ax.set_title("genuine recovery (0–100 axis); DiffBIR off-scale — its detections are fabricated, not recovered",fontsize=9,color="0.3",pad=6)
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F4_detection_recovery.{ext}"))
print("saved F4_detection_recovery (full battery + diffusion)")
