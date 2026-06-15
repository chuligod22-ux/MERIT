# -*- coding: utf-8 -*-
"""Regenerate ONLY F1_mpd_triangle (no title; no axis arrows; legend moved off labels; size=PSNR). Others untouched."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.25,
                     "font.family":"DejaVu Sans"})
DATA=os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")); FIG=os.path.join(DATA,"figs")
rows=list(csv.DictReader(open(os.path.join(DATA,"mpd_axes.csv"))))
M={r["method"]:dict(PSNR=float(r["PSNR"]),LPIPS=float(r["LPIPS"]),MTFerr=float(r["MTF_err"])) for r in rows}

fig,ax=plt.subplots(figsize=(6.2,4.6))
groups={"denoise-only":(["TV denoise","NL-means"],"#1f77b4"),
        "oracle deconv":(["Wiener (oracle)","Richardson-Lucy"],"#ff7f0e"),
        "naive/degraded":(["degraded","unsharp"],"#7f7f7f"),
        "generative (diffusion)":(["SD-x4 diffusion","DiffBIR v2.1"],"#9467bd"),
        "learned (ours)":(["NAFNet-big","MERIT (ours)"],"#2ca02c")}
F1_OFF={"degraded":(8,-12,"left"),"DiffBIR v2.1":(2,8,"center"),
        "NL-means":(6,-11,"left"),"TV denoise":(6,4,"left")}
pmin,pmax=min(v["PSNR"] for v in M.values()),max(v["PSNR"] for v in M.values())
for label,(names,col) in groups.items():
    xs=[M[n]["LPIPS"] for n in names]; ys=[M[n]["MTFerr"] for n in names]
    ss=[60+340*(M[n]["PSNR"]-pmin)/(pmax-pmin) for n in names]
    ax.scatter(xs,ys,s=ss,c=col,alpha=0.8,edgecolors="k",linewidths=0.6,label=label,zorder=3)
    for n in names:
        tag="MERIT" if "MERIT" in n else n.split(" (")[0]
        dx,dy,ha=F1_OFF.get(tag,(5,4,"left"))
        ax.annotate(tag,(M[n]["LPIPS"],M[n]["MTFerr"]),xytext=(dx,dy),ha=ha,
                    textcoords="offset points",fontsize=8,fontweight="bold" if "MERIT" in n else "normal")
ax.set_xlabel("Perception  —  LPIPS to clean  (lower better)")
ax.set_ylabel("Measurement  —  |MTF50 error|  (lower better)")
ax.legend(fontsize=7.5,loc="center right",bbox_to_anchor=(1.0,0.60),ncol=1,
          framealpha=0.9,markerscale=0.45,handletextpad=0.4,labelspacing=0.6)
ax.set_ylim(-0.003,max(v["MTFerr"] for v in M.values())*1.15)
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F1_mpd_triangle.{ext}"))
print("saved F1_mpd_triangle (no title/arrows, legend repositioned)")
