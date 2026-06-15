# -*- coding: utf-8 -*-
"""Regenerate ONLY F8_classical_battery (no title; de-overlapped labels). Other figures untouched."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.25,
                     "font.family":"DejaVu Sans"})
DATA=os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")); FIG=os.path.join(DATA,"figs")
rows=list(csv.DictReader(open(os.path.join(DATA,"mpd_axes.csv"))))
M={r["method"]:dict(PSNR=float(r["PSNR"]),LPIPS=float(r["LPIPS"]),MTFerr=float(r["MTF_err"])) for r in rows}

fig,ax=plt.subplots(figsize=(6.0,4.4))
order=["degraded","unsharp","Wiener (oracle)","Richardson-Lucy","TV denoise","NL-means",
       "SD-x4 diffusion","DiffBIR v2.1","NAFNet-big","MERIT (ours)"]
LBL_OFF={"DiffBIR v2.1":(-3,9,"center"),"degraded":(7,-13,"left"),"NL-means":(6,-11,"left")}
for n in order:
    is_ours="MERIT" in n
    ax.scatter(M[n]["PSNR"],M[n]["MTFerr"],s=150 if is_ours else 80,
               c="#2ca02c" if is_ours else "#1f77b4",marker="*" if is_ours else "o",
               edgecolors="k",linewidths=0.6,zorder=3)
    tag="MERIT" if is_ours else n.split(" (")[0]
    dx,dy,ha=LBL_OFF.get(tag,(4,3,"left"))
    ax.annotate(tag,(M[n]["PSNR"],M[n]["MTFerr"]),xytext=(dx,dy),ha=ha,textcoords="offset points",
                fontsize=7.5,fontweight="bold" if is_ours else "normal")
ax.set_xlabel("Distortion fidelity  —  PSNR to clean (dB, higher better)")
ax.set_ylabel("Measurement  —  |MTF50 error| (lower better)")
ax.set_ylim(-0.003,max(M[n]["MTFerr"] for n in order)*1.15)
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F8_classical_battery.{ext}"))
print("saved F8_classical_battery (no title, de-overlapped)")
