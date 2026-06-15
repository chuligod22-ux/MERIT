# -*- coding: utf-8 -*-
"""Fig: the 16 chart-ROI images of the MTF50 eval (4 blur levels x 4 noise seeds).
Grid = 4 rows (blur sigma_h = 1.5/2.0/2.5/3.0) x 4 cols (noise seeds 0-3) of degraded
inputs, plus a 5th column with the MERIT reconstruction. Per-panel MTF50 (cy/px).
Saves F9_eval_strip.{png,pdf}."""
import os, sys, glob
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA  = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
CKPT  = os.path.join(WEIGHTS, "restorer_task.pt")
dev = "cuda"
ck = torch.load(CKPT, map_location=dev)
net = (NAFNetUNet(width=ck["width"]) if ck.get("arch") == "unet"
       else NAFNetRestorer(width=ck["width"], n_blocks=ck["n_blocks"])).to(dev).eval()
net.load_state_dict(ck["state"])

CAM1 = glob.glob(os.path.join(NRIQA,"04_data","raw","cam1","60km_2.5m_ISO100","MTF*","frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64); Mg = 48
big = img[158-Mg:554+Mg, 937-Mg:1133+Mg]
inner = lambda a: a[Mg:Mg+396, Mg:Mg+196]
clean = inner(big); m_sharp = esf_mtf(clean)[2]
crop = lambda a: a[120:300]          # display band across the vertical edge (measurement uses full ROI)

def naf1(deg, sh, sv, nstd):
    xi = deg/255.0; b = np.full_like(xi, 0.5*(sh+sv)/BLUR_MAX); n = np.full_like(xi, nstd/NOISE_MAX)
    x = torch.from_numpy(np.stack([xi,b,n],0)[None]).float().to(dev)
    Hh,Ww = x.shape[-2:]; ph,pw = (8-Hh%8)%8,(8-Ww%8)%8
    xp = torch.nn.functional.pad(x,(0,pw,0,ph),mode="reflect")
    with torch.no_grad(): return net(xp)[0,0,:Hh,:Ww].cpu().numpy()*255.0

LEV = [1.5, 2.0, 2.5, 3.0]; SEEDS = [0, 1, 2, 3]
deg = {}; m_deg = {}; res = {}; m_res = {}
for sb in LEV:
    for sd in SEEDS:
        d = gaussian_filter(big, sigma=(sb*0.9, sb)) + np.random.RandomState(sd).randn(*big.shape)*2.5
        deg[(sb,sd)] = inner(d); m_deg[(sb,sd)] = esf_mtf(inner(d))[2]
    r = naf1(gaussian_filter(big, sigma=(sb*0.9, sb)) + np.random.RandomState(0).randn(*big.shape)*2.5, sb, sb*0.9, 2.5)
    res[sb] = inner(r); m_res[sb] = esf_mtf(inner(r))[2]

nrow, ncol = len(LEV), len(SEEDS)+2   # clean ref + 4 degraded seeds + MERIT
fig = plt.figure(figsize=(1.45*ncol, 1.7*nrow+0.2))
gs = GridSpec(nrow, ncol, hspace=0.07, wspace=0.05, left=0.075, right=0.995, top=0.955, bottom=0.015)
VMIN, VMAX = np.percentile(clean,2), np.percentile(clean,98)
def panel(r,c,arr,mtf,col,title=None):
    ax = fig.add_subplot(gs[r,c]); ax.imshow(crop(arr), cmap="gray", vmin=VMIN, vmax=VMAX, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title,fontsize=10,color=col,pad=3)
    ax.text(0.5,0.04,f"{mtf:.3f}",transform=ax.transAxes,ha="center",va="bottom",fontsize=9,color="white",
            fontweight="bold",bbox=dict(boxstyle="round,pad=0.15",fc=col,ec="none",alpha=0.85))
    for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(1.4)
for i,sb in enumerate(LEV):
    panel(i,0,clean,m_sharp,"#222", "clean (ref)" if i==0 else None)
    for j,sd in enumerate(SEEDS):
        panel(i,j+1,deg[(sb,sd)],m_deg[(sb,sd)],"#a33", f"seed {sd}" if i==0 else None)
    panel(i,ncol-1,res[sb],m_res[sb],"#1a6e1a", "MERIT" if i==0 else None)
    fig.text(0.026,0.955-(i+0.5)/nrow*0.94,f"$\\sigma_h$={sb}",ha="center",va="center",fontsize=10,fontweight="bold",rotation=90)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(DATA,"figs",f"F9_eval_strip.{ext}"), dpi=170 if ext=="png" else None, bbox_inches="tight")
print("clean", round(m_sharp,4))
for sb in LEV:
    print(f"sb={sb}: deg seeds {[round(m_deg[(sb,sd)],4) for sd in SEEDS]}  MERIT {round(m_res[sb],4)}")
print("saved F9_eval_strip (16 inputs + MERIT)")
