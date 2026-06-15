# -*- coding: utf-8 -*-
"""§4.1 data-acquisition figure for MERIT (displayed as Fig. 2).
(a) inspection platform (vehicle photo + cross-section; internal sub-labels / wiring panel cropped out).
(b,c) cam1 ISO 12233 chart — best vs marginal (reference synthetic degradation). A box at the chart
      CENTRE marks the slant-edge region where MTF50 is measured; the quantitative difference
      (MTF50 0.077 -> ~0.039 cy/px) is given in the caption.
(d,e) cam2 lining-crack crop — best vs marginal (the visual difference is clear; no box).
Degradation = anisotropic Gaussian blur (sigma_v 2.4 / sigma_h 2.6 px) + Gaussian noise (sigma_n 2.5 DN).
Sub-captions at the bottom-centre of each panel.
"""
import os, sys, glob
import numpy as np, cv2
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

ROOT  = os.environ.get("MERIT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
NRIQA = os.path.join(ROOT, "02_nriqa")
SYS   = os.path.join(ROOT, "01_tunnelscanning", "01_paper", "output", "ieee_tim_v2", "figures", "F1_system_configuration.png")
CAM1  = sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "*.png")))[0]
CAM2  = sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", "crack_d25_ISO100_V60", "*.png")))[0]
OUT   = os.path.join(ROOT, "03_restoration", "04_data", "figs")
INP, DISP = 448, 896
SH, SV, NSTD = 2.6, 2.4, 2.5


def rgb(p):
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def degrade(clean):
    rng = np.random.RandomState(0)
    d = np.stack([gaussian_filter(clean[..., c].astype(np.float64), (SV, SH)) for c in range(3)], -1)
    return np.clip(d + rng.randn(*d.shape) * NSTD, 0, 255).astype(np.uint8)


# (a) platform — top row only
sysimg = rgb(SYS); Hs = sysimg.shape[0]; sysimg = sysimg[0:int(0.355 * Hs), :]

# (b,c) cam1 chart ROI
c1 = rgb(CAM1); H1, W1 = c1.shape[:2]
chart = c1[int(0.02 * H1):int(0.52 * H1), int(0.02 * W1):int(0.40 * W1)]
chart_deg = degrade(chart)

# (d,e) cam2 crack crop (segmenter-centred); no box
c2 = rgb(CAM2); H2, W2 = c2.shape[:2]
try:
    import torch, torchvision.transforms as T
    CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
    from utils import load_unet_vgg16
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    unet = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()
    top = (int(H2 * 0.55) // INP) * INP; Wc = (W2 // INP) * INP
    pm = np.zeros((top, Wc), np.float32)
    with torch.no_grad():
        for yy in range(0, top - INP + 1, INP):
            for xx in range(0, Wc - INP + 1, INP):
                t = tfm(c2[yy:yy + INP, xx:xx + INP]).unsqueeze(0).cuda()
                pm[yy:yy + INP, xx:xx + INP] = torch.sigmoid(unet(t))[0, 0].cpu().numpy()
    ys, xs = np.where(pm > 0.4)
    cyk, cxk = (int(np.median(ys)), int(np.median(xs))) if len(ys) > 20 else np.unravel_index(pm.argmax(), pm.shape)
    y0 = int(np.clip(cyk - DISP // 2, 0, top - DISP)); x0 = int(np.clip(cxk - DISP // 2, 0, Wc - DISP))
    crack = c2[y0:y0 + DISP, x0:x0 + DISP]
    print("cam2 crack centred via segmenter")
except Exception as e:
    print("segmenter unavailable, centre crop:", e)
    crack = c2[H2 // 2 - DISP // 2:H2 // 2 + DISP // 2, W2 // 2 - DISP // 2:W2 // 2 + DISP // 2]
crack_deg = degrade(crack)


def subcap(ax, text):
    ax.text(0.5, -0.05, text, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)


fig = plt.figure(figsize=(8.6, 10.2))
gs = GridSpec(3, 2, height_ratios=[0.62, 1.0, 1.0], hspace=0.22, wspace=0.06,
              left=0.015, right=0.985, top=0.99, bottom=0.03)
axA = fig.add_subplot(gs[0, :]); axA.imshow(sysimg); axA.axis("off")
subcap(axA, "(a) Dual-camera high-speed mobile inspection platform (cam1 → resolution chart, cam2 → tunnel-lining wall)")

cam = [(gs[1, 0], chart, "(b) cam1 — best condition (ISO 12233 chart)"),
       (gs[1, 1], chart_deg, "(c) cam1 — marginal condition (MTF50 ≈ 50% of best)"),
       (gs[2, 0], crack, "(d) cam2 — best condition (tunnel-lining crack)"),
       (gs[2, 1], crack_deg, "(e) cam2 — marginal condition (detection fitness ≈ 1% of best)")]
for cell, img, cap in cam:
    ax = fig.add_subplot(cell); ax.imshow(img); ax.axis("off"); subcap(ax, cap)
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor("k"); s.set_linewidth(0.6)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"F0_acquisition.{ext}"), dpi=200, bbox_inches="tight")
print("saved F0_acquisition (no boxes; relative sub-captions)")
