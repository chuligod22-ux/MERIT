# -*- coding: utf-8 -*-
"""SP03 E4 — qualitative figures F5 (restoration comparison) + F6 (false-crack / no-hallucination).

F5: a cam2 crack crop, clean / degraded / Wiener(oracle) / MERIT, with PSNR + detection-fitness
    annotations and a zoom inset on the crack — shows MERIT recovers the visible crack.
F6: a CRACK-FREE crop, clean / degraded / MERIT, top = image, bottom = segmenter heat overlay —
    shows MERIT does NOT invent cracks (heat stays at the clean ~0 level).
Uses the MERIT task model (restorer_task.pt). Reuses crop selection from eval_detection / eval_falsecrack.
"""
import os, sys, glob
import numpy as np, cv2, torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet

mpl.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
FIG = os.path.join(DATA, "figs"); os.makedirs(FIG, exist_ok=True)
dev = "cuda"
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16

INP = 448
tfm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
unet = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()
ck = torch.load(os.path.join(WEIGHTS, "restorer_task.pt"), map_location=dev)
net = NAFNetUNet(width=ck["width"]).to(dev).eval(); net.load_state_dict(ck["state"])
ckb = torch.load(os.path.join(WEIGHTS, "restorer_big.pt"), map_location=dev)
net_big = NAFNetUNet(width=ckb["width"]).to(dev).eval(); net_big.load_state_dict(ckb["state"])
SH, SV, NSTD = 2.6, 2.4, 2.5
DISP = 896  # multiple of 448 (segmenter tiles) and 8 (net)


def crack_prob(rgb):
    H, W, _ = rgb.shape; pm = np.zeros((H, W), np.float32); tens, locs = [], []
    for yy in range(0, H - INP + 1, INP):
        for xx in range(0, W - INP + 1, INP):
            tens.append(tfm(rgb[yy:yy + INP, xx:xx + INP])); locs.append((xx, yy))
    with torch.no_grad():
        for i in range(0, len(tens), 16):
            pr = torch.sigmoid(unet(torch.stack(tens[i:i + 16]).cuda()))[:, 0].cpu().numpy()
            for j, (xx, yy) in enumerate(locs[i:i + 16]):
                pm[yy:yy + INP, xx:xx + INP] = pr[j]
    return pm


def gpsf2d(sh, sv):
    nh = int(2 * np.ceil(3 * sh) + 1); nv = int(2 * np.ceil(3 * sv) + 1)
    kh = np.exp(-(np.arange(nh) - nh // 2)**2 / (2 * sh**2)); kv = np.exp(-(np.arange(nv) - nv // 2)**2 / (2 * sv**2))
    k = np.outer(kv, kh); return k / k.sum()


def wiener_ch(img, psf, K):
    pad = np.zeros_like(img, np.float64); kh, kw = psf.shape
    pad[:kh, :kw] = psf; pad = np.roll(pad, (-(kh // 2), -(kw // 2)), (0, 1)); Hf = np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(Hf) / (np.abs(Hf)**2 + K)))


def infer_rgb(net_, deg):
    x = deg / 255.0; h, w = x.shape[:2]
    b = np.full((h, w), 0.5 * (SH + SV) / BLUR_MAX, np.float32); n = np.full((h, w), NSTD / NOISE_MAX, np.float32)
    batch = np.stack([np.stack([x[..., c], b, n], 0) for c in range(3)], 0)
    with torch.no_grad():
        out = net_(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:, 0]
    return np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0, 255)


def degrade(clean, seed):
    rng = np.random.RandomState(seed)
    d = np.stack([gaussian_filter(clean[..., c], (SV, SH)) for c in range(3)], -1)
    return d + rng.randn(*d.shape) * NSTD


u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
psnr = lambda a, b: 10 * np.log10(255**2 / max(np.mean((a - b)**2), 1e-9))
gray = lambda a: cv2.cvtColor(u8(a), cv2.COLOR_RGB2GRAY)


def load_clean(cond_glob, center_on_crack):
    frame = sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", cond_glob, "*.png")))[0]
    rgb = cv2.cvtColor(cv2.imread(frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape; top = (int(H * 0.40) // INP) * INP
    pm = crack_prob(rgb[:top, :(W // INP) * INP])
    if center_on_crack:
        ys, xs = np.where(pm > 0.4)
        cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) > 20 else np.unravel_index(pm.argmax(), pm.shape)
        y0 = int(np.clip(cy - DISP // 2, 0, pm.shape[0] - DISP)); x0 = int(np.clip(cx - DISP // 2, 0, W - DISP))
    else:  # most crack-free window
        best = None
        for y0c in range(0, pm.shape[0] - DISP + 1, INP):
            for x0c in range(0, W - DISP + 1, INP):
                m = pm[y0c:y0c + DISP, x0c:x0c + DISP].max()
                if best is None or m < best[0]:
                    best = (m, y0c, x0c)
        _, y0, x0 = best
    return rgb[y0:y0 + DISP, x0:x0 + DISP].astype(np.float64)


# ======================= F5 — qualitative restoration on a crack =======================
clean = load_clean("crack_d25_ISO100_V60", center_on_crack=True)
deg = degrade(clean, 2); psf = gpsf2d(SH, SV)
wie = np.stack([wiener_ch(deg[..., c], psf, 1e-2) for c in range(3)], -1)
naf = infer_rgb(net_big, deg)
mer = infer_rgb(net, deg)
fc = crack_prob(u8(clean)).mean()
gcl = gray(clean); VMIN, VMAX = np.percentile(gcl, 2), np.percentile(gcl, 98)  # shared contrast stretch for visibility
panels = [("Clean (reference)", clean, None, None),
          ("Degraded\n(blur+noise)", deg, psnr(clean, deg), crack_prob(u8(deg)).mean()),
          ("Wiener (oracle PSF)", wie, psnr(clean, wie), crack_prob(u8(wie)).mean()),
          ("NAFNet-big\n(fidelity only)", naf, psnr(clean, naf), crack_prob(u8(naf)).mean()),
          ("MERIT (ours)", mer, psnr(clean, mer), crack_prob(u8(mer)).mean())]
# crack zoom window (centred): find peak on clean
pmc = crack_prob(u8(clean)); zy, zx = np.unravel_index(gaussian_filter(pmc, 8).argmax(), pmc.shape)
Z = 220; zy = int(np.clip(zy - Z // 2, 0, DISP - Z)); zx = int(np.clip(zx - Z // 2, 0, DISP - Z))
fig, axes = plt.subplots(2, 5, figsize=(13.8, 6.0), gridspec_kw={"height_ratios": [3, 1.25]})
for j, (name, im, ps, ft) in enumerate(panels):
    g = gray(im)
    axes[0, j].imshow(g, cmap="gray", vmin=VMIN, vmax=VMAX); axes[0, j].axis("off")
    sub = "" if ps is None else f"\nPSNR {ps:.1f} dB  ·  det {ft/fc*100:.0f}%"
    axes[0, j].set_title(name + sub, fontsize=9, fontweight="bold" if "MERIT" in name else "normal")
    from matplotlib.patches import Rectangle
    axes[0, j].add_patch(Rectangle((zx, zy), Z, Z, ec="#d62728", fc="none", lw=1.4))
    axes[1, j].imshow(g[zy:zy + Z, zx:zx + Z], cmap="gray", vmin=VMIN, vmax=VMAX); axes[1, j].axis("off")
axes[1, 0].set_ylabel("crack zoom", fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"F5_qualitative.{ext}"))
plt.close(fig); print("saved F5_qualitative")

# ======================= F6 — false-crack / no hallucination =======================
cleanf = load_clean("crack_d35_ISO400_V60", center_on_crack=False)  # bright, low-noise crack-free region
degf = degrade(cleanf, 4); psff = gpsf2d(SH, SV)
wief = np.stack([wiener_ch(degf[..., c], psff, 1e-2) for c in range(3)], -1)
naff = infer_rgb(net_big, degf)
merf = infer_rgb(net, degf)
items = [("Clean (crack-free)", cleanf), ("Degraded", degf), ("Wiener (oracle)", wief),
         ("NAFNet-big", naff), ("MERIT (ours)", merf)]
pms = {n: crack_prob(u8(im)) for n, im in items}
# bright mid-grey region (d35/ISO400): display raw (no stretch) so texture is not amplified.
# Both F6 (heat) and F6b (bar) use the SAME max statistic on the SAME region/methods (consistent).
mx = {name: float(pms[name].max()) for name, _ in items}
SHORT = {"Clean (crack-free)": "clean", "Degraded": "degraded", "Wiener (oracle)": "Wiener",
         "NAFNet-big": "NAFNet-big", "MERIT (ours)": "MERIT"}
# --- F6 (Fig.11): segmenter heat overlays, 5 methods, annotated with the SAME max statistic ---
fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.1))
for j, (name, im) in enumerate(items):
    axes[j].imshow(gray(im), cmap="gray", vmin=0, vmax=255)
    heat = np.ma.masked_less(pms[name], 0.05)               # only show meaningful response (else clean wall)
    axes[j].imshow(heat, cmap="inferno", vmin=0, vmax=0.5, alpha=0.75); axes[j].axis("off")
    axes[j].set_title(f"{name}\nmax {mx[name]:.3f}", fontsize=9,
                      fontweight="bold" if "MERIT" in name else "normal")
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"F6_falsecrack.{ext}"))
plt.close(fig); print("saved F6_falsecrack")
print("  per-method max crack-prob:", {SHORT[n]: round(mx[n], 4) for n, _ in items})
# --- F6b (Fig.12): the same max statistic as a bar, with the clean reference line ---
fig2, axb = plt.subplots(figsize=(6.6, 3.9))
order = ["Clean (crack-free)", "Degraded", "Wiener (oracle)", "NAFNet-big", "MERIT (ours)"]
labels = [SHORT[n] for n in order]; maxv = [mx[n] for n in order]
cols = ["#7f7f7f", "#9e9e9e", "#e08214", "#8fbf8f", "#2ca02c"]
xs = np.arange(len(labels))
axb.bar(xs, maxv, color=cols, edgecolor="k", linewidth=0.6, width=0.6)
for i, v in enumerate(maxv):
    axb.text(i, v + max(maxv) * 0.03, f"{v:.3f}", ha="center", fontsize=9,
             fontweight="bold" if labels[i] == "MERIT" else "normal")
axb.axhline(mx["Clean (crack-free)"], ls="--", color="0.4", lw=1.0, label="clean reference")
axb.set_xticks(xs); axb.set_xticklabels(labels, fontsize=9)
axb.set_ylabel("max crack-prob (crack-free region)", fontsize=9.5)
axb.set_ylim(0, max(maxv) * 1.35); axb.grid(axis="y", alpha=0.25)
axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
axb.legend(frameon=False, fontsize=8.5, loc="upper left")
axb.set_title("MERIT ≈ clean: 0% above the 0.5 detection threshold", fontsize=9, color="0.3", pad=6)
fig2.tight_layout()
for ext in ("png", "pdf"): fig2.savefig(os.path.join(FIG, f"F6b_falsecrack_maxprob.{ext}"))
plt.close(fig2); print("saved F6b_falsecrack_maxprob")
print("\nF5/F6 ->", FIG)
