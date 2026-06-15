# -*- coding: utf-8 -*-
"""
SP03 — backbone-agnosticism evaluation. ONE identical protocol applied to every (backbone, mode):

  Measurement : chart ROI MTF50 recovery (% of clean), ISO-12233 e-SFR        [eval_mpd protocol]
  Task        : crack-detection-fitness recovery on a held-out cam2 crack crop [eval_detection protocol]
  Hallucination: max false-crack probability on a crack-free crop              [eval_falsecrack protocol]

Same degraded inputs (fixed seeds) for all models, so ON-vs-OFF isolates the framework and the
spread across ON rows isolates the backbone. Reads restorer_<bk>_<mode>.pt written by train_backbone.py.
"""
import os, sys, glob, warnings, csv
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet
from model_backbones import PlainUNet, RestormerLite, MambaLite

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
dev = "cuda"
sh, sv, nstd = 2.6, 2.4, 2.5


def load_net(ckpt):
    ck = torch.load(ckpt, map_location=dev)
    arch, w, in_ch = ck["arch"], ck["width"], ck["in_ch"]
    net = {"nafnet": NAFNetUNet, "plainunet": PlainUNet, "restormer": RestormerLite, "mamba": MambaLite}[arch](in_ch=in_ch, width=w)
    net.load_state_dict(ck["state"]); net.to(dev).eval()
    return net, in_ch


def restore_gray(net, in_ch, deg):
    """deg: HxW float [0,255] -> restored HxW float [0,255]. Builds 3ch prompt if in_ch==3."""
    xi = np.clip(deg, 0, 255) / 255.0
    if in_ch == 3:
        b = np.full_like(xi, 0.5 * (sh + sv) / BLUR_MAX); n = np.full_like(xi, nstd / NOISE_MAX)
        arr = np.stack([xi, b, n], 0)
    else:
        arr = xi[None]
    x = torch.from_numpy(arr[None]).float().to(dev)
    H, W = x.shape[-2:]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        return net(xp)[0, 0, :H, :W].cpu().numpy() * 255.0


def restore_rgb(net, in_ch, deg):
    """deg: HxWx3 -> restored HxWx3 (per-channel grayscale restoration, like eval_detection)."""
    return np.clip(np.stack([restore_gray(net, in_ch, deg[..., c]) for c in range(3)], -1), 0, 255)


# ===================== Measurement: chart ROI MTF50 =====================
CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64)
Mg = 48
inner = lambda a: a[Mg:Mg + 396, Mg:Mg + 196]
big = img[158 - Mg:554 + Mg, 937 - Mg:1133 + Mg]
clean_chart = inner(big); _, _, m_clean = esf_mtf(clean_chart)
deg_chart = gaussian_filter(big, sigma=(sv, sh)) + np.random.RandomState(5).randn(*big.shape) * nstd
_, _, m_deg = esf_mtf(inner(deg_chart))


def mtf_pct(net, in_ch):
    r = restore_gray(net, in_ch, deg_chart)
    _, _, m = esf_mtf(inner(r))
    return m, m / m_clean * 100.0


# ===================== Task + Hallucination: segmenter =====================
INP = 448
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16
tfm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
seg = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()
u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)


def crack_pm(rgb):
    H, W, _ = rgb.shape; pm = np.zeros((H, W), np.float32); tens, locs = [], []
    for yy in range(0, H - INP + 1, INP):
        for xx in range(0, W - INP + 1, INP):
            tens.append(tfm(rgb[yy:yy + INP, xx:xx + INP])); locs.append((xx, yy))
    with torch.no_grad():
        for i in range(0, len(tens), 16):
            pr = torch.sigmoid(seg(torch.stack(tens[i:i + 16]).cuda()))[:, 0].cpu().numpy()
            for j, (xx, yy) in enumerate(locs[i:i + 16]):
                pm[yy:yy + INP, xx:xx + INP] = pr[j]
    return pm


fitness = lambda rgb: float(crack_pm(rgb).mean())

# --- crack crop (eval_detection protocol) ---
import pandas as pd
cq = pd.read_csv(os.path.join(NRIQA, "04_data", "composite_Q.csv"))
sel = cq[cq.cond == "crack_d25_ISO100_V60"].sort_values("fit_mean_top", ascending=False).iloc[0]
rgb = cv2.cvtColor(cv2.imread(os.path.join(NRIQA, "04_data", "raw", "cam2", sel.cond, sel.frame), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
H, W, _ = rgb.shape; top = int((H // INP) * INP * 0.40)
pm0 = crack_pm(rgb[:((top // INP) * INP), :((W // INP) * INP)])
ys, xs = np.where(pm0 > 0.4)
cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) > 20 else np.unravel_index(pm0.argmax(), pm0.shape)
CH, CW = 1344, 1792
y0 = int(np.clip(cy - CH // 2, 0, pm0.shape[0] - CH)); x0 = int(np.clip(cx - CW // 2, 0, W - CW))
crack_clean = rgb[y0:y0 + CH, x0:x0 + CW].astype(np.float64)
rng2 = np.random.RandomState(2)
crack_deg = np.stack([gaussian_filter(crack_clean[..., c], (sv, sh)) for c in range(3)], -1) + rng2.randn(CH, CW, 3) * nstd
f_clean = fitness(u8(crack_clean)); f_deg = fitness(u8(crack_deg))


def det_recovery(net, in_ch):
    r = restore_rgb(net, in_ch, crack_deg); fr = fitness(u8(r))
    return (fr - f_deg) / (f_clean - f_deg + 1e-9) * 100.0

# --- crack-free crop (eval_falsecrack protocol) ---
frame = sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", "crack_d25_ISO100_V60", "*.png")))[0]
rgbf = cv2.cvtColor(cv2.imread(frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
Hf, Wf, _ = rgbf.shape; topR = int((Hf // INP) * INP * 0.40); best = None
for yy in range(0, topR - INP + 1, INP):
    for xx in range(0, Wf - INP + 1, INP):
        m = crack_pm(rgbf[yy:yy + INP, xx:xx + INP]).max()
        if best is None or m < best[0]:
            best = (float(m), yy, xx)
_, fy, fx = best
cf_clean = rgbf[fy:fy + INP, fx:fx + INP].astype(np.float64)
cf_deg = np.stack([gaussian_filter(cf_clean[..., c], (sv, sh)) for c in range(3)], -1) + np.random.RandomState(6).randn(INP, INP, 3) * nstd
cf_clean_max = crack_pm(u8(cf_clean)).max()


def false_crack_max(net, in_ch):
    r = restore_rgb(net, in_ch, cf_deg)
    return float(crack_pm(u8(r)).max())


# ===================== run all models =====================
print(f"clean chart MTF50={m_clean:.3f} (deg {m_deg:.3f})  clean det fitness={f_clean:.5f} (deg {f_deg:.5f}, {f_deg/f_clean*100:.0f}%)")
print(f"crack-free clean max-prob={cf_clean_max:.3f}\n")
print(f"{'backbone':12s}{'mode':5s}{'params':>8s}{'MTF50':>8s}{'%clean':>8s}{'det.rec':>9s}{'fc-max':>8s}")
rows = []
for bk in ["nafnet", "plainunet", "restormer", "mamba"]:
    for md in ["on", "off"]:
        ck = os.path.join(WEIGHTS, f"restorer_{bk}_{md}.pt")
        if not os.path.exists(ck):
            print(f"  {bk:12s}{md:5s}  (missing)"); continue
        net, in_ch = load_net(ck)
        npar = sum(p.numel() for p in net.parameters()) / 1e6
        m, mp = mtf_pct(net, in_ch); dr = det_recovery(net, in_ch); fc = false_crack_max(net, in_ch)
        rows.append([bk, md, f"{npar:.2f}", f"{m:.3f}", f"{mp:.0f}", f"{dr:+.0f}", f"{fc:.3f}"])
        print(f"  {bk:12s}{md:5s}{npar:7.2f}M{m:8.3f}{mp:7.0f}%{dr:+8.0f}%{fc:8.3f}")

with open(os.path.join(DATA, "backbone_axes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["backbone", "mode", "params_M", "MTF50", "pct_clean", "det_recovery_pct", "falsecrack_max"])
    w.writerow(["_clean", "-", "-", f"{m_clean:.4f}", "100", "-", f"{cf_clean_max:.4f}"])
    w.writerow(["_degraded", "-", "-", f"{m_deg:.4f}", f"{m_deg/m_clean*100:.0f}", "0", "-"])
    w.writerows(rows)
print(f"\nsaved backbone_axes.csv")
print("read: ON rows should all recover MTF toward clean + positive detection + low false-crack (backbone-agnostic);")
print("OFF rows (plain L1, no framework) should under-recover MTF and detection - isolating the framework.")
