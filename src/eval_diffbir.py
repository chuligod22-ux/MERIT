# -*- coding: utf-8 -*-
"""
SP03 D (reviewer attack D) — DiffBIR v2.1 baseline on the MPD + false-crack protocol.

DiffBIR (ECCV'24, generative diffusion prior for blind image restoration) is run via its official
v2.1 inference (denoise task, upscale 1, empty prompt) on the SAME degraded operands used by
eval_mpd.py (chart ROI) and eval_diffusion_falsecrack.py (crack-free cam2 crop). Restored PNGs are
read back from 05_tmp/diffbir_out and scored on the identical three axes:
  Distortion = PSNR(clean)   Perception = LPIPS(clean)   Measurement = |MTF50_restored - MTF50_clean|
plus the false-crack (segmenter) response on the crack-free region.

This supplies a strong, restoration-purpose generative baseline (vs the SD-x4 upscaler), closing the
"weak baseline" gap. Run prep_diffbir_in.py + DiffBIR inference first (see diffbir_run.log).
"""
import os, sys, glob, warnings, csv
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter
import lpips as lpips_lib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from mtf_util import imread_u, esf_mtf

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
DBOUT = os.path.join(os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")), "diffbir_out")
dev = "cuda"
sh, sv, nstd = 2.6, 2.4, 2.5
psnr = lambda a, b: 10 * np.log10(255.0**2 / max(np.mean((np.clip(a, 0, 255) - b)**2), 1e-9))

# ---------- chart ROI (mirrors eval_mpd.py) ----------
CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64)
Mg = 48
inner = lambda a: a[Mg:Mg + 396, Mg:Mg + 196]
big = img[158 - Mg:554 + Mg, 937 - Mg:1133 + Mg]
clean = inner(big); _, _, m_clean = esf_mtf(clean)

lpf = lpips_lib.LPIPS(net="alex").to(dev).eval()


def lpips_d(a, b):
    def t(x):
        x = np.clip(x, 0, 255) / 127.5 - 1.0
        return torch.from_numpy(np.stack([x] * 3)[None]).float().to(dev)
    with torch.no_grad():
        return float(lpf(t(a), t(b)).item())


# DiffBIR restored chart -> grayscale, same size (492x292), take inner ROI
db_chart = cv2.cvtColor(cv2.imread(os.path.join(DBOUT, "chart_deg.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float64)
if db_chart.shape != big.shape:
    db_chart = cv2.resize(db_chart, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_AREA)
ci = inner(db_chart); _, _, m_db = esf_mtf(ci)
P, L, M = psnr(ci, clean), lpips_d(ci, clean), abs(m_db - m_clean)

print(f"clean chart MTF50 = {m_clean:.3f} cy/px\n")
print(f"{'method':18s}{'PSNR(dB)':>10s}{'LPIPS':>9s}{'MTF50':>8s}{'MTFerr':>9s}")
print(f"  {'DiffBIR v2.1':16s}{P:10.1f}{L:9.4f}{m_db:8.3f}{M:9.4f}")
print(f"  (clean MTF50 {m_clean:.3f}; degraded/SD-x4/MERIT rows in mpd_axes.csv)\n")

# ---------- false-crack on crack-free cam2 crop (mirrors eval_diffusion_falsecrack.py) ----------
INP = 448
sys.path.insert(0, os.path.join(NRIQA, "03_src", "crack_seg"))
import torchvision.transforms as T
from utils import load_unet_vgg16
tfm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
unet = load_unet_vgg16(os.path.join(NRIQA, "03_src", "crack_seg", "models", "model_unet_vgg16_best.pt")).cuda().eval()
u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)


def crack_pm(rgb448):
    with torch.no_grad():
        return torch.sigmoid(unet(tfm(rgb448).unsqueeze(0).cuda()))[0, 0].cpu().numpy()


# reconstruct crack-free clean (same selection as prep) + DiffBIR restored
frame = sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", "crack_d25_ISO100_V60", "*.png")))[0]
rgb = cv2.cvtColor(cv2.imread(frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
H, W, _ = rgb.shape; topR = int((H // INP) * INP * 0.40)
best = None
for y0 in range(0, topR - INP + 1, INP):
    for x0 in range(0, W - INP + 1, INP):
        m = crack_pm(rgb[y0:y0 + INP, x0:x0 + INP]).max()
        if best is None or m < best[0]:
            best = (float(m), y0, x0)
_, y0, x0 = best
cf_clean = rgb[y0:y0 + INP, x0:x0 + INP].astype(np.float64)
db_cf = cv2.cvtColor(cv2.imread(os.path.join(DBOUT, "crackfree_deg.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float64)
if db_cf.shape[:2] != (INP, INP):
    db_cf = cv2.resize(db_cf, (INP, INP), interpolation=cv2.INTER_AREA)

print(f"{'image':18s}{'mean prob':>11s}{'max prob':>10s}{'frac>0.5':>10s}")
fc = {}
for name, im in [("clean", cf_clean), ("DiffBIR v2.1", db_cf)]:
    p = crack_pm(u8(im))
    fc[name] = (p.mean(), p.max(), (p > 0.5).mean() * 100)
    print(f"  {name:16s}{p.mean():11.5f}{p.max():10.3f}{(p>0.5).mean()*100:9.3f}%")

# ---------- persist ----------
with open(os.path.join(DATA, "diffbir_axes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["axis", "metric", "value"])
    w.writerow(["chart", "clean_MTF50", f"{m_clean:.4f}"])
    w.writerow(["chart", "DiffBIR_MTF50", f"{m_db:.4f}"])
    w.writerow(["chart", "DiffBIR_PSNR", f"{P:.2f}"])
    w.writerow(["chart", "DiffBIR_LPIPS", f"{L:.4f}"])
    w.writerow(["chart", "DiffBIR_MTFerr", f"{M:.4f}"])
    w.writerow(["falsecrack", "clean_max", f"{fc['clean'][1]:.4f}"])
    w.writerow(["falsecrack", "DiffBIR_mean", f"{fc['DiffBIR v2.1'][0]:.5f}"])
    w.writerow(["falsecrack", "DiffBIR_max", f"{fc['DiffBIR v2.1'][1]:.4f}"])
    w.writerow(["falsecrack", "DiffBIR_frac>0.5", f"{fc['DiffBIR v2.1'][2]:.3f}"])
print(f"\nsaved -> {os.path.join(DATA, 'diffbir_axes.csv')}")
print("read: DiffBIR should look perceptually sharp (decent LPIPS) but MISS the measurement axis")
print("(MTF50 != clean) and/or raise false-crack response = the generative-family failure mode.")
