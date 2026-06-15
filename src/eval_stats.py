# -*- coding: utf-8 -*-
"""
SP03 A — statistical evaluation across many frames / degradation realizations.

(1) Detection recovery over N cam2 crack frames (mostly held-out conditions): clean -> degrade ->
    {Wiener, NAFNet} -> segmenter detection fitness; report mean+-std recovery and win rate.
(2) MTF50 + PSNR over the chart ROI across blur levels x seeds: mean+-std.
"""
import os, sys, glob
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
CKPT = os.environ.get("SP03_CKPT", os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer_big.pt"))
dev = "cuda"
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16

INP = 448
tfm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
unet = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()
ck = torch.load(CKPT, map_location=dev)
net = (NAFNetUNet(width=ck["width"]) if ck.get("arch") == "unet"
       else NAFNetRestorer(width=ck["width"], n_blocks=ck["n_blocks"])).to(dev).eval()
net.load_state_dict(ck["state"])
u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)


def crack_prob(rgb):
    H, W, _ = rgb.shape; pm = np.zeros((H, W), np.float32); tens, locs = [], []
    for yy in range(0, H - INP + 1, INP):
        for xx in range(0, W - INP + 1, INP):
            tens.append(tfm(rgb[yy:yy + INP, xx:xx + INP])); locs.append((xx, yy))
    with torch.no_grad():
        for i in range(0, len(tens), 32):
            pr = torch.sigmoid(unet(torch.stack(tens[i:i + 32]).cuda()))[:, 0].cpu().numpy()
            for j, (xx, yy) in enumerate(locs[i:i + 32]):
                pm[yy:yy + INP, xx:xx + INP] = pr[j]
    return pm


def gpsf(s):
    n = int(2 * np.ceil(3 * s) + 1); k1 = np.exp(-(np.arange(n) - n // 2)**2 / (2 * s**2))
    k = np.outer(k1, k1); return k / k.sum()


def wiener(img, psf, K=1e-2):
    pad = np.zeros_like(img, np.float64); kh, kw = psf.shape; pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1)); Hf = np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(Hf) / (np.abs(Hf)**2 + K)))


def naf_rgb(deg, sh, sv, nstd, CH, CW):
    x = deg / 255.0
    b = np.full((CH, CW), 0.5 * (sh + sv) / BLUR_MAX, np.float32); n = np.full((CH, CW), nstd / NOISE_MAX, np.float32)
    batch = np.stack([np.stack([x[..., c], b, n], 0) for c in range(3)], 0)
    with torch.no_grad():
        out = net(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:, 0]
    return np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0, 255)


# ---------- (1) detection recovery statistics ----------
CH, CW = 896, 1344
conds = ["crack_d25_ISO200_V80", "crack_d35_ISO200_V60", "crack_d35_ISO200_V80",
         "crack_d45_ISO100_V60", "crack_d45_ISO100_V80", "crack_d25_ISO100_V80"]   # mostly held-out conditions
frames = []
for c in conds:
    frames += sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", c, "*.png")))[:4]

sh, sv, nstd = 2.6, 2.4, 2.5
psf = gpsf(2.5)
rec_w, rec_n, wins = [], [], 0
print(f"detection recovery over crack frames (degrade blur h{sh}/v{sv} noise{nstd}):")
for fr in frames:
    rgb = cv2.cvtColor(cv2.imread(fr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape; top = int((H // INP) * INP * 0.40)
    pm = crack_prob(rgb[:((top // INP) * INP), :((W // INP) * INP)])
    if pm.max() < 0.4:
        continue
    ys, xs = np.where(pm > 0.4); cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - CH // 2, 0, pm.shape[0] - CH)); x0 = int(np.clip(cx - CW // 2, 0, W - CW))
    clean = rgb[y0:y0 + CH, x0:x0 + CW].astype(np.float64)
    fc = float(crack_prob(u8(clean)).mean())
    if fc < 1e-3:
        continue
    rng = np.random.RandomState(7)
    deg = np.stack([gaussian_filter(clean[..., c], sigma=(sv, sh)) for c in range(3)], -1) + rng.randn(CH, CW, 3) * nstd
    fd = float(crack_prob(u8(deg)).mean())
    fw = float(crack_prob(u8(np.stack([wiener(deg[..., c], psf) for c in range(3)], -1))).mean())
    fn = float(crack_prob(u8(naf_rgb(deg, sh, sv, nstd, CH, CW))).mean())
    rw = (fw - fd) / (fc - fd + 1e-9) * 100; rn = (fn - fd) / (fc - fd + 1e-9) * 100
    rec_w.append(rw); rec_n.append(rn); wins += int(rn > rw)
    print(f"  {os.path.basename(os.path.dirname(fr))[6:]:18s}/{os.path.basename(fr):16s} "
          f"clean {fc:.4f}  Wiener {rw:+5.0f}%  NAFNet {rn:+5.0f}%")

rec_w, rec_n = np.array(rec_w), np.array(rec_n)
print(f"\n  N={len(rec_n)} frames | detection recovery of lost fitness (mean +- std):")
print(f"    Wiener  {rec_w.mean():+5.0f}% +- {rec_w.std():.0f}")
print(f"    NAFNet  {rec_n.mean():+5.0f}% +- {rec_n.std():.0f}   | NAFNet > Wiener in {wins}/{len(rec_n)} frames")

# ---------- (2) MTF + PSNR statistics on the chart ROI ----------
CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64); Mg = 48
big = img[158 - Mg:554 + Mg, 937 - Mg:1133 + Mg]
inner = lambda a: a[Mg:Mg + 396, Mg:Mg + 196]
clean = inner(big); _, _, m_sharp = esf_mtf(clean)
psnr = lambda a, b: 10 * np.log10(255.0**2 / max(np.mean((np.clip(a, 0, 255) - b)**2), 1e-9))


def naf1(deg, sh, sv, nstd):
    xi = deg / 255.0; b = np.full_like(xi, 0.5 * (sh + sv) / BLUR_MAX); n = np.full_like(xi, nstd / NOISE_MAX)
    x = torch.from_numpy(np.stack([xi, b, n], 0)[None]).float().to(dev)
    Hh, Ww = x.shape[-2:]; ph, pw = (8 - Hh % 8) % 8, (8 - Ww % 8) % 8
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        return net(xp)[0, 0, :Hh, :Ww].cpu().numpy() * 255.0


mn, pn = [], []
for sb in [1.5, 2.0, 2.5, 3.0]:
    for seed in range(4):
        rng = np.random.RandomState(seed)
        deg = gaussian_filter(big, sigma=(sb * 0.9, sb)) + rng.randn(*big.shape) * 2.5
        r = naf1(deg, sb, sb * 0.9, 2.5); _, _, m = esf_mtf(inner(r))
        mn.append(m); pn.append(psnr(inner(r), clean))
print(f"\nMTF/PSNR over chart ROI (4 blur x 4 seeds = {len(mn)} runs), clean MTF50={m_sharp:.3f}:")
print(f"  NAFNet MTF50 {np.mean(mn):.3f} +- {np.std(mn):.3f}  ({np.mean(mn)/m_sharp*100:.0f}% of clean) | "
      f"PSNR {np.mean(pn):.1f} +- {np.std(pn):.1f} dB")
