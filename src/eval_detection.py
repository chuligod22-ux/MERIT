# -*- coding: utf-8 -*-
"""
SP03 detection-axis evaluation — does restoration recover crack DETECTION fitness?

A clean cam2 frame (good detection) is degraded (blur+noise like a far/high-ISO condition),
then restored with classical Wiener vs the trained NAFNet. The pretrained crack segmenter
(UNet16-VGG16, khanhha — the same instrument SP02 used as ground truth) scores detection
fitness on a crack-region crop for clean / degraded / Wiener / NAFNet.
"""
import os, sys, glob
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
CKPT = os.environ.get("SP03_CKPT", os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer.pt"))
dev = "cuda"
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16

INP = 448
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tfm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
unet = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()


def crack_prob(rgb):
    """Per-pixel crack probability map over the tiled crop (rgb uint8)."""
    H, W, _ = rgb.shape
    pm = np.zeros((H, W), np.float32)
    tens, locs = [], []
    for yy in range(0, H - INP + 1, INP):
        for xx in range(0, W - INP + 1, INP):
            tens.append(tfm(rgb[yy:yy + INP, xx:xx + INP])); locs.append((xx, yy))
    with torch.no_grad():
        for i in range(0, len(tens), 32):
            pr = torch.sigmoid(unet(torch.stack(tens[i:i + 32]).cuda()))[:, 0].cpu().numpy()
            for j, (xx, yy) in enumerate(locs[i:i + 32]):
                pm[yy:yy + INP, xx:xx + INP] = pr[j]
    return pm


def fitness(rgb):
    return float(crack_prob(rgb).mean())


# ---- pick a clean, well-detected cam2 frame and crop around its crack ----
import pandas as pd
cq = pd.read_csv(os.path.join(NRIQA, "04_data", "composite_Q.csv"))
sel = cq[cq.cond == "crack_d25_ISO100_V60"].sort_values("fit_mean_top", ascending=False).iloc[0]
frame = os.path.join(NRIQA, "04_data", "raw", "cam2", sel.cond, sel.frame)
bgr = cv2.imread(frame, cv2.IMREAD_COLOR); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W, _ = rgb.shape
top = int((H // INP) * INP * 0.40)
pm0 = crack_prob(rgb[:((top // INP) * INP), :((W // INP) * INP)])
ys, xs = np.where(pm0 > 0.4)
cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) > 20 else np.unravel_index(pm0.argmax(), pm0.shape)
CH, CW = 1344, 1792                       # multiple of 448
y0 = int(np.clip(cy - CH // 2, 0, pm0.shape[0] - CH)); x0 = int(np.clip(cx - CW // 2, 0, W - CW))
clean = rgb[y0:y0 + CH, x0:x0 + CW].astype(np.float64)
print(f"clean frame {sel.cond}/{sel.frame}  crop ({CH}x{CW}) around crack at ({cy},{cx})")

# ---- restorers ----
ck = torch.load(CKPT, map_location=dev)
net = (NAFNetUNet(width=ck["width"]) if ck.get("arch") == "unet"
       else NAFNetRestorer(width=ck["width"], n_blocks=ck["n_blocks"])).to(dev).eval()
net.load_state_dict(ck["state"])


def gpsf2d(sh, sv):
    nh = int(2 * np.ceil(3 * sh) + 1); nv = int(2 * np.ceil(3 * sv) + 1)
    kh = np.exp(-(np.arange(nh) - nh // 2)**2 / (2 * sh**2)); kv = np.exp(-(np.arange(nv) - nv // 2)**2 / (2 * sv**2))
    k = np.outer(kv, kh); return k / k.sum()


def wiener_ch(img, psf, K):
    pad = np.zeros_like(img, np.float64); kh, kw = psf.shape
    pad[:kh, :kw] = psf; pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    Hf = np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(Hf) / (np.abs(Hf)**2 + K)))


def nafnet_rgb(deg, sh, sv, nstd):
    x = deg / 255.0
    bmap = np.full((CH, CW), 0.5 * (sh + sv) / BLUR_MAX, np.float32)
    nmap = np.full((CH, CW), nstd / NOISE_MAX, np.float32)
    batch = np.stack([np.stack([x[..., c], bmap, nmap], 0) for c in range(3)], 0)
    with torch.no_grad():
        out = net(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:, 0]   # (3,H,W)
    return np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0, 255)


sh, sv, nstd = 2.6, 2.4, 2.5
rng = np.random.RandomState(2)
deg = np.stack([gaussian_filter(clean[..., c], sigma=(sv, sh)) for c in range(3)], -1)
deg = deg + rng.randn(*deg.shape) * nstd
psf = gpsf2d(sh, sv)
wie = np.stack([wiener_ch(deg[..., c], psf, 1e-2) for c in range(3)], -1)
naf = nafnet_rgb(deg, sh, sv, nstd)


def u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


print(f"\ndegradation: blur h{sh}/v{sv}, noise {nstd}")
print(f"{'image':16s}{'detection fitness (mean crack prob)':>36s}")
fc = fitness(u8(clean)); print(f"  {'clean (ceiling)':16s}{fc:36.5f}")
fd = fitness(u8(deg));   print(f"  {'degraded':16s}{fd:36.5f}   ({fd/fc*100:.0f}% of clean)")
fw = fitness(u8(wie));   print(f"  {'Wiener':16s}{fw:36.5f}   ({fw/fc*100:.0f}% of clean)")
fn = fitness(u8(naf));   print(f"  {'NAFNet (ours)':16s}{fn:36.5f}   ({fn/fc*100:.0f}% of clean)")
print(f"\n  recovery of lost detection:  Wiener {(fw-fd)/(fc-fd+1e-9)*100:+.0f}%   "
      f"NAFNet {(fn-fd)/(fc-fd+1e-9)*100:+.0f}%")
