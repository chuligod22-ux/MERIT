# -*- coding: utf-8 -*-
"""
SP03 A (TN1) — Measurement-Perception-Distortion (MPD) triangle.

For each restoration method, measure THREE axes on the held-out chart ROI (degrade -> restore):
  Distortion  = PSNR to clean (higher better)         [classic fidelity]
  Perception  = LPIPS to clean (lower better)          [perceptual realism]
  Measurement = |MTF50_restored - MTF50_clean| (lower) [does it preserve the PHYSICAL quantity]
Thesis: generative methods optimize perception but hallucinate MTF (high measurement error);
regression denoisers optimize distortion but over-smooth MTF; MERIT targets the measurement axis.
"""
import os, glob, warnings
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, denoise_tv_chambolle, denoise_nl_means
from PIL import Image
import lpips as lpips_lib
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet
warnings.filterwarnings("ignore")

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
dev = "cuda"
lpf = lpips_lib.LPIPS(net="alex").to(dev).eval()

CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64)
Mg = 48
big = img[158 - Mg:554 + Mg, 937 - Mg:1133 + Mg]
inner = lambda a: a[Mg:Mg + 396, Mg:Mg + 196]
clean = inner(big); _, _, m_clean = esf_mtf(clean)

psnr = lambda a, b: 10 * np.log10(255.0**2 / max(np.mean((np.clip(a, 0, 255) - b)**2), 1e-9))


def lpips_d(a, b):
    def t(x):
        x = np.clip(x, 0, 255) / 127.5 - 1.0
        return torch.from_numpy(np.stack([x] * 3)[None]).float().to(dev)
    with torch.no_grad():
        return float(lpf(t(a), t(b)).item())


def gpsf2d(sh, sv):
    nh = int(2 * np.ceil(3 * sh) + 1); nv = int(2 * np.ceil(3 * sv) + 1)
    kh = np.exp(-(np.arange(nh) - nh // 2)**2 / (2 * sh**2)); kv = np.exp(-(np.arange(nv) - nv // 2)**2 / (2 * sv**2))
    k = np.outer(kv, kh); return k / k.sum()


def wiener_f(im, psf, K):
    pad = np.zeros_like(im); kh, kw = psf.shape; pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), (0, 1)); H = np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(im) * np.conj(H) / (np.abs(H)**2 + K)))


sh, sv, nstd = 2.6, 2.4, 2.5
rng = np.random.RandomState(5)
deg = gaussian_filter(big, sigma=(sv, sh)) + rng.randn(*big.shape) * nstd
psf = gpsf2d(sh, sv); d01 = np.clip(deg, 0, 255) / 255.0


def load_naf(name):
    ck = torch.load(os.path.join(DATA, name), map_location=dev)
    n = NAFNetUNet(width=ck["width"]).to(dev).eval(); n.load_state_dict(ck["state"]); return n


net_big, net_task = load_naf("restorer_big.pt"), load_naf("restorer_task.pt")


def naf(im, net):
    xi = np.clip(im, 0, 255) / 255.0
    b = np.full_like(xi, 0.5 * (sh + sv) / BLUR_MAX); nn = np.full_like(xi, nstd / NOISE_MAX)
    x = torch.from_numpy(np.stack([xi, b, nn], 0)[None]).float().to(dev)
    H, W = x.shape[-2:]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        return net(xp)[0, 0, :H, :W].cpu().numpy() * 255.0


def wiener_best(im, psf):
    best = None
    for K in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        r = wiener_f(im, psf, K); _, _, m = esf_mtf(inner(r))
        if best is None or m > best[1]:
            best = (r, m)
    return best[0]


from diffusers import StableDiffusionUpscalePipeline
print("loading SD x4 ...", flush=True)
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16).to(dev)
pipe.set_progress_bar_config(disable=True)
import cv2


def diffusion(im):
    lr = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).convert("RGB")
    with torch.no_grad():
        hr = pipe(prompt="", image=lr, num_inference_steps=50, guidance_scale=0).images[0]
    return cv2.resize(np.array(hr.convert("L")).astype(np.float64), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)


methods = [
    ("degraded", deg),
    ("Wiener (oracle)", wiener_best(deg, psf)),
    ("Richardson-Lucy", richardson_lucy(d01, psf / psf.sum(), num_iter=20) * 255.0),
    ("TV denoise", denoise_tv_chambolle(d01, weight=0.08) * 255.0),
    ("NL-means", denoise_nl_means(d01, patch_size=5, patch_distance=6, h=0.05) * 255.0),
    ("unsharp", deg + 1.5 * (deg - gaussian_filter(deg, 1.5))),
    ("SD-x4 diffusion", diffusion(big)),
    ("NAFNet-big", naf(deg, net_big)),
    ("MERIT (ours)", naf(deg, net_task)),
]

print(f"\nclean MTF50 = {m_clean:.3f}   degrade blur h{sh}/v{sv} noise{nstd}\n")
print(f"{'method':18s}{'Distortion':>12s}{'Perception':>12s}{'Measurement':>13s}")
print(f"{'':18s}{'PSNR(dB)up':>12s}{'LPIPS dn':>12s}{'MTFerr dn':>13s}")
rows = []
for name, im in methods:
    ii = inner(im); _, _, m = esf_mtf(ii)
    P, L, M = psnr(ii, clean), lpips_d(ii, clean), abs(m - m_clean)
    rows.append((name, P, L, M, m))
    print(f"  {name:16s}{P:12.1f}{L:12.4f}{M:13.4f}")

import csv
with open(os.path.join(DATA, "mpd_axes.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "PSNR", "LPIPS", "MTF_err", "MTF50"]); w.writerows(rows)
print(f"\nsaved mpd_axes.csv  | clean MTF50={m_clean:.3f}")
print("read: MERIT should sit best on Measurement (low MTFerr) while staying good on Distortion;")
print("generative (SD-x4) = decent Perception but high MTFerr (hallucinates the physical quantity).")
