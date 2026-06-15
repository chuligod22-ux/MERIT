# -*- coding: utf-8 -*-
"""
SP03 A — broaden the MTF eval + characterize the 106% (fidelity, not hallucination).

Sweep degradation severity on the held-out chart ROI; for each, restore with Wiener (oracle PSF)
and NAFNet and report BOTH MTF50 (resolution recovery) AND PSNR-to-CLEAN (fidelity). A faithful
restorer recovers MTF50 toward the CLEAN level (~0.077) with HIGH PSNR; over-sharpening/hallucination
would push MTF50 well above clean and/or lower PSNR.
"""
import os, glob
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
CKPT = os.environ.get("SP03_CKPT", os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer_big.pt"))
dev = "cuda"


def gpsf2d(sh, sv):
    nh = int(2 * np.ceil(3 * sh) + 1); nv = int(2 * np.ceil(3 * sv) + 1)
    kh = np.exp(-(np.arange(nh) - nh // 2)**2 / (2 * sh**2)); kv = np.exp(-(np.arange(nv) - nv // 2)**2 / (2 * sv**2))
    k = np.outer(kv, kh); return k / k.sum()


def wiener(img, psf, K):
    pad = np.zeros_like(img, np.float64); kh, kw = psf.shape
    pad[:kh, :kw] = psf; pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    H = np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(H) / (np.abs(H)**2 + K)))


def psnr(a, b):
    mse = np.mean((np.clip(a, 0, 255) - np.clip(b, 0, 255))**2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0**2 / mse)


CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64)
x1, y1, x2, y2 = 937, 158, 1133, 554
Mg = 48
big = img[y1 - Mg:y2 + Mg, x1 - Mg:x2 + Mg]
iy, ix, hh, ww = Mg, Mg, y2 - y1, x2 - x1
inner = lambda a: a[iy:iy + hh, ix:ix + ww]
clean = inner(big)
_, _, m_sharp = esf_mtf(clean)

ck = torch.load(CKPT, map_location=dev)
net = (NAFNetUNet(width=ck["width"]) if ck.get("arch") == "unet"
       else NAFNetRestorer(width=ck["width"], n_blocks=ck["n_blocks"])).to(dev).eval()
net.load_state_dict(ck["state"])


def naf(deg, sh, sv, nstd):
    xi = np.clip(deg, 0, 255) / 255.0
    b = np.full_like(xi, 0.5 * (sh + sv) / BLUR_MAX); n = np.full_like(xi, nstd / NOISE_MAX)
    x = torch.from_numpy(np.stack([xi, b, n], 0)[None]).float().to(dev)
    H, W = x.shape[-2:]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        return net(xp)[0, 0, :H, :W].cpu().numpy() * 255.0


print(f"held-out chart ROI  clean(sharp) MTF50 = {m_sharp:.3f} cy/px   [model: {os.path.basename(CKPT)}]")
print(f"\n{'blur sig':>9s} | {'degraded':>16s} | {'Wiener (oracle)':>20s} | {'NAFNet (ours)':>20s}")
print(f"{'':9s} | {'MTF50':>7s}{'PSNR':>9s} | {'MTF50':>7s}{'PSNR':>9s}{'noise':>4s} | {'MTF50':>7s}{'PSNR':>9s}{'noise':>4s}")
rng = np.random.RandomState(3)
for sb in [1.0, 1.5, 2.0, 2.5, 3.0]:
    sh, sv, nstd = sb, sb * 0.9, 2.5
    deg = gaussian_filter(big, sigma=(sv, sh)) + rng.randn(*big.shape) * nstd
    _, _, md = esf_mtf(inner(deg))
    psf = gpsf2d(sh, sv)
    mw, pw_, fw = -1, 0, 0
    for K in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        r = wiener(deg, psf, K); _, _, mr = esf_mtf(inner(r))
        if mr > mw:
            mw, pw_, fw = mr, psnr(inner(r), clean), inner(r)[:, :40].std()
    rn = naf(deg, sh, sv, nstd); _, _, mn = esf_mtf(inner(rn))
    print(f"{sb:9.1f} | {md:7.3f}{psnr(inner(deg),clean):9.1f} | "
          f"{mw:7.3f}{pw_:9.1f}{fw:4.1f} | {mn:7.3f}{psnr(inner(rn),clean):9.1f}{inner(rn)[:,:40].std():4.1f}")
print(f"\nfidelity read: NAFNet MTF50 should track clean ({m_sharp:.3f}) with HIGH PSNR;")
print("over-sharpening/hallucination would show MTF50 >> clean and/or low PSNR.")
