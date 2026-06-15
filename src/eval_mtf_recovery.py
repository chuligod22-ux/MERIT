# -*- coding: utf-8 -*-
"""
SP03 decisive evaluation — MTF50 recovery on the HELD-OUT cam1 chart (frame_000102).

Degrade the sharp chart ROI (anisotropic blur + sensor noise), then restore with
  (a) classical Wiener deconvolution (true PSF)  vs  (b) the trained NAFNet (measured prompt),
and measure ISO-12233 e-SFR MTF50 for each. The learned restorer should recover MTF without
amplifying noise, where classical Wiener cannot.
"""
import os, glob
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
CKPT = os.environ.get("SP03_CKPT", os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer.pt"))
dev = "cuda" if torch.cuda.is_available() else "cpu"


def gaussian_psf2d(sh, sv):
    nh = int(2 * np.ceil(3 * max(sh, 0.3)) + 1); nv = int(2 * np.ceil(3 * max(sv, 0.3)) + 1)
    ax_h = np.arange(nh) - nh // 2; ax_v = np.arange(nv) - nv // 2
    kh = np.exp(-ax_h**2 / (2 * max(sh, 1e-3)**2)); kv = np.exp(-ax_v**2 / (2 * max(sv, 1e-3)**2))
    k = np.outer(kv, kh); return k / k.sum()


def wiener(img, psf, K):
    pad = np.zeros_like(img, np.float64); kh, kw = psf.shape
    pad[:kh, :kw] = psf; pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    H = np.fft.fft2(pad); G = np.fft.fft2(img)
    return np.real(np.fft.ifft2(G * np.conj(H) / (np.abs(H)**2 + K)))


# ---- load held-out chart ROI (the MTF-crop, never used in training) with context margin ----
CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64)
x1, y1, x2, y2 = 937, 158, 1133, 554
Mg = 48
big = img[y1 - Mg:y2 + Mg, x1 - Mg:x2 + Mg]
iy, ix, hh, ww = Mg, Mg, y2 - y1, x2 - x1
inner = lambda a: a[iy:iy + hh, ix:ix + ww]
flat = lambda a: float(inner(a)[:, :40].std())
_, _, m_sharp = esf_mtf(inner(big))

# ---- load trained NAFNet ----
ck = torch.load(CKPT, map_location=dev)
net = (NAFNetUNet(width=ck["width"]) if ck.get("arch") == "unet"
       else NAFNetRestorer(width=ck["width"], n_blocks=ck["n_blocks"])).to(dev).eval()
net.load_state_dict(ck["state"])


def restore_nafnet(deg, sh, sv, nstd):
    x_img = np.clip(deg, 0, 255) / 255.0
    bmap = np.full_like(x_img, 0.5 * (sh + sv) / BLUR_MAX)
    nmap = np.full_like(x_img, nstd / NOISE_MAX)
    x = torch.from_numpy(np.stack([x_img, bmap, nmap], 0)[None]).float().to(dev)
    H, W = x.shape[-2:]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8        # encoder-decoder needs multiple-of-8
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        out = net(xp)[0, 0, :H, :W].cpu().numpy()
    return out * 255.0


print(f"held-out chart ROI  sharp MTF50 = {m_sharp:.3f} cy/px\n")
print(f"{'degradation':22s}{'MTF50':>8s}{'vs sharp':>10s}{'flat-noise sd':>14s}")
rng = np.random.RandomState(1)
for sh, sv, nstd in [(1.8, 1.6, 2.5), (2.6, 2.4, 2.5)]:
    deg = gaussian_filter(big, sigma=(sv, sh)) + rng.randn(*big.shape) * nstd
    _, _, m_deg = esf_mtf(inner(deg))
    psf = gaussian_psf2d(sh, sv)
    # Wiener: best K by sweep (most generous to the baseline)
    mw, fw = -1, None
    for K in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        r = wiener(deg, psf, K); _, _, mr = esf_mtf(inner(r))
        if mr > mw:
            mw, fw = mr, flat(r)
    rest_n = restore_nafnet(deg, sh, sv, nstd)
    _, _, m_naf = esf_mtf(inner(rest_n))
    print(f"\n  blur(h{sh},v{sv}) noise{nstd}")
    print(f"    {'degraded':18s}{m_deg:8.3f}{m_deg/m_sharp*100:9.0f}%{flat(deg):14.1f}")
    print(f"    {'Wiener (best K)':18s}{mw:8.3f}{mw/m_sharp*100:9.0f}%{fw:14.1f}")
    print(f"    {'NAFNet (ours)':18s}{m_naf:8.3f}{m_naf/m_sharp*100:9.0f}%{flat(rest_n):14.1f}")
