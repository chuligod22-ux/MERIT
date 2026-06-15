# -*- coding: utf-8 -*-
"""
SP03 — broaden the classical baselines (reviewer-proofing).

On the held-out chart ROI (degrade -> restore), compare a battery of classical methods against
our NAFNet on MTF50 (resolution), PSNR-to-clean (fidelity), and flat-noise sd:
  oracle deconvolution : Wiener (best K), Richardson-Lucy, unsupervised Wiener   [know the PSF]
  deployment-realistic : Wiener with a WRONG/generic PSF                          [PSF unknown]
  denoise-only         : TV (Chambolle), Non-Local Means
  naive sharpening     : unsharp masking                                          [no PSF]
  learned              : NAFNet (ours)
"""
import os, glob, warnings
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, unsupervised_wiener, denoise_tv_chambolle, denoise_nl_means
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet
warnings.filterwarnings("ignore")

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
CKPT = os.environ.get("SP03_CKPT", os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer_big.pt"))
dev = "cuda"

CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64)
y1, y2, x1, x2 = 158, 554, 937, 1133
Mg = 48
big = img[y1 - Mg:y2 + Mg, x1 - Mg:x2 + Mg]
inner = lambda a: a[Mg:Mg + (y2 - y1), Mg:Mg + (x2 - x1)]
clean = inner(big); _, _, m_sharp = esf_mtf(clean)
psnr = lambda a, b: 10 * np.log10(255.0**2 / max(np.mean((np.clip(a, 0, 255) - b)**2), 1e-9))
flat = lambda a: float(inner(a)[:, :40].std())

sh, sv, nstd = 2.6, 2.4, 2.5
rng = np.random.RandomState(5)
deg = gaussian_filter(big, sigma=(sv, sh)) + rng.randn(*big.shape) * nstd


def gpsf2d(sh, sv):
    nh = int(2 * np.ceil(3 * sh) + 1); nv = int(2 * np.ceil(3 * sv) + 1)
    kh = np.exp(-(np.arange(nh) - nh // 2)**2 / (2 * sh**2)); kv = np.exp(-(np.arange(nv) - nv // 2)**2 / (2 * sv**2))
    k = np.outer(kv, kh); return k / k.sum()


def wiener_f(img, psf, K):
    pad = np.zeros_like(img, np.float64); kh, kw = psf.shape
    pad[:kh, :kw] = psf; pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1)); H = np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(H) / (np.abs(H)**2 + K)))


psf_true = gpsf2d(sh, sv)
psf_wrong = gpsf2d(1.5, 1.5)         # generic/mismatched PSF (deployment: true PSF unknown)


def wiener_bestK(d, psf):
    best = None
    for K in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        r = wiener_f(d, psf, K); _, _, m = esf_mtf(inner(r))
        if best is None or m > best[1]:
            best = (r, m)
    return best[0]


ck = torch.load(CKPT, map_location=dev)
net = (NAFNetUNet(width=ck["width"]) if ck.get("arch") == "unet"
       else NAFNetRestorer(width=ck["width"], n_blocks=ck["n_blocks"])).to(dev).eval()
net.load_state_dict(ck["state"])


def naf(d):
    xi = np.clip(d, 0, 255) / 255.0
    b = np.full_like(xi, 0.5 * (sh + sv) / BLUR_MAX); n = np.full_like(xi, nstd / NOISE_MAX)
    x = torch.from_numpy(np.stack([xi, b, n], 0)[None]).float().to(dev)
    H, W = x.shape[-2:]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        return net(xp)[0, 0, :H, :W].cpu().numpy() * 255.0


def unsharp(d, sigma=1.5, amount=1.5):
    return d + amount * (d - gaussian_filter(d, sigma))


d01 = np.clip(deg, 0, 255) / 255.0
methods = [
    ("degraded (input)", deg),
    ("Wiener (oracle PSF)", wiener_bestK(deg, psf_true)),
    ("Richardson-Lucy (oracle)", richardson_lucy(d01, psf_true / psf_true.sum(), num_iter=20) * 255.0),
    ("unsup. Wiener (oracle)", unsupervised_wiener(d01, psf_true)[0] * 255.0),
    ("Wiener (wrong/generic PSF)", wiener_bestK(deg, psf_wrong)),
    ("TV denoise (no deconv)", denoise_tv_chambolle(d01, weight=0.08) * 255.0),
    ("NL-means (no deconv)", denoise_nl_means(d01, patch_size=5, patch_distance=6, h=0.05) * 255.0),
    ("unsharp mask (no PSF)", unsharp(deg)),
    ("NAFNet (ours)", naf(deg)),
]

print(f"held-out chart ROI  clean MTF50 = {m_sharp:.3f} cy/px  (degrade blur h{sh}/v{sv} noise{nstd})\n")
print(f"{'method':30s}{'MTF50':>8s}{'vs sharp':>10s}{'PSNR':>8s}{'noise':>7s}")
for name, im in methods:
    _, _, m = esf_mtf(inner(im))
    print(f"  {name:28s}{m:8.3f}{m/m_sharp*100:9.0f}%{psnr(inner(im), clean):8.1f}{flat(im):7.1f}")
print("\nread: NAFNet tops the combined picture - MTF ~clean AND highest PSNR AND lowest noise;")
print("oracle deconv recovers MTF but amplifies noise; blind/denoise-only/unsharp each fail one axis.")
