# -*- coding: utf-8 -*-
"""
SP03 A — false-crack hallucination test (the operationally critical fidelity check).

On a CRACK-FREE region (segmenter ~0 on clean), degrade then restore; the restorer must NOT
invent cracks. We report the segmenter response for clean / degraded / Wiener / NAFNet — a faithful
restorer keeps it near the clean (~0) level; hallucination would raise it.
"""
import os, sys, glob
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
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


CH, CW = 1344, 1792
# pick a clean cam2 frame and find the most CRACK-FREE window (lowest max prob)
frame = sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", "crack_d25_ISO100_V60", "*.png")))[0]
rgb = cv2.cvtColor(cv2.imread(frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
H, W, _ = rgb.shape
top = int((H // INP) * INP * 0.40)
pm = crack_prob(rgb[:((top // INP) * INP), :((W // INP) * INP)])
best = None
for y0 in range(0, pm.shape[0] - CH + 1, INP):
    for x0 in range(0, W - CW + 1, INP * 2):
        sub = pm[y0:y0 + CH, x0:x0 + CW]
        if best is None or sub.max() < best[0]:
            best = (float(sub.max()), y0, x0)
_, y0, x0 = best
clean = rgb[y0:y0 + CH, x0:x0 + CW].astype(np.float64)
print(f"crack-free crop at ({y0},{x0}), clean max crack-prob = {best[0]:.3f}\n")


def gpsf2d(s):
    n = int(2 * np.ceil(3 * s) + 1); k1 = np.exp(-(np.arange(n) - n // 2)**2 / (2 * s**2))
    k = np.outer(k1, k1); return k / k.sum()


def wiener_ch(img, psf, K):
    pad = np.zeros_like(img); kh, kw = psf.shape; pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1)); Hf = np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(Hf) / (np.abs(Hf)**2 + K)))


def naf_rgb(deg, sh, sv, nstd):
    x = deg / 255.0
    b = np.full((CH, CW), 0.5 * (sh + sv) / BLUR_MAX, np.float32); n = np.full((CH, CW), nstd / NOISE_MAX, np.float32)
    batch = np.stack([np.stack([x[..., c], b, n], 0) for c in range(3)], 0)
    with torch.no_grad():
        out = net(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:, 0]
    return np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0, 255)


sh, sv, nstd = 2.6, 2.4, 2.5
rng = np.random.RandomState(4)
deg = np.stack([gaussian_filter(clean[..., c], sigma=(sv, sh)) for c in range(3)], -1) + rng.randn(CH, CW, 3) * nstd
psf = gpsf2d(2.5)
wie = np.stack([wiener_ch(deg[..., c], psf, 1e-2) for c in range(3)], -1)
nafr = naf_rgb(deg, sh, sv, nstd)
u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)


def resp(rgb):
    pm = crack_prob(u8(rgb))
    return float(pm.mean()), float((pm > 0.5).mean())


print(f"{'image':18s}{'mean crack-prob':>16s}{'frac>0.5':>10s}   (crack-free: all should stay ~clean)")
for name, im in [("clean", clean), ("degraded", deg), ("Wiener", wie), ("NAFNet (ours)", nafr)]:
    m, f = resp(im)
    print(f"  {name:16s}{m:16.5f}{f*100:9.3f}%")
print("\nhallucination read: NAFNet mean/frac must NOT exceed clean -> no invented cracks.")
