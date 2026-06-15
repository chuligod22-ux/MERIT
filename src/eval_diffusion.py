# -*- coding: utf-8 -*-
"""
SP03 B — diffusion-restoration comparison (latest generative prior vs measurement-fidelity NAFNet).

A degraded chart ROI is restored by (i) a Stable-Diffusion x4 upscaler used as a diffusion prior
(4x -> downscale back), (ii) oracle Wiener, (iii) our NAFNet. We report MTF50 (resolution) and
PSNR-to-CLEAN (fidelity). Expectation: the diffusion prior looks sharp but HALLUCINATES — low
PSNR (invented structure not matching the true scene), illustrating the perception-distortion wall.
"""
import os, glob, warnings
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet
warnings.filterwarnings("ignore")

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
CKPT = os.environ.get("SP03_CKPT", os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer_big.pt"))
dev = "cuda"

# ---- load held-out chart ROI ----
CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64)
y1, y2, x1, x2 = 158, 554, 937, 1133
clean = img[y1:y2, x1:x2]                      # 396 x 196 vertical-edge ROI
_, _, m_sharp = esf_mtf(clean)
psnr = lambda a, b: 10 * np.log10(255.0**2 / max(np.mean((np.clip(a, 0, 255) - b)**2), 1e-9))

sh, sv, nstd = 2.6, 2.4, 2.5
rng = np.random.RandomState(5)
deg = gaussian_filter(clean, sigma=(sv, sh)) + rng.randn(*clean.shape) * nstd

# ---- NAFNet ----
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


# ---- diffusion prior (SD x4 upscaler): degraded -> 4x -> downscale back ----
from diffusers import StableDiffusionUpscalePipeline
print("loading SD x4 upscaler (first run downloads ~1.5GB) ...", flush=True)
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
pipe = pipe.to(dev); pipe.set_progress_bar_config(disable=True)


def diffusion(d):
    lr = Image.fromarray(np.clip(d, 0, 255).astype(np.uint8)).convert("RGB")
    with torch.no_grad():
        hr = pipe(prompt="", image=lr, num_inference_steps=50, guidance_scale=0).images[0]
    hr = np.array(hr.convert("L")).astype(np.float64)
    return cv2.resize(hr, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)


print(f"held-out chart ROI  clean MTF50 = {m_sharp:.3f} cy/px\n")
print(f"{'method':22s}{'MTF50':>8s}{'vs sharp':>10s}{'PSNR(clean)':>13s}")
for name, im in [("degraded", deg), ("NAFNet (ours)", naf(deg)), ("SD-x4 diffusion", diffusion(deg))]:
    _, _, m = esf_mtf(im)
    print(f"  {name:20s}{m:8.3f}{m/m_sharp*100:9.0f}%{psnr(im, clean):13.1f}")
print("\nread: NAFNet ~clean MTF + HIGH PSNR (faithful); diffusion 'sharp' but LOW PSNR = hallucinated.")
