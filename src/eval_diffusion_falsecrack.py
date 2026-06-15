# -*- coding: utf-8 -*-
"""
SP03 B+ — diffusion false-crack test (the crispest hallucination evidence).

On a CRACK-FREE concrete region, degrade then restore with NAFNet vs the SD x4 diffusion prior;
the segmenter scores crack response. A faithful restorer keeps it ~clean (no invented cracks); a
generative prior may synthesize plausible crack-like texture -> false detections.
"""
import os, sys, glob, warnings
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetRestorer, NAFNetUNet
warnings.filterwarnings("ignore")

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


def crack_pm(rgb448):
    with torch.no_grad():
        p = torch.sigmoid(unet(tfm(rgb448).unsqueeze(0).cuda()))[0, 0].cpu().numpy()
    return p


# ---- crack-free 448 crop ----
frame = sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", "crack_d25_ISO100_V60", "*.png")))[0]
rgb = cv2.cvtColor(cv2.imread(frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
H, W, _ = rgb.shape; topR = int((H // INP) * INP * 0.40)
best = None
for y0 in range(0, topR - INP + 1, INP):
    for x0 in range(0, W - INP + 1, INP):
        sub = rgb[y0:y0 + INP, x0:x0 + INP]
        m = crack_pm(sub).max()
        if best is None or m < best[0]:
            best = (float(m), y0, x0)
_, y0, x0 = best
clean = rgb[y0:y0 + INP, x0:x0 + INP].astype(np.float64)
print(f"crack-free 448 crop at ({y0},{x0}), clean max crack-prob = {best[0]:.3f}\n")

sh, sv, nstd = 2.6, 2.4, 2.5
rng = np.random.RandomState(6)
deg = np.stack([gaussian_filter(clean[..., c], sigma=(sv, sh)) for c in range(3)], -1) + rng.randn(INP, INP, 3) * nstd


def naf_rgb(d):
    x = d / 255.0
    b = np.full((INP, INP), 0.5 * (sh + sv) / BLUR_MAX, np.float32); n = np.full((INP, INP), nstd / NOISE_MAX, np.float32)
    batch = np.stack([np.stack([x[..., c], b, n], 0) for c in range(3)], 0)
    with torch.no_grad():
        out = net(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:, 0]
    return np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0, 255)


from diffusers import StableDiffusionUpscalePipeline
print("loading SD x4 upscaler ...", flush=True)
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16).to(dev)
pipe.set_progress_bar_config(disable=True)


def diffusion_rgb(d):
    lr = Image.fromarray(u8(d))
    with torch.no_grad():
        hr = pipe(prompt="", image=lr, num_inference_steps=50, guidance_scale=0).images[0]
    return cv2.resize(np.array(hr).astype(np.float64), (INP, INP), interpolation=cv2.INTER_AREA)


print(f"{'image':18s}{'mean prob':>11s}{'max prob':>10s}{'frac>0.5':>10s}")
for name, im in [("clean", clean), ("degraded", deg), ("NAFNet (ours)", naf_rgb(deg)),
                 ("SD-x4 diffusion", diffusion_rgb(deg))]:
    p = crack_pm(u8(im))
    print(f"  {name:16s}{p.mean():11.5f}{p.max():10.3f}{(p>0.5).mean()*100:9.3f}%")
print("\nread: crack-free -> all should stay ~clean. Invented crack texture raises max/frac (hallucination).")
