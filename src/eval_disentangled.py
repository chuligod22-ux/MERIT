# -*- coding: utf-8 -*-
"""SP03 B — ablation: physics-disentangled (dual-branch) vs single MERIT, on MTF recovery + detection."""
import os, sys, glob, warnings
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet
from model_disentangled import DisentangledRestorer
warnings.filterwarnings("ignore")

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
dev = "cuda"
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16

INP = 448
tfm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
unet = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()

ck = torch.load(os.path.join(WEIGHTS, "restorer_task.pt"), map_location=dev)
merit = NAFNetUNet(width=ck["width"]).to(dev).eval(); merit.load_state_dict(ck["state"])
ckd = torch.load(os.path.join(WEIGHTS, "restorer_disentangled.pt"), map_location=dev)
disent = DisentangledRestorer(width=ckd["width"]).to(dev).eval(); disent.load_state_dict(ckd["state"])
u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
print(f"MERIT(single) {sum(p.numel() for p in merit.parameters())/1e6:.2f}M  | "
      f"Disentangled {sum(p.numel() for p in disent.parameters())/1e6:.2f}M\n")


def prep(im, sh, sv, nstd, sz):
    xi = np.clip(im, 0, 255) / 255.0
    b = np.full(sz, 0.5 * (sh + sv) / BLUR_MAX, np.float32); n = np.full(sz, nstd / NOISE_MAX, np.float32)
    return torch.from_numpy(np.stack([xi, b, n], 0)[None]).float().to(dev)


def restore(net, im, sh, sv, nstd, dual=False):
    x = prep(im, sh, sv, nstd, im.shape)
    H, W = x.shape[-2:]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        out = net(xp)
        out = out[0] if dual else out
    return out[0, 0, :H, :W].cpu().numpy() * 255.0


# ---- MTF sweep ----
CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64); Mg = 48
big = img[158 - Mg:554 + Mg, 937 - Mg:1133 + Mg]
inner = lambda a: a[Mg:Mg + 396, Mg:Mg + 196]
clean = inner(big); _, _, m_sharp = esf_mtf(clean)
psnr = lambda a, b: 10 * np.log10(255.0**2 / max(np.mean((np.clip(a, 0, 255) - b)**2), 1e-9))
ms, ps, md, pd = [], [], [], []
for sb in [1.5, 2.0, 2.5, 3.0]:
    for s in range(3):
        rng = np.random.RandomState(s); deg = gaussian_filter(big, (sb * 0.9, sb)) + rng.randn(*big.shape) * 2.5
        rs = restore(merit, deg, sb, sb * 0.9, 2.5); rd = restore(disent, deg, sb, sb * 0.9, 2.5, dual=True)
        _, _, a = esf_mtf(inner(rs)); _, _, b = esf_mtf(inner(rd))
        ms.append(a); ps.append(psnr(inner(rs), clean)); md.append(b); pd.append(psnr(inner(rd), clean))
print(f"MTF recovery (clean {m_sharp:.3f}, 12 runs):")
print(f"  MERIT(single)  MTF50 {np.mean(ms):.3f}+-{np.std(ms):.3f} ({np.mean(ms)/m_sharp*100:.0f}%)  PSNR {np.mean(ps):.1f}")
print(f"  Disentangled   MTF50 {np.mean(md):.3f}+-{np.std(md):.3f} ({np.mean(md)/m_sharp*100:.0f}%)  PSNR {np.mean(pd):.1f}")

# ---- detection ----
def crack_prob(rgb):
    H, W, _ = rgb.shape; pm = np.zeros((H, W), np.float32); t, l = [], []
    for yy in range(0, H - INP + 1, INP):
        for xx in range(0, W - INP + 1, INP):
            t.append(tfm(rgb[yy:yy + INP, xx:xx + INP])); l.append((xx, yy))
    with torch.no_grad():
        for i in range(0, len(t), 32):
            pr = torch.sigmoid(unet(torch.stack(t[i:i + 32]).cuda()))[:, 0].cpu().numpy()
            for j, (xx, yy) in enumerate(l[i:i + 32]):
                pm[yy:yy + INP, xx:xx + INP] = pr[j]
    return pm


def restore_rgb(net, deg, sh, sv, nstd, dual=False):
    CH, CW = deg.shape[:2]
    outs = []
    for c in range(3):
        outs.append(restore(net, deg[..., c], sh, sv, nstd, dual=dual))
    return np.clip(np.stack(outs, -1), 0, 255)


CH, CW = 896, 1344
conds = ["crack_d25_ISO200_V80", "crack_d35_ISO200_V60", "crack_d35_ISO200_V80",
         "crack_d45_ISO100_V60", "crack_d25_ISO100_V80"]
frames = []
for c in conds:
    frames += sorted(glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", c, "*.png")))[:4]
sh, sv, nstd = 2.6, 2.4, 2.5
rm, rd2, wins = [], [], 0
for fr in frames:
    rgb = cv2.cvtColor(cv2.imread(fr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape; top = int((H // INP) * INP * 0.40)
    pm = crack_prob(rgb[:((top // INP) * INP), :((W // INP) * INP)])
    if pm.max() < 0.4:
        continue
    ys, xs = np.where(pm > 0.4); cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - CH // 2, 0, pm.shape[0] - CH)); x0 = int(np.clip(cx - CW // 2, 0, W - CW))
    cl = rgb[y0:y0 + CH, x0:x0 + CW].astype(np.float64)
    fc = float(crack_prob(u8(cl)).mean())
    if fc < 1e-3:
        continue
    rng = np.random.RandomState(7)
    deg = np.stack([gaussian_filter(cl[..., c], (sv, sh)) for c in range(3)], -1) + rng.randn(CH, CW, 3) * nstd
    fd = float(crack_prob(u8(deg)).mean())
    fm = float(crack_prob(u8(restore_rgb(merit, deg, sh, sv, nstd))).mean())
    fr2 = float(crack_prob(u8(restore_rgb(disent, deg, sh, sv, nstd, dual=True))).mean())
    rm.append((fm - fd) / (fc - fd + 1e-9) * 100); rd2.append((fr2 - fd) / (fc - fd + 1e-9) * 100)
    wins += int(rd2[-1] > rm[-1])
rm, rd2 = np.array(rm), np.array(rd2)
print(f"\nDetection recovery (N={len(rm)}):")
print(f"  MERIT(single)  {rm.mean():+.0f}% +- {rm.std():.0f}")
print(f"  Disentangled   {rd2.mean():+.0f}% +- {rd2.std():.0f}   | disent>single in {wins}/{len(rm)}")
print("\nverdict: disentangled wins -> architecture contribution (B); else -> prompt already suffices (honest).")
